"""Agent Zoo userspace plugin for LiteLLM/provider rate-limit retry deadlines.

This module is intentionally self-contained so it can be installed as an
Agent Zoo source-file plugin via ``azo-plugin install <repo-or-git-url>``.
It observes generic harness runtime events and only overrides default retry
behavior when it can confidently identify a rate-limit error and parse a retry
or reset deadline.
"""

from __future__ import annotations

import datetime as _dt
import email.utils
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

from agent_utils import Feature
from agent_utils.components import HarnessEventHandler
from agent_utils.harness_events import HarnessControl, HarnessDecision, HarnessEffect, WakeSpec

PLUGIN_NAME = "litellm_rate_limit_retry"
STATE_ATTR = "litellm_rate_limit_retry_state"
DEFAULT_MAX_DELAY_SECONDS = 20 * 60
DEFAULT_TICK_SECONDS = 1.0
DEFAULT_MAX_CAPTURES = 64
DEFAULT_EMIT_ONE_SHOT_NOTICE = False

_RETRY_AFTER_KEYS = {"retry-after", "retry_after", "retryafter"}
_RESET_HEADER_PREFIXES = (
    "x_ratelimit_reset",
    "x_rate_limit_reset",
    "ratelimit_reset",
    "rate_limit_reset",
)
_BODY_DEADLINE_KEYS = {
    "retry_after",
    "retryafter",
    "retry_after_seconds",
    "retryafterseconds",
    "retry_after_ms",
    "retryafterms",
    "reset_at",
    "resetat",
    "reset_time",
    "resettime",
    "rate_limit_reset_at",
    "ratelimitresetat",
    "rate_limit_reset_time",
    "ratelimitresettime",
}
_RATE_LIMIT_TEXT_MARKERS = (
    "rate limit",
    "ratelimit",
    "rate_limit",
    "too many requests",
    "too-many-requests",
    "429",
    "x-ratelimit",
    "requests per minute",
    "tokens per minute",
    "rpm",
    "tpm",
)
_PROVIDER_TEXT_MARKERS = (
    "litellm",
    "openai",
    "anthropic",
    "azure openai",
    "bedrock",
    "gemini",
    "vertex",
    "provider",
    "llm.provider",
)
_CLASS_MARKERS = (
    "ratelimit",
    "rate_limit",
    "toomanyrequests",
    "too_many_requests",
    "too-many-requests",
)
_DURATION_TOKEN_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>milliseconds?|msecs?|ms|seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)",
    re.IGNORECASE,
)
_TEXT_DURATION_PATTERNS = (
    re.compile(r"\bretry\s*-?\s*after\s*:?[\s=]+(?P<value>[^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\btry\s+again\s+in\s+(?P<value>[^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\breset\s+in\s+(?P<value>[^.;,\n]+)", re.IGNORECASE),
)
_TEXT_DATE_PATTERNS = (
    re.compile(r"\breset\s+at\s+(?P<value>[^;\n]+)", re.IGNORECASE),
    re.compile(r"\bretry\s*-?\s*after\s+(?P<value>[A-Z][a-z]{2},\s+[^;\n]+)", re.IGNORECASE),
)


@dataclass(frozen=True)
class DeadlineParse:
    """A successfully parsed runtime-clock retry deadline."""

    deadline: float
    delay_seconds: float
    source: str
    signal: str


def register_features(builder, *, session, config):
    """Register the common-scope harness event handler feature."""

    cfg = _plugin_config(config)
    builder.add(
        Feature(
            name=PLUGIN_NAME,
            components=[LiteLLMRateLimitRetryHandler(cfg)],
            order=[HarnessEventHandler],
        )
    )


class LiteLLMRateLimitRetryHandler(HarnessEventHandler):
    """Capture LiteLLM/provider rate-limit errors and drive retry countdown UX."""

    event_kinds = {"runtime.error", "runtime.wait.tick"}
    init = {STATE_ATTR: dict}
    optional_reads = {STATE_ATTR}
    writes = {STATE_ATTR}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        self.max_delay_seconds = _positive_float(cfg.get("max_delay_seconds"), DEFAULT_MAX_DELAY_SECONDS)
        self.tick_seconds = _positive_float(cfg.get("tick_seconds"), DEFAULT_TICK_SECONDS)
        self.max_captures = max(1, int(_positive_float(cfg.get("max_captures"), DEFAULT_MAX_CAPTURES)))
        self.emit_one_shot_notice = bool(cfg.get("emit_one_shot_notice", DEFAULT_EMIT_ONE_SHOT_NOTICE))

    def handle_harness_event(self, event, state):
        if getattr(event, "kind", "") == "runtime.error":
            return self._handle_runtime_error(event, state)
        if getattr(event, "kind", "") == "runtime.wait.tick":
            return self._handle_wait_tick(event, state)
        return None

    def _handle_runtime_error(self, event, state):
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            return None

        parsed = parse_rate_limit_deadline(payload, max_delay_seconds=self.max_delay_seconds)
        if parsed is None:
            return None

        source_event_id = str(getattr(event, "id", "") or "")
        store = _state_store(state)
        captures = store.setdefault("captures", {})
        now = _runtime_now(payload)
        attempt = _safe_int(payload.get("attempt"))
        captures[source_event_id] = {
            "source_event_id": source_event_id,
            "deadline": parsed.deadline,
            "delay_seconds": parsed.delay_seconds,
            "captured_runtime": now,
            "attempt": attempt,
            "source": parsed.source,
            "signal": parsed.signal,
            "wait_ids": [],
        }
        _prune_captures(captures, self.max_captures)

        text = _status_text(parsed.delay_seconds)
        effects = [HarnessEffect("status", {"active": True, "text": text})]
        if self.emit_one_shot_notice:
            effects.append(
                HarnessEffect(
                    "notice",
                    {
                        "level": "warning",
                        "text": f"LiteLLM/provider rate limit detected; retrying in {math.ceil(parsed.delay_seconds)}s.",
                    },
                )
            )

        return HarnessDecision(
            control=HarnessControl.RETRY,
            wake=WakeSpec(at=parsed.deadline, tick_seconds=self.tick_seconds, reason="rate limit retry deadline"),
            effects=effects,
            reason="LiteLLM/provider rate limit retry deadline",
            payload={
                "plugin": PLUGIN_NAME,
                "source_event_id": source_event_id,
                "deadline": parsed.deadline,
                "delay_seconds": parsed.delay_seconds,
                "source": parsed.source,
                "signal": parsed.signal,
            },
        )

    def _handle_wait_tick(self, event, state):
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            return None
        source_event_id = str(payload.get("source_event_id") or "")
        if not source_event_id:
            return None

        store = _state_store(state)
        capture = (store.get("captures") or {}).get(source_event_id)
        if not isinstance(capture, dict):
            return None

        wait_id = str(payload.get("wait_id") or "")
        if wait_id:
            wait_ids = capture.setdefault("wait_ids", [])
            if isinstance(wait_ids, list) and wait_id not in wait_ids:
                wait_ids.append(wait_id)

        remaining = _float_or_none(payload.get("remaining_seconds"))
        if remaining is None:
            deadline = _float_or_none(payload.get("deadline")) or _float_or_none(capture.get("deadline"))
            now = _float_or_none(payload.get("now"))
            remaining = max(0.0, deadline - now) if deadline is not None and now is not None else 0.0
        remaining = max(0.0, remaining)

        if bool(payload.get("done")):
            text = "LiteLLM rate limit wait complete; retrying now"
        else:
            text = _status_text(remaining)

        capture["last_tick"] = {
            "wait_id": wait_id,
            "remaining_seconds": remaining,
            "tick_index": _safe_int(payload.get("tick_index")),
            "done": bool(payload.get("done")),
        }

        return HarnessDecision(
            effects=[HarnessEffect("status", {"active": True, "text": text})],
            reason="LiteLLM/provider rate limit countdown status",
            payload={"plugin": PLUGIN_NAME, "source_event_id": source_event_id, "wait_id": wait_id},
        )


def parse_rate_limit_deadline(payload: dict[str, Any], *, max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS) -> DeadlineParse | None:
    """Return a runtime-clock deadline for a confident rate-limit payload.

    The parser is deliberately two-stage: first find a positive rate-limit
    signal, then parse retry/reset timing from headers, structured bodies, or
    conservative text patterns.
    """

    runtime_now = _runtime_now(payload)
    wall_time = _wall_time(payload)
    evidence = _collect_evidence(payload)
    signal = _rate_limit_signal(payload, evidence)
    if signal is None:
        return None

    for header_name, value in evidence.headers:
        normalized = _norm_key(header_name)
        if normalized in _RETRY_AFTER_KEYS:
            deadline = _parse_retry_after_value(value, runtime_now, wall_time, max_delay_seconds)
            if deadline is not None:
                return _result(deadline, runtime_now, f"header:{header_name}", signal)

    for header_name, value in evidence.headers:
        normalized = _norm_key(header_name)
        if normalized.startswith(_RESET_HEADER_PREFIXES):
            deadline = _parse_reset_value(value, runtime_now, wall_time, max_delay_seconds)
            if deadline is not None:
                return _result(deadline, runtime_now, f"header:{header_name}", signal)

    for key, value in evidence.body_fields:
        deadline = _parse_body_deadline_field(key, value, runtime_now, wall_time, max_delay_seconds)
        if deadline is not None:
            return _result(deadline, runtime_now, f"body:{key}", signal)

    for text in evidence.texts:
        deadline = _parse_text_deadline(text, runtime_now, wall_time, max_delay_seconds)
        if deadline is not None:
            return _result(deadline, runtime_now, "text", signal)

    return None


@dataclass
class _Evidence:
    statuses: list[int]
    class_names: list[str]
    headers: list[tuple[str, Any]]
    body_fields: list[tuple[str, Any]]
    texts: list[str]


def _collect_evidence(payload: dict[str, Any]) -> _Evidence:
    exception = payload.get("exception")
    statuses: list[int] = []
    class_names: list[str] = []
    headers: list[tuple[str, Any]] = []
    body_fields: list[tuple[str, Any]] = []
    texts: list[str] = []

    def add_text(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (bytes, bytearray)):
            try:
                value = value.decode("utf-8", "replace")
            except Exception:
                return
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                texts.append(stripped)

    for key in ("status_code", "status", "http_status", "http_status_code"):
        status = _safe_int(payload.get(key))
        if status is not None:
            statuses.append(status)

    exc_type = str(payload.get("exception_type") or "")
    if exc_type:
        class_names.append(exc_type)
    if exception is not None:
        class_names.append(f"{type(exception).__module__}.{type(exception).__qualname__}")
        for attr in ("status_code", "status", "http_status", "code"):
            status = _safe_int(getattr(exception, attr, None))
            if status is not None:
                statuses.append(status)
        response = getattr(exception, "response", None)
        if response is not None:
            status = _safe_int(getattr(response, "status_code", None) or getattr(response, "status", None))
            if status is not None:
                statuses.append(status)
            headers.extend(_headers_from(getattr(response, "headers", None)))
            add_text(getattr(response, "text", None))
            add_text(getattr(response, "content", None))
            json_func = getattr(response, "json", None)
            if callable(json_func):
                try:
                    _collect_body_fields(json_func(), body_fields, texts)
                except Exception:
                    pass
        for attr in ("headers", "response_headers"):
            headers.extend(_headers_from(getattr(exception, attr, None)))
        for attr in ("body", "json_body", "provider_payload", "payload"):
            _collect_body_fields(getattr(exception, attr, None), body_fields, texts)
        add_text(str(exception))

    for key in ("headers", "response_headers"):
        headers.extend(_headers_from(payload.get(key)))
    for key in ("message", "display_message", "detail", "traceback_text"):
        add_text(payload.get(key))
    for key in ("provider_payload", "body", "response", "error", "json"):
        _collect_body_fields(payload.get(key), body_fields, texts)

    # Dedupe while preserving order.
    seen_headers = set()
    unique_headers = []
    for name, value in headers:
        marker = (str(name).lower(), str(value))
        if marker not in seen_headers:
            seen_headers.add(marker)
            unique_headers.append((str(name), value))

    unique_texts = []
    seen_texts = set()
    for text in texts:
        if text not in seen_texts:
            seen_texts.add(text)
            unique_texts.append(text)

    return _Evidence(statuses=statuses, class_names=class_names, headers=unique_headers, body_fields=body_fields, texts=unique_texts)


def _rate_limit_signal(payload: dict[str, Any], evidence: _Evidence) -> str | None:
    if any(status == 429 for status in evidence.statuses):
        return "status:429"

    for name in evidence.class_names:
        compact = re.sub(r"[^a-z0-9]+", "", name.lower())
        if any(marker.replace("_", "").replace("-", "") in compact for marker in _CLASS_MARKERS):
            return "exception-type"

    if _has_reset_header(evidence):
        return "rate-limit-reset-header"

    text = "\n".join(evidence.texts).lower()
    has_rate_limit = any(marker in text for marker in _RATE_LIMIT_TEXT_MARKERS)
    phase = str(payload.get("phase") or "").lower()
    has_provider_context = any(marker in text for marker in _PROVIDER_TEXT_MARKERS) or "provider" in phase or "llm" in phase
    has_retry_after_header = _has_retry_after_header(evidence)

    if has_rate_limit and (has_provider_context or has_retry_after_header):
        return "message"

    # Retry-After is also valid for generic 503/service-maintenance responses.
    # Treat it as rate-limit evidence only when paired with some LiteLLM/provider
    # context and the event does not look like a generic server-side outage.
    if has_retry_after_header and has_provider_context and not _looks_like_generic_retry_after(evidence.statuses, text):
        return "retry-after-provider-context"

    return None


def _has_retry_after_header(evidence: _Evidence) -> bool:
    return any(_norm_key(name) in _RETRY_AFTER_KEYS for name, _value in evidence.headers)


def _has_reset_header(evidence: _Evidence) -> bool:
    return any(_norm_key(name).startswith(_RESET_HEADER_PREFIXES) for name, _value in evidence.headers)


def _looks_like_generic_retry_after(statuses: Iterable[int], text: str) -> bool:
    if any(status in {500, 502, 503, 504} for status in statuses):
        return True
    generic_markers = (
        "maintenance",
        "service unavailable",
        "temporarily unavailable",
        "gateway timeout",
        "bad gateway",
        "server error",
        "503",
    )
    return any(marker in text for marker in generic_markers)


def _headers_from(value: Any) -> list[tuple[str, Any]]:
    if value is None:
        return []
    try:
        items = value.items()  # Mapping and many HTTP header containers.
    except Exception:
        return []
    result = []
    try:
        for key, item in items:
            result.append((str(key), item))
    except Exception:
        return []
    return result


def _collect_body_fields(value: Any, body_fields: list[tuple[str, Any]], texts: list[str], *, depth: int = 0) -> None:
    if value is None or depth > 4:
        return
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:
            return
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return
        texts.append(stripped)
        if stripped[:1] in "[{":
            try:
                _collect_body_fields(json.loads(stripped), body_fields, texts, depth=depth + 1)
            except Exception:
                pass
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if _norm_key(key_text) in _BODY_DEADLINE_KEYS:
                body_fields.append((key_text, item))
            if key_text.lower() in {"message", "detail", "error", "type", "code"}:
                if isinstance(item, str):
                    texts.append(item)
            _collect_body_fields(item, body_fields, texts, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value[:20]:
            _collect_body_fields(item, body_fields, texts, depth=depth + 1)


def _parse_retry_after_value(value: Any, runtime_now: float, wall_time: float, max_delay: float) -> float | None:
    numeric = _float_or_none(value)
    if numeric is not None:
        return _deadline_from_relative(numeric, runtime_now, max_delay)
    text = str(value or "").strip()
    if not text:
        return None
    seconds = _parse_duration_seconds(text)
    if seconds is not None:
        return _deadline_from_relative(seconds, runtime_now, max_delay)
    return _deadline_from_date(text, runtime_now, wall_time, max_delay)


def _parse_reset_value(value: Any, runtime_now: float, wall_time: float, max_delay: float) -> float | None:
    numeric = _float_or_none(value)
    if numeric is not None:
        if numeric >= 1_000_000_000_000:  # epoch milliseconds
            return _deadline_from_wall_epoch(numeric / 1000.0, runtime_now, wall_time, max_delay)
        if numeric >= 1_000_000_000:  # epoch seconds
            return _deadline_from_wall_epoch(numeric, runtime_now, wall_time, max_delay)
        return _deadline_from_relative(numeric, runtime_now, max_delay)
    text = str(value or "").strip()
    if not text:
        return None
    seconds = _parse_duration_seconds(text)
    if seconds is not None:
        return _deadline_from_relative(seconds, runtime_now, max_delay)
    return _deadline_from_date(text, runtime_now, wall_time, max_delay)


def _parse_body_deadline_field(key: str, value: Any, runtime_now: float, wall_time: float, max_delay: float) -> float | None:
    normalized = _norm_key(key)
    if normalized in {"retry_after_ms", "retryafterms"}:
        ms = _float_or_none(value)
        if ms is not None:
            return _deadline_from_relative(ms / 1000.0, runtime_now, max_delay)
    if normalized in {"retry_after_seconds", "retryafterseconds"}:
        seconds = _float_or_none(value)
        if seconds is not None:
            return _deadline_from_relative(seconds, runtime_now, max_delay)
    if normalized in {"retry_after", "retryafter"}:
        return _parse_retry_after_value(value, runtime_now, wall_time, max_delay)
    return _parse_reset_value(value, runtime_now, wall_time, max_delay)


def _parse_text_deadline(text: str, runtime_now: float, wall_time: float, max_delay: float) -> float | None:
    for pattern in _TEXT_DURATION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        seconds = _parse_duration_seconds(match.group("value"))
        if seconds is not None:
            return _deadline_from_relative(seconds, runtime_now, max_delay)
    for pattern in _TEXT_DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        deadline = _deadline_from_date(match.group("value").strip(), runtime_now, wall_time, max_delay)
        if deadline is not None:
            return deadline
    return None


def _parse_duration_seconds(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    # Plain numeric duration strings are seconds.
    numeric = _float_or_none(text)
    if numeric is not None:
        return numeric

    total = 0.0
    matched = False
    for match in _DURATION_TOKEN_RE.finditer(text):
        matched = True
        number = float(match.group("num"))
        unit = match.group("unit").lower()
        if unit.startswith("ms") or unit.startswith("millisecond") or unit.startswith("msec"):
            total += number / 1000.0
        elif unit.startswith("s"):
            total += number
        elif unit.startswith("m"):
            total += number * 60.0
        elif unit.startswith("h"):
            total += number * 3600.0
    if matched:
        return total
    return None


def _deadline_from_date(value: str, runtime_now: float, wall_time: float, max_delay: float) -> float | None:
    wall_epoch = _parse_wall_epoch(value)
    if wall_epoch is None:
        return None
    return _deadline_from_wall_epoch(wall_epoch, runtime_now, wall_time, max_delay)


def _parse_wall_epoch(value: str) -> float | None:
    text = str(value or "").strip().strip('"\'')
    if not text:
        return None

    # Trim common trailing prose/punctuation without damaging RFC 7231 dates.
    text = re.sub(r"\s+(?:utc|gmt)?\s*(?:[).,]+)?$", lambda m: m.group(0).rstrip(".,)"), text, flags=re.IGNORECASE).strip()

    try:
        dt = email.utils.parsedate_to_datetime(text)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_dt.timezone.utc)
            return dt.timestamp()
    except Exception:
        pass

    iso_text = text
    if iso_text.endswith("Z"):
        iso_text = iso_text[:-1] + "+00:00"
    try:
        dt = _dt.datetime.fromisoformat(iso_text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _deadline_from_relative(seconds: float, runtime_now: float, max_delay: float) -> float | None:
    if not math.isfinite(seconds) or seconds < 0:
        return None
    if seconds > max_delay:
        return None
    return runtime_now + seconds


def _deadline_from_wall_epoch(wall_epoch: float, runtime_now: float, wall_time: float, max_delay: float) -> float | None:
    if not math.isfinite(wall_epoch):
        return None
    delay = wall_epoch - wall_time
    # Allow tiny negative skew; an already expired reset means immediate retry.
    if delay < -5.0:
        return None
    delay = max(0.0, delay)
    if delay > max_delay:
        return None
    return runtime_now + delay


def _result(deadline: float, runtime_now: float, source: str, signal: str) -> DeadlineParse | None:
    if not math.isfinite(deadline):
        return None
    return DeadlineParse(deadline=deadline, delay_seconds=max(0.0, deadline - runtime_now), source=source, signal=signal)


def _runtime_now(payload: dict[str, Any]) -> float:
    value = _float_or_none(payload.get("runtime_now"))
    if value is not None:
        return value
    return time.monotonic()


def _wall_time(payload: dict[str, Any]) -> float:
    value = _float_or_none(payload.get("wall_time"))
    if value is not None:
        return value
    return time.time()


def _status_text(remaining_seconds: float) -> str:
    seconds = max(0, int(math.ceil(remaining_seconds)))
    return f"LiteLLM rate limited; retrying in {seconds}s"


def _state_store(state) -> dict[str, Any]:
    current = getattr(state, STATE_ATTR, None)
    if not isinstance(current, dict):
        current = {}
        setattr(state, STATE_ATTR, current)
    current.setdefault("version", 1)
    current.setdefault("captures", {})
    return current


def _prune_captures(captures: dict[str, Any], max_captures: int) -> None:
    if len(captures) <= max_captures:
        return
    sortable = []
    for key, value in captures.items():
        if isinstance(value, dict):
            sortable.append((float(value.get("captured_runtime") or 0.0), key))
        else:
            sortable.append((0.0, key))
    for _created, key in sorted(sortable)[: max(0, len(captures) - max_captures)]:
        captures.pop(key, None)


def _plugin_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    value = config.get(PLUGIN_NAME)
    if isinstance(value, dict):
        return dict(value)
    # Also accept the package/module name for users who configure by repo name.
    value = config.get("litellm_rate_limit_retry")
    return dict(value) if isinstance(value, dict) else {}


def _norm_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _positive_float(value: Any, default: float) -> float:
    parsed = _float_or_none(value)
    if parsed is None or parsed <= 0:
        return float(default)
    return float(parsed)


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_int(value: Any) -> int | None:
    number = _float_or_none(value)
    if number is None:
        return None
    try:
        return int(number)
    except Exception:
        return None


__all__ = [
    "DEFAULT_MAX_DELAY_SECONDS",
    "LiteLLMRateLimitRetryHandler",
    "PLUGIN_NAME",
    "STATE_ATTR",
    "parse_rate_limit_deadline",
    "register_features",
]
