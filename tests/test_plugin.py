from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from types import SimpleNamespace

import pytest

from agent_utils.harness_events import HarnessControl, HarnessEvent

from litellm_rate_limit_retry_plugin import (
    LiteLLMRateLimitRetryHandler,
    STATE_ATTR,
    parse_rate_limit_deadline,
    register_features,
)


RUNTIME_NOW = 1000.0
WALL_TIME = 1_700_000_000.0


def payload(**overrides):
    base = {
        "phase": "llm.provider",
        "runtime_now": RUNTIME_NOW,
        "wall_time": WALL_TIME,
        "attempt": 1,
        "message": "LiteLLM proxy error: rate limit exceeded",
    }
    base.update(overrides)
    return base


def test_retry_after_numeric_seconds_header():
    parsed = parse_rate_limit_deadline(payload(status_code=429, headers={"Retry-After": "42"}))

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 42)
    assert parsed.source == "header:Retry-After"
    assert parsed.signal == "status:429"


def test_retry_after_http_date_converts_wall_clock_to_runtime_clock():
    wall_deadline = WALL_TIME + 90
    http_date = dt.datetime.fromtimestamp(wall_deadline, tz=dt.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    parsed = parse_rate_limit_deadline(payload(status_code=429, headers={"Retry-After": http_date}))

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 90)


@pytest.mark.parametrize(
    "value, expected_delay",
    [
        (str(int((WALL_TIME + 30) * 1000)), 30),
        (dt.datetime.fromtimestamp(WALL_TIME + 45, tz=dt.timezone.utc).isoformat(), 45),
        ("1m 30s", 90),
    ],
)
def test_x_ratelimit_reset_header_formats(value, expected_delay):
    parsed = parse_rate_limit_deadline(payload(status_code=429, headers={"x-ratelimit-reset-requests": value}))

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + expected_delay)
    assert parsed.source == "header:x-ratelimit-reset-requests"



def test_standard_ratelimit_reset_header_is_signal_and_deadline():
    parsed = parse_rate_limit_deadline(
        payload(message="LiteLLM provider response", headers={"RateLimit-Reset": "30"})
    )

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 30)
    assert parsed.source == "header:RateLimit-Reset"
    assert parsed.signal == "rate-limit-reset-header"


def test_standard_ratelimit_reset_lowercase_header_is_signal_and_deadline():
    parsed = parse_rate_limit_deadline(
        payload(message="LiteLLM provider response", headers={"ratelimit-reset": "45"})
    )

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 45)
    assert parsed.source == "header:ratelimit-reset"
    assert parsed.signal == "rate-limit-reset-header"


def test_retry_after_http_date_with_month_name_is_not_parsed_as_duration_prefix():
    http_date = "Fri, 01 May 2026 12:00:00 GMT"
    payload_wall_time = dt.datetime(2026, 5, 1, 11, 58, 30, tzinfo=dt.timezone.utc).timestamp()

    parsed = parse_rate_limit_deadline(
        payload(status_code=429, wall_time=payload_wall_time, headers={"Retry-After": http_date})
    )

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 90)
    assert parsed.source == "header:Retry-After"
def test_bare_retry_after_header_is_ignored_for_generic_503_maintenance():
    parsed = parse_rate_limit_deadline(
        payload(
            status_code=503,
            message="LiteLLM provider service unavailable for maintenance",
            headers={"Retry-After": "30"},
        )
    )

    assert parsed is None


def test_retry_after_header_with_provider_context_still_parses_when_not_generic_503():
    parsed = parse_rate_limit_deadline(
        payload(message="LiteLLM provider asked client to pause", headers={"Retry-After": "12"})
    )

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 12)
    assert parsed.signal == "retry-after-provider-context"


def test_body_retry_after_ms_field():
    parsed = parse_rate_limit_deadline(
        payload(
            status_code=429,
            provider_payload={"error": {"message": "too many requests", "retry_after_ms": 12500}},
        )
    )

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 12.5)
    assert parsed.source == "body:retry_after_ms"


def test_nested_error_code_429_and_limit_resets_at_utc_message():
    wall_deadline = WALL_TIME + 60
    reset_text = dt.datetime.fromtimestamp(wall_deadline, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    parsed = parse_rate_limit_deadline(
        payload(
            phase="runtime",
            message="",
            error={
                "code": "429",
                "message": (
                    "Rate limit exceeded for model_per_key: demo. Limit type: requests. "
                    f"Current limit: 4, Remaining: 0. Limit resets at: {reset_text}"
                ),
                "param": None,
                "type": "None",
            },
        )
    )

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 60)
    assert parsed.source == "text"
    assert parsed.signal == "status:429"


def test_top_level_payload_error_code_429_and_limit_resets_at_utc_message():
    wall_deadline = WALL_TIME + 75
    reset_text = dt.datetime.fromtimestamp(wall_deadline, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    parsed = parse_rate_limit_deadline(
        payload(
            phase="runtime",
            message="",
            payload={
                "error": {
                    "code": "429",
                    "message": (
                        "Rate limit exceeded for model_per_key: demo. Limit type: requests. "
                        f"Current limit: 4, Remaining: 0. Limit resets at: {reset_text}"
                    ),
                    "param": None,
                    "type": "None",
                }
            },
        )
    )

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 75)
    assert parsed.source == "text"
    assert parsed.signal == "status:429"

def test_recursively_finds_rate_limit_reset_in_unknown_envelope_shape():
    wall_deadline = WALL_TIME + 90
    reset_text = dt.datetime.fromtimestamp(wall_deadline, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    parsed = parse_rate_limit_deadline(
        payload(
            phase="runtime",
            message="",
            arbitrary_wrapper={
                "events": [
                    {
                        "metadata": {"code": "429"},
                        "diagnostic_blob": (
                            "provider failed: Limit resets at: "
                            f"{reset_text}', 'type': 'None', 'param': None"
                        ),
                    }
                ]
            },
        )
    )

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 90)
    assert parsed.source == "text"
    assert parsed.signal == "status:429"


def test_recursively_finds_camel_case_status_code_signal():
    reset_at = dt.datetime.fromtimestamp(WALL_TIME + 33, tz=dt.timezone.utc).isoformat()

    parsed = parse_rate_limit_deadline(
        payload(
            phase="runtime",
            message="",
            arbitrary_wrapper={"statusCode": 429, "details": {"resetAt": reset_at}},
        )
    )

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 33)
    assert parsed.source == "body:resetAt"
    assert parsed.signal == "status:429"


def test_body_reset_at_iso_field():
    reset_at = dt.datetime.fromtimestamp(WALL_TIME + 17, tz=dt.timezone.utc).isoformat()

    parsed = parse_rate_limit_deadline(payload(status_code=429, provider_payload={"resetAt": reset_at}))

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 17)
    assert parsed.source == "body:resetAt"


def test_text_try_again_in_duration_requires_rate_limit_signal():
    parsed = parse_rate_limit_deadline(payload(message="LiteLLM RateLimitError: try again in 1m 30s"))

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 90)


def test_text_retry_after_http_date_accepts_header_like_colon_separator():
    http_date = "Fri, 01 May 2026 12:00:00 GMT"
    payload_wall_time = dt.datetime(2026, 5, 1, 11, 58, 30, tzinfo=dt.timezone.utc).timestamp()

    parsed = parse_rate_limit_deadline(
        payload(
            wall_time=payload_wall_time,
            message=f"LiteLLM RateLimitError: Retry-After: {http_date}",
        )
    )

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 90)
    assert parsed.source == "text"


def test_text_retry_duration_keeps_decimal_seconds():
    parsed = parse_rate_limit_deadline(payload(message="LiteLLM RateLimitError: try again in 1.5s"))

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 1.5)

    parsed = parse_rate_limit_deadline(payload(message="LiteLLM RateLimitError: retry after: 0.25 seconds"))

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 0.25)


def test_text_duration_does_not_parse_unit_prefix_inside_word():
    parsed = parse_rate_limit_deadline(payload(message="LiteLLM RateLimitError: try again in 1 month"))

    assert parsed is None



def test_text_duration_does_not_parse_partial_number_before_punctuation():
    assert parse_rate_limit_deadline(payload(message="LiteLLM RateLimitError: try again in 1,000 seconds")) is None
    assert parse_rate_limit_deadline(payload(message="LiteLLM RateLimitError: retry after: 2,500 ms")) is None


def test_text_retry_deadline_ignored_without_positive_rate_limit_signal():
    parsed = parse_rate_limit_deadline(
        payload(phase="tool", message="temporary tool issue; try again in 1m 30s")
    )

    assert parsed is None


def test_delay_above_configured_max_is_not_captured():
    parsed = parse_rate_limit_deadline(payload(status_code=429, headers={"Retry-After": "3600"}), max_delay_seconds=1200)

    assert parsed is None


def test_exception_response_headers_are_parsed_without_persisting_exception():
    response = SimpleNamespace(status_code=429, headers={"Retry-After": "8"}, text="rate limit")

    class RateLimitError(Exception):
        pass

    exc = RateLimitError("LiteLLM RateLimitError")
    exc.response = response

    parsed = parse_rate_limit_deadline(payload(exception=exc, exception_type="litellm.RateLimitError"))

    assert parsed is not None
    assert parsed.deadline == pytest.approx(RUNTIME_NOW + 8)


def test_handler_runtime_error_returns_retry_decision_and_serializable_state():
    handler = LiteLLMRateLimitRetryHandler()
    state = SimpleNamespace()
    event = HarnessEvent("evt-1", "runtime.error", 1, payload(status_code=429, headers={"Retry-After": "5"}))

    decision = handler.handle_harness_event(event, state)

    assert decision is not None
    assert decision.control is HarnessControl.RETRY
    assert decision.wake.at == pytest.approx(RUNTIME_NOW + 5)
    assert decision.wake.tick_seconds == 1.0
    assert decision.effects[0].kind == "status"
    assert "retrying in 5s" in decision.effects[0].payload["text"]
    stored = getattr(state, STATE_ATTR)["captures"]["evt-1"]
    assert stored["deadline"] == pytest.approx(RUNTIME_NOW + 5)
    assert "exception" not in stored


def test_handler_returns_none_for_unconfident_runtime_error():
    handler = LiteLLMRateLimitRetryHandler()
    state = SimpleNamespace()
    event = HarnessEvent("evt-1", "runtime.error", 1, payload(message="network reset", headers={}))

    assert handler.handle_harness_event(event, state) is None


def test_wait_tick_updates_status_for_captured_source_event():
    handler = LiteLLMRateLimitRetryHandler()
    state = SimpleNamespace()
    error_event = HarnessEvent("evt-1", "runtime.error", 1, payload(status_code=429, headers={"Retry-After": "5"}))
    assert handler.handle_harness_event(error_event, state) is not None

    tick = HarnessEvent(
        "evt-2",
        "runtime.wait.tick",
        2,
        {
            "wait_id": "evt-1:wait",
            "source_event_id": "evt-1",
            "source_event_kind": "runtime.error",
            "source_control": "retry",
            "reason": "LiteLLM/provider rate limit retry deadline",
            "now": RUNTIME_NOW + 2,
            "deadline": RUNTIME_NOW + 5,
            "remaining_seconds": 3.1,
            "total_seconds": 5,
            "elapsed_seconds": 1.9,
            "tick_index": 2,
            "done": False,
            "end_reason": "",
        },
    )

    decision = handler.handle_harness_event(tick, state)

    assert decision is not None
    assert decision.control is HarnessControl.CONTINUE
    assert decision.effects == [decision.effects[0]]
    assert decision.effects[0].kind == "status"
    assert decision.effects[0].payload == {"active": True, "text": "LiteLLM rate limited; retrying in 4s"}
    stored = getattr(state, STATE_ATTR)["captures"]["evt-1"]
    assert stored["wait_ids"] == ["evt-1:wait"]
    assert stored["last_tick"]["remaining_seconds"] == pytest.approx(3.1)


def test_wait_tick_for_unknown_source_event_is_ignored():
    handler = LiteLLMRateLimitRetryHandler()
    state = SimpleNamespace(**{STATE_ATTR: {"version": 1, "captures": {}}})
    tick = HarnessEvent("evt-2", "runtime.wait.tick", 2, {"source_event_id": "missing", "remaining_seconds": 3})

    assert handler.handle_harness_event(tick, state) is None


def test_plugin_imports_with_agent_zoo_file_loader_style():
    module_name = "agent_zoo_plugin_loader_regression"
    source_path = sys.modules[parse_rate_limit_deadline.__module__].__file__
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    assert module.PLUGIN_NAME == "litellm_rate_limit_retry"


def test_register_features_adds_named_feature():
    class Builder:
        def __init__(self):
            self.features = []

        def add(self, feature):
            self.features.append(feature)

    builder = Builder()
    register_features(builder, session=SimpleNamespace(), config={})

    assert [feature.name for feature in builder.features] == ["litellm_rate_limit_retry"]
    assert isinstance(builder.features[0].components[0], LiteLLMRateLimitRetryHandler)
