from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from types import SimpleNamespace

import pytest

from agent_utils.core.events import InterruptRequested, RenderRequested, TimerFired
from agent_utils.core.io import CancelOp, StartTimer, SubmitOp
from agent_utils.core.session import Session
from agent_utils.core.state import State
from agent_zoo.events import (
    BeforeLLM,
    EmitNotice,
    LLMFailed,
    ProjectionRequested,
    RetryRequested,
    SetStatus,
)
from agent_zoo.pipelines.default import build_default_builder

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


def test_handler_llm_failed_stops_event_schedules_timer_and_stores_serializable_state():
    handler = LiteLLMRateLimitRetryHandler({"emit_one_shot_notice": True})
    state = State()
    emitted = []
    event = LLMFailed(
        error="LiteLLM proxy error: rate limit exceeded",
        attempt=1,
        provider_payload={"status_code": 429, "headers": {"Retry-After": "5"}},
        request_id="evt-1",
        ts=RUNTIME_NOW,
    )

    handler.handle(event, state, emitted.append)

    assert event.stopped is True
    effects = state.io.take_outbox()
    assert len(effects) == 1
    assert isinstance(effects[0], SubmitOp)
    assert isinstance(effects[0].request, StartTimer)
    assert effects[0].request.delay_s == pytest.approx(1.0)
    statuses = [ev for ev in emitted if isinstance(ev, SetStatus)]
    assert statuses and "retrying in 5s" in statuses[0].label
    notices = [ev for ev in emitted if isinstance(ev, EmitNotice)]
    assert notices and "retrying in 5s" in notices[0].text
    stored = getattr(state, STATE_ATTR)["captures"]["evt-1"]
    assert stored["deadline"] == pytest.approx(RUNTIME_NOW + 5)
    assert "exception" not in stored


def test_handler_ignores_unconfident_llm_failed():
    handler = LiteLLMRateLimitRetryHandler()
    state = State()
    emitted = []
    event = LLMFailed(error="network reset", attempt=1, request_id="evt-1", ts=RUNTIME_NOW)

    handler.handle(event, state, emitted.append)

    assert event.stopped is False
    assert emitted == []
    assert state.io.take_outbox() == []


def test_timer_tick_updates_status_and_triggers_retry_on_deadline():
    handler = LiteLLMRateLimitRetryHandler({"tick_seconds": 1.0})
    state = State()
    emitted = []
    error_event = LLMFailed(
        error="LiteLLM RateLimitError",
        attempt=1,
        provider_payload={"status_code": 429, "headers": {"Retry-After": "2.5"}},
        request_id="evt-1",
        ts=RUNTIME_NOW,
    )
    handler.handle(error_event, state, emitted.append)
    first_op = state.io.take_outbox()[0].op_id

    # Intermediate tick at +1.0s (1.5s remaining -> "retrying in 2s")
    emitted.clear()
    tick1 = TimerFired(op_id=first_op, ts=RUNTIME_NOW + 1.0)
    handler.handle(tick1, state, emitted.append)
    assert tick1.stopped is True
    statuses = [ev for ev in emitted if isinstance(ev, SetStatus)]
    assert statuses[0].label == "LiteLLM rate limited; retrying in 2s"
    assert not any(isinstance(ev, BeforeLLM) for ev in emitted)
    stored = getattr(state, STATE_ATTR)["captures"]["evt-1"]
    assert stored["last_tick"]["remaining_seconds"] == pytest.approx(1.5)
    assert stored["last_tick"]["done"] is False
    second_op = state.io.take_outbox()[0].op_id

    # Final tick at +2.5s (0s remaining -> emits RenderRequested + BeforeLLM)
    emitted.clear()
    tick2 = TimerFired(op_id=second_op, ts=RUNTIME_NOW + 2.5)
    handler.handle(tick2, state, emitted.append)
    assert tick2.stopped is True
    statuses = [ev for ev in emitted if isinstance(ev, SetStatus)]
    assert statuses[0].label == "LiteLLM rate limit wait complete; retrying now"
    assert any(isinstance(ev, RenderRequested) for ev in emitted)
    assert any(isinstance(ev, BeforeLLM) for ev in emitted)
    assert stored["last_tick"]["done"] is True
    assert state.io.take_outbox() == []


def test_timer_fired_for_unknown_op_is_ignored():
    handler = LiteLLMRateLimitRetryHandler()
    state = State()
    emitted = []
    tick = TimerFired(op_id="op_unknown", ts=RUNTIME_NOW + 1.0)

    handler.handle(tick, state, emitted.append)

    assert tick.stopped is False
    assert emitted == []


def test_interrupt_and_retry_cancel_active_wait():
    handler = LiteLLMRateLimitRetryHandler()
    state = State()
    event = LLMFailed(
        error="LiteLLM RateLimitError",
        attempt=1,
        provider_payload={"status_code": 429, "headers": {"Retry-After": "10"}},
        request_id="evt-1",
        ts=RUNTIME_NOW,
    )
    handler.handle(event, state, lambda _ev: None)
    op_id = state.io.take_outbox()[0].op_id

    interrupt = InterruptRequested(reason="user")
    handler.handle(interrupt, state, lambda _ev: None)
    assert interrupt.stopped is False
    effects = state.io.take_outbox()
    assert any(isinstance(eff, CancelOp) and eff.op_id == op_id for eff in effects)

    # Firing the cancelled op is a no-op
    emitted = []
    handler.handle(TimerFired(op_id=op_id, ts=RUNTIME_NOW + 10), state, emitted.append)
    assert emitted == []


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


def test_register_features_adds_named_feature_and_composes_with_default_builder():
    session = Session()
    builder = build_default_builder(
        session, config={}, include_plugins=False, include_llm_drivers=False
    )
    register_features(builder, session=session, config={})
    pipeline = builder.build()

    assert builder.has("litellm_rate_limit_retry_settings")
    assert builder.has("litellm_rate_limit_retry")
    names = [type(c).__name__ for c in pipeline.components]
    assert names.index("LiteLLMRateLimitRetryHandler") < names.index("DefaultRetryPolicy")
