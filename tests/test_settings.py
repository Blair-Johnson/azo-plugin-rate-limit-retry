"""Read-only effective settings, through the actual pipeline and registry."""
from agent_utils import PipelineBuilder, Session
from agent_utils.builder import flatten
from agent_zoo.runtime.settings import Setting, SettingsContext, SettingsRegistry
from litellm_rate_limit_retry_plugin import register_features


def catalog(config):
    session = Session()
    builder = PipelineBuilder(session)
    register_features(builder, session=session, config=config)
    session.pipeline = builder.build()
    registry = SettingsRegistry([c for c in flatten(session.pipeline) if isinstance(c, Setting)])
    ctx = SettingsContext(session)
    result = registry.handle({"type": "plugin_settings", "action": "list", "request_id": "test"}, ctx)
    assert result["ok"]
    return registry, ctx, result["settings"]


def test_defaults_readonly_and_no_host_config_mutation():
    config = {"unrelated": {"prompt": "untouched"}}
    registry, ctx, rows = catalog(config)
    assert {r["key"]: r["value"] for r in rows} == {
        "rate_limit_retry.max_delay_ms": 1200000,
        "rate_limit_retry.tick_ms": 1000,
        "rate_limit_retry.max_captures": 64,
        "rate_limit_retry.emit_one_shot_notice": False,
    }
    assert all(not r["writable"] and r["apply"] == "reload" for r in rows)
    result = registry.handle({"type": "plugin_settings", "action": "set", "request_id": "write",
                              "binding": registry.binding, "key": "rate_limit_retry.tick_ms", "value": 1500}, ctx)
    assert not result["ok"]
    assert config == {"unrelated": {"prompt": "untouched"}}


def test_reload_fractional_seconds_exact_or_error():
    config = {"litellm_rate_limit_retry": {"tick_seconds": 0.125, "max_delay_seconds": 12.345,
                                           "max_captures": 7, "emit_one_shot_notice": True}}
    _, _, rows = catalog(config)
    assert {r["key"]: r["value"] for r in rows} == {
        "rate_limit_retry.max_delay_ms": 12345, "rate_limit_retry.tick_ms": 125,
        "rate_limit_retry.max_captures": 7, "rate_limit_retry.emit_one_shot_notice": True,
    }
    config["litellm_rate_limit_retry"]["tick_seconds"] = 0.0001
    _, _, rows = catalog(config)
    tick = next(r for r in rows if r["key"].endswith("tick_ms"))
    assert tick["value"] is None and tick["error"]
