"""Saved defaults, through the actual pipeline and registry."""
import json

from agent_utils import PipelineBuilder, Session
from agent_utils.builder import flatten
from agent_zoo.runtime.settings import Setting, SettingsContext, SettingsRegistry
from litellm_rate_limit_retry_plugin import register_features, _defaults_path


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


def test_defaults_writable_and_no_host_config_mutation(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_HOME", str(tmp_path))
    config = {"unrelated": {"prompt": "untouched"}}
    registry, ctx, rows = catalog(config)
    assert {r["key"]: r["value"] for r in rows} == {
        "rate_limit_retry.max_delay_ms": 1200000,
        "rate_limit_retry.tick_ms": 1000,
        "rate_limit_retry.max_captures": 64,
        "rate_limit_retry.emit_one_shot_notice": False,
    }
    assert all(r["writable"] and r["apply"] == "reload" for r in rows)
    result = registry.handle({"type": "plugin_settings", "action": "set", "request_id": "write",
                              "binding": registry.binding, "key": "rate_limit_retry.tick_ms", "value": 1500}, ctx)
    assert result["ok"]
    assert json.loads(_defaults_path().read_text())["litellm_rate_limit_retry"]["tick_seconds"] == 1.5
    assert config == {"unrelated": {"prompt": "untouched"}}


def test_reload_fractional_seconds_exact_or_error(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_HOME", str(tmp_path))
    saved = {"litellm_rate_limit_retry": {"tick_seconds": 0.125, "max_delay_seconds": 12.345,
                                         "max_captures": 7, "emit_one_shot_notice": True}}
    path = _defaults_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(saved))
    override = {"litellm_rate_limit_retry": {"tick_seconds": 99}}
    _, _, rows = catalog(override)
    assert {r["key"]: r["value"] for r in rows} == {
        "rate_limit_retry.max_delay_ms": 12345, "rate_limit_retry.tick_ms": 125,
        "rate_limit_retry.max_captures": 7, "rate_limit_retry.emit_one_shot_notice": True,
    }
    saved["litellm_rate_limit_retry"]["tick_seconds"] = 0.0001
    path.write_text(json.dumps(saved))
    _, _, rows = catalog(override)
    tick = next(r for r in rows if r["key"].endswith("tick_ms"))
    assert tick["value"] is None and tick["error"]
