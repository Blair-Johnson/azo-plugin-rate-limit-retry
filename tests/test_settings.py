"""Saved defaults, through the actual pipeline, SettingsRegistry, and PluginStoreDriver."""
import json

from agent_utils.core.pipeline import PipelineBuilder
from agent_utils.core.session import Session
from agent_zoo.durability import PluginStoreDriver, StoreUpdate
from agent_zoo.features.settings import Setting, SettingsContext, SettingsRegistry
from agent_zoo.testing import assert_pipeline_sans_io
from litellm_rate_limit_retry_plugin import register_features, _defaults_path


def catalog(config):
    session = Session()
    builder = PipelineBuilder()
    register_features(builder, config=config)
    session.pipeline = builder.build(suppress_warnings=True)
    assert_pipeline_sans_io(session.pipeline)
    registry = SettingsRegistry([c for c in session.pipeline.components if isinstance(c, Setting)])
    ctx = SettingsContext(state=session.state, emit=lambda _ev: None)
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
    # Drain deferred StoreUpdate through PluginStoreDriver (0.4.0 Sans-I/O persistence).
    store_driver = PluginStoreDriver(state_root=tmp_path)
    store_reqs = [eff.request for eff in ctx.state.io.take_outbox() if isinstance(eff.request, StoreUpdate)]
    assert len(store_reqs) == 1
    store_driver.execute(store_reqs[0])
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
