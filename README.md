# Agent Zoo LiteLLM Rate-Limit Retry Plugin

Userspace Agent Zoo plugin that observes generic harness events and automatically retries LiteLLM/OpenAI/Anthropic-style rate-limit failures at the provider-supplied retry/reset deadline.

It is a source-file plugin: the live file is `src/litellm_rate_limit_retry_plugin.py` and `pixi.toml` declares it under `[tool.agent-zoo.plugin]` for `common` scope.

## Install

From a git URL:

```bash
azo-plugin install https://github.com/Blair-Johnson/azo-plugin-rate-limit-retry.git
```

Then reload the Agent Zoo backend/TUI:

```text
/reload
```

Use `azo-plugin list` to confirm installation and `azo-plugin uninstall azo-plugin-rate-limit-retry` to remove it.

## Behavior

The plugin registers one `HarnessEventHandler` for:

- `runtime.error`
- `runtime.wait.tick`

On `runtime.error`, it conservatively detects rate-limit failures and parses a retry/reset deadline. If parsing is confident, it returns:

```python
HarnessDecision(
    control=HarnessControl.RETRY,
    wake=WakeSpec(at=<runtime_deadline>, tick_seconds=1.0),
)
```

The countdown UX uses `HarnessEffect("status", {"active": True, "text": "... retrying in Ns"})`, so it appears as active status/spinner text in the little TUI status line. It does not emit repeated `notice` effects for the countdown.

If no confident deadline is found, the handler returns `None` and Agent Zoo's default runtime error policy handles the error.

## Parser policy

Signals considered include:

- HTTP/status code `429`
- exception type names containing `RateLimit` or `TooManyRequests`
- rate-limit headers such as standard `RateLimit-Reset` / `ratelimit-reset` variants and provider `x-ratelimit-reset*` headers; `Retry-After` only when paired with other rate-limit/provider evidence
- LiteLLM/OpenAI/Anthropic/provider-ish rate-limit text markers

Deadline sources include:

- `Retry-After` numeric seconds
- `Retry-After` HTTP date
- standard `RateLimit-Reset` / `ratelimit-reset` and provider `x-ratelimit-reset*` relative seconds, epoch seconds/ms, ISO dates, HTTP dates, and duration strings such as `1m 30s`
- JSON/body fields: `retry_after`, `retry_after_seconds`, `retry_after_ms`, `reset_at`, `resetAt`, `reset_time`
- conservative text after a positive rate-limit signal, such as `retry after 42 seconds`, `try again in 1m 30s`, `reset in 12s`, and `reset at <date>`

Parsed absolute wall-clock dates are converted to harness runtime-clock deadlines using `runtime.error` payload fields `runtime_now` and `wall_time`.

## State and privacy

Plugin-owned state is stored under `litellm_rate_limit_retry_state`, keyed by `source_event_id`. It stores serializable metadata only: deadline, delay, source, signal, attempts, wait ids, and tick summaries. It does **not** persist raw exception objects or response headers.

## Configuration

Safe defaults are built into the plugin:

```json
{
  "litellm_rate_limit_retry": {
    "max_delay_seconds": 1200,
    "tick_seconds": 1.0,
    "max_captures": 64,
    "emit_one_shot_notice": false
  }
}
```

The same defaults are included in `config/litellm_rate_limit_retry.json` and declared in `pixi.toml` so `azo-plugin install` can copy it to the plugin config area. Current Agent Zoo userspace plugin loading passes the main harness config to `register_features`; if you merge the JSON under the same key into that config, the plugin will honor it. Missing config is safe.

`max_delay_seconds` is intentionally bounded (default 20 minutes). Longer parsed provider deadlines are ignored so default policy can take over instead of silently parking a session for an unexpectedly long time.

## Test and inspect

From this repo, run the test Pixi task. It sets the local development `PYTHONPATH` for the harness API checkout plus this repo's `src` directory:

```bash
pixi run -e test test
```

Static source-file plugin inspection from the Agent Zoo development checkout:

```bash
AGENT_ZOO_REPO=/path/to/agent-zoo
AGENT_UTILS_SRC=/path/to/agent-utils/src
PYTHONPATH="$AGENT_UTILS_SRC" \
python "$AGENT_ZOO_REPO/skills/agent-zoo-userspace-plugins/scripts/static_inspect_plugin.py" \
  src/litellm_rate_limit_retry_plugin.py
```

Dry-run install validation without touching your live Agent Zoo plugin state:

```bash
REPO=$(pwd)
AGENT_ZOO_REPO=/path/to/agent-zoo
AGENT_UTILS_SRC=/path/to/agent-utils/src
TMP_STATE=$(mktemp -d)
cd "$AGENT_ZOO_REPO"
AGENT_ZOO_HOME="$TMP_STATE" \
PYTHONPATH="$AGENT_UTILS_SRC:." \
pixi run python -m agent_zoo.plugin_install install "$REPO" --dry-run
rm -rf "$TMP_STATE"
```

Note: the current development installer prepares a managed source checkout before printing dry-run actions, so use an isolated `AGENT_ZOO_STATE_ROOT` for non-mutating validation.

## Limitations

- This plugin does not contact LiteLLM or providers; it only inspects runtime error payloads.
- It only overrides retry behavior when a rate-limit signal and a bounded retry/reset deadline are both found.
- It avoids durable notice spam; the countdown is status-line progress text.
- Header/body formats vary by provider and proxy version. Unknown formats intentionally fall back to Agent Zoo's default error policy.
