"""Plugin Settings sections rendered by the HUD, and their test buttons. Mixed
into JarvisLive (main.py).
"""

import json
from memory.config_manager import get_plugin_config, save_plugin_config
from core import telemetry
from tool_connectors.registry import ToolRegistry
from live.constants import FOLLOW_UP_SECONDS
from live.prompting import _load_api_config


class SettingsPanelsMixin:
    """Plugin Settings sections rendered by the HUD, and their test buttons."""

    # ── Engine mode settings (Cloud vs. Local) ────────────────────────────────
    # Rendered by the existing, fully generic PluginSettingsOverlay (ui.py) —
    # it iterates whatever sections ui.get_plugin_settings() returns and knows
    # nothing about any specific plugin, so prepending a hand-built section
    # here needed no new Qt code. Persisted through the same
    # save_plugin_config("local_engine", …) path a real plugin's settings
    # would use.

    def _settings_schemas(self) -> list[dict]:
        return ([self._input_guard_settings_section(),
                 self._awareness_settings_section(), self._listening_settings_section(),
                 self._engine_settings_section(), self._claude_settings_section(),
                  self._fast_commands_section(), self._sentiment_settings_section(),
                  self._performance_settings_section(), self._routing_settings_section(),
                  self._mcp_settings_section(), self._hot_reload_settings_section()]
                + self._plugin_registry.settings_schemas())

    # ── Skill hot reload (core/skill_watcher.py) ───────────────────────────────
    # Same generic PluginSettingsOverlay rendering as every section above.
    # Like ENGINE mode, the toggle itself needs a restart to take effect (the
    # watcher thread is started once in __init__) — this section exists so
    # the setting is visible and persisted, not to apply it live.
    def _hot_reload_settings_section(self) -> dict:
        return {
            "plugin":    "hot_reload",
            "namespace": "hot_reload",
            "title":     "🔁 HOT RELOAD — reload plugins/actions on file save (restart to apply)",
            "fields": [
                {"key": "enabled", "type": "toggle",
                 "label": "Watch plugins/ and actions/ and reload changed files automatically",
                 "default": False},
            ],
            "values": get_plugin_config("hot_reload"),
            "action": {"label": "RELOAD ALL NOW", "run": self._reload_all_skills},
        }

    def _reload_all_skills(self, values: dict) -> tuple[bool, str]:
        """Manual one-shot reload — works whether or not the background
        watcher is running, since it drives the same registry.reload_all()
        the watcher itself calls per-file."""
        plugin_results = self._plugin_registry.reload_all()
        action_results = self._action_registry.reload_all()
        changed = [f"{name}: {msg}" for name, ok, msg in plugin_results + action_results if ok]
        failed  = [f"{name}: {msg}" for name, ok, msg in plugin_results + action_results if not ok
                   and "no TOOL dict" not in msg]
        if self._mode == "cloud" and changed:
            self.request_reconnect(keep_context=True, reason="skills reloaded")
        lines = [f"Reloaded {len(changed)} file(s)."]
        lines += changed[:10]
        if failed:
            lines.append(f"{len(failed)} failed:")
            lines += failed[:10]
        return (not failed, "\n".join(lines))

    # ── MCP connector (tool_connectors/connectors/mcp_connector.py) ───────────
    # Server list is edited as raw JSON in one text field rather than a
    # repeating add/remove row widget — same PluginSettingsOverlay rendering,
    # no new Qt code. CHECK SERVER HEALTH reuses the connector registry's own
    # health_report(), which already never raises per server (see
    # tool_connectors/mcp_connector_base.py's health_check()).
    def _mcp_settings_section(self) -> dict:
        return {
            "plugin":    "mcp_connector",
            "namespace": "mcp_connector",
            "title":     "🔌 MCP SERVERS — connect Model Context Protocol servers",
            "fields": [
                {"key": "mcp_servers_json", "type": "text", "label": 'Servers (JSON list — see tool_connectors/README.md)',
                 "default": "[]"},
            ],
            "values": {"mcp_servers_json": json.dumps(
                get_plugin_config("mcp_connector").get("mcp_servers", []))},
            "action": {"label": "CHECK SERVER HEALTH", "run": self._check_mcp_health},
        }

    def _check_mcp_health(self, values: dict) -> tuple[bool, str]:
        raw = values.get("mcp_servers_json", "[]")
        try:
            servers = json.loads(raw or "[]")
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        if not isinstance(servers, list):
            return False, "mcp_servers must be a JSON list of {name, transport, command|url} objects."

        # Persist under the key mcp_connector.py's config reader actually
        # expects (this section's own "mcp_servers_json" field is just this
        # UI's edit box), then re-discover so health reflects what was just typed.
        save_plugin_config("mcp_connector", {"mcp_servers": servers})
        registry = ToolRegistry(logger=lambda _msg: None).discover()
        if not registry.connectors():
            return (servers == [], "No MCP servers configured." if servers == [] else
                    "No servers registered — check names/transport in the JSON above.")

        report = registry.health_report()
        lines = [f"{'✓' if ok else '✗'} {name}" for name, ok in sorted(report.items())]
        return all(report.values()), "\n".join(lines)

    # ── Backend router (core/backend_router.py) ───────────────────────────────
    # One comma-separated text field per TaskKind rather than a new dropdown-
    # per-row widget — same PluginSettingsOverlay rendering as every section
    # above, no new Qt code. Read by core.backend_router.load_policy_from_config()
    # whenever a caller (e.g. actions/dev_agent.py, once migrated) asks for a
    # policy built from these saved values instead of DEFAULT_POLICY.
    def _routing_settings_section(self) -> dict:
        from core.backend_router import DEFAULT_POLICY, TaskKind, load_policy_from_config
        saved = get_plugin_config("routing")
        # What each task kind will ACTUALLY use right now — after the
        # local-model gate (a sub-7B Ollama model can't lead an order) and
        # minus anything unconfigured or in breaker cooldown. Without this,
        # the raw fields read "ollama, gemini, claude" while every call
        # quietly went to Gemini, and nobody could tell which one answered.
        try:
            note = ("Resolved right now (backends: gemini, gemini_lite, claude, ollama):\n"
                    + self._describe_routing(load_policy_from_config(saved)))
        except Exception as e:
            note = f"(could not resolve routing: {e})"
        return {
            "plugin":    "routing",
            "namespace": "routing",
            "title":     "🧭 ROUTING — backend order per task kind (comma-separated)",
            "note":      note,
            "fields": [
                {"key": kind.value, "type": "text", "label": kind.value.replace("_", " ").title(),
                 "default": ", ".join(DEFAULT_POLICY[kind])}
                for kind in TaskKind
            ],
            "values": saved,
            "action": {"label": "TEST ALL BACKENDS", "run": self._test_all_backends},
        }

    @staticmethod
    def _describe_routing(policy) -> str:
        from core.backend_router import describe_routing
        return describe_routing(policy)

    def _test_all_backends(self, values: dict) -> tuple[bool, str]:
        """Quick health probe for ollama/claude/gemini — independent of the
        saved routing order, since a backend's reachability doesn't depend on
        which task kinds are configured to use it — followed by the resolved
        order for the values currently typed in the fields (saved or not)."""
        import time as _time
        from core import llm_client
        from core.backend_router import load_policy_from_config
        from core.claude_bridge import get_claude_config, get_claude_settings

        results = []
        t0 = _time.monotonic()
        reachable = llm_client.ensure_ollama_running(timeout=3)
        results.append(f"ollama: {'reachable' if reachable else 'unreachable'} "
                        f"({(_time.monotonic()-t0)*1000:.0f}ms)")

        api_key, _, _ = get_claude_settings(get_claude_config())
        results.append(f"claude: {'configured' if api_key else 'no API key'}")

        gemini_key = _load_api_config().get("gemini_api_key", "")
        results.append(f"gemini: {'configured' if gemini_key else 'no API key'}")

        try:
            results.append("")
            results.append(self._describe_routing(load_policy_from_config(values or {})))
        except Exception as e:
            results.append(f"(could not resolve routing: {e})")

        return True, "\n".join(results)

    # ── Per-turn telemetry (core/telemetry.py) ────────────────────────────────
    # Same generic PluginSettingsOverlay rendering as every section above — the
    # "days" field is the only persisted value, and REFRESH STATS reuses the
    # existing TEST CONNECTION mechanism (an (ok, message) tuple rendered into
    # the section's status QLabel) purely as a read-only report surface, so no
    # new Qt widget was needed for this.
    def _performance_settings_section(self) -> dict:
        return {
            "plugin":    "performance",
            "namespace": "performance",
            "title":     "📊 PERFORMANCE — per-turn latency & token telemetry",
            "fields": [
                {"key": "days", "type": "text", "label": "Window (days)", "default": "7"},
            ],
            "values": get_plugin_config("performance"),
            "action": {"label": "REFRESH STATS", "run": self._refresh_performance_stats},
        }

    def _refresh_performance_stats(self, values: dict) -> tuple[bool, str]:
        try:
            days = max(1, int(float(values.get("days") or 7)))
        except (TypeError, ValueError):
            days = 7
        try:
            summary = telemetry.summary(days=days)
        except Exception as e:
            return False, f"Could not read telemetry: {e}"

        lines = [f"Last {days}d:"]
        if not summary["backends"]:
            lines.append("No turns recorded yet.")
        for name, s in sorted(summary["backends"].items()):
            p50 = f"{s['p50_time_to_first_audio_ms']:.0f}ms" if s["p50_time_to_first_audio_ms"] is not None else "n/a"
            p95 = f"{s['p95_time_to_first_audio_ms']:.0f}ms" if s["p95_time_to_first_audio_ms"] is not None else "n/a"
            lines.append(
                f"{name}: {s['turns']} turns · p50 {p50} · p95 {p95} · "
                f"tok in/out {s['tokens_in']}/{s['tokens_out']} · "
                f"fast-intent {s['fast_intent_hit_rate']*100:.0f}% · "
                f"interrupted {s['interrupted_rate']*100:.0f}%"
            )
        if summary["tools"]:
            top = sorted(summary["tools"].items(), key=lambda kv: -kv[1]["avg_ms"])[:5]
            lines.append("Slowest tools: " + ", ".join(f"{n} {v['avg_ms']:.0f}ms" for n, v in top))
        fs = summary.get("false_successes") or {}
        if fs.get("count"):
            worst = sorted(fs["by_tool"].items(), key=lambda kv: -kv[1])[:3]
            lines.append(f"False successes (failed tool, no admission): {fs['count']} — "
                         + ", ".join(f"{n}×{c}" for n, c in worst))
        else:
            lines.append("False successes: 0")
        return True, "\n".join(lines)

    # ── Claude collaboration mode (core/claude_bridge.py) ─────────────────────
    # Same generic PluginSettingsOverlay rendering as ENGINE/TONE ADAPTATION
    # above — no new Qt code. Unlike ENGINE, this toggle needs no restart: it's
    # read fresh via is_claude_engine_enabled() each time dev_agent/code_helper
    # pick a model, not cached at session start.
    def _claude_settings_section(self) -> dict:
        return {
            "plugin":    "claude_engine",
            "namespace": "claude_engine",
            "title":     "🤝 CLAUDE COLLAB MODE — Claude backs dev_agent/code_helper",
            "fields": [
                {"key": "enabled", "type": "toggle",
                 "label": "Use Claude instead of Gemini for dev_agent/code_helper text generation",
                 "default": False},
                {"key": "api_key", "type": "password", "label": "Anthropic API key",
                 "default": "", "placeholder": "sk-ant-..."},
                {"key": "model", "type": "text", "label": "Model",
                 "default": "claude-sonnet-5", "placeholder": "claude-sonnet-5"},
                {"key": "max_tokens", "type": "text", "label": "Max reply tokens",
                 "default": "1024"},
            ],
            "values": get_plugin_config("claude_engine"),
            "action": {"label": "TEST CONNECTION", "run": self._test_claude_engine},
        }

    def _test_claude_engine(self, values: dict) -> tuple[bool, str]:
        """Off-thread reachability probe for the settings panel's TEST button —
        checks the key the user just typed, before they even save, with the
        smallest possible real request (max_tokens=1) since Anthropic has no
        unauthenticated health endpoint to ping."""
        import requests
        from core.claude_bridge import ANTHROPIC_VERSION, API_URL

        api_key = str(values.get("api_key") or "").strip()
        model   = str(values.get("model") or "claude-sonnet-5").strip()
        if not api_key:
            return False, "No API key entered."
        try:
            resp = requests.post(
                API_URL,
                json={"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
                headers={
                    "x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return True, f"Reachable — '{model}' responded."
            detail = ""
            try:
                detail = resp.json().get("error", {}).get("message", "")
            except Exception:
                pass
            return False, f"HTTP {resp.status_code}: {detail}".strip()
        except Exception as e:
            return False, f"Request failed: {e}"

    def _sentiment_settings_section(self) -> dict:
        return {
            "plugin":    "sentiment_adapter",
            "namespace": "sentiment_adapter",
            "title":     "🎭 TONE ADAPTATION — adjust style to your mood",
            "fields": [
                {"key": "enabled", "type": "toggle",
                 "label": "Adjust tone/verbosity to detected mood (never changes facts or safety)",
                 "default": True},
                {"key": "persist_history", "type": "toggle",
                 "label": "Remember detected mood signals across sessions (off = this session only)",
                 "default": False},
            ],
            "values": get_plugin_config("sentiment_adapter"),
        }

    def _fast_commands_section(self) -> dict:
        return {
            "plugin":    "fast_commands",
            "namespace": "fast_commands",
            "title":     "⚡ FAST COMMANDS — skip the model for fixed actions",
            "fields": [
                {"key": "enabled", "type": "toggle",
                 "label": "Run typed device commands locally (instant)",
                 "default": True},
                {"key": "voice", "type": "toggle",
                 "label": "Also for spoken commands (volume, pause, open app…)",
                 "default": True},
            ],
            "values": get_plugin_config("fast_commands"),
        }

    # ── Input guard (core/input_guard.py) ─────────────────────────────────────

    def _input_guard_settings_section(self) -> dict:
        from core import input_guard
        return {
            "plugin":    "input_guard",
            "namespace": "input_guard",
            "title":     "⌨ INPUT GUARD — typing into other apps",
            "fields": [
                {"key": "confirm_actions", "type": "toggle",
                 "label": "Ask me to CONFIRM on screen first (off = JARVIS just does it): "
                          "sending messages, Enter in chats, typing into unlisted apps, "
                          "shutdown/restart/Wi-Fi",
                 "default": False},
                {"key": "enabled", "type": "toggle",
                 "label": "When asking is on: also for typing into apps not on the list",
                 "default": True},
                {"key": "confirm_messages", "type": "toggle",
                 "label": "When asking is on: also for every send_message",
                 "default": True},
                {"key": "allowed_apps", "type": "text",
                 "label": "Apps JARVIS may type into freely (comma-separated)",
                 "default": ", ".join(input_guard.DEFAULT_ALLOWED_APPS)},
                {"key": "messaging_apps", "type": "text",
                 "label": "Messaging apps where pressing Enter needs a yes",
                 "default": ", ".join(input_guard.DEFAULT_MESSAGING_APPS)},
            ],
            "values": get_plugin_config("input_guard"),
        }

    def _listening_settings_section(self) -> dict:
        return {
            "plugin":    "listening",
            "namespace": "listening",
            "title":     "🎙 LISTENING — when JARVIS pays attention",
            "fields": [
                {"key": "follow_up_seconds", "type": "text",
                 "label": "Seconds of quiet before it needs 'Hey Jarvis' again (wake-word mode)",
                 "default": str(int(FOLLOW_UP_SECONDS))},
            ],
            "values": get_plugin_config("listening"),
        }

    def _engine_settings_section(self) -> dict:
        return {
            "plugin":    "engine",
            "namespace": "local_engine",
            "title":     "🧠 ENGINE — Local Mode (offline STT/LLM/TTS)",
            "fields": [
                {"key": "enabled", "type": "toggle", "label": "Run fully local (restart required)",
                 "default": False},
                {"key": "auto_fallback", "type": "toggle",
                 "label": "Switch to Local Mode by itself when Gemini is down or out of quota",
                 "default": True},
                {"key": "llm_provider", "type": "choice", "label": "Backend",
                 "options": ["ollama", "openai"], "default": "ollama"},
                {"key": "llm_url", "type": "text", "label": "Server URL",
                 "default": "http://localhost:11434", "placeholder": "http://localhost:11434"},
                {"key": "llm_model", "type": "text", "label": "Model name",
                 "default": "llama3.2", "placeholder": "llama3.2"},
                {"key": "llm_temperature", "type": "text", "label": "Temperature (0.0-2.0)",
                 "default": "0.7"},
                {"key": "llm_max_tokens", "type": "text", "label": "Max reply tokens",
                 "default": "300"},
            ],
            "values": get_plugin_config("local_engine"),
            "action": {"label": "TEST CONNECTION", "run": self._test_local_engine},
        }

    def _test_local_engine(self, values: dict) -> tuple[bool, str]:
        """Off-thread reachability probe for the settings panel's TEST button —
        checks the values the user just typed, before they even save, so a
        wrong URL/port is caught here rather than by the local pipeline
        failing silently later."""
        import requests
        provider = str(values.get("llm_provider") or "ollama").strip().lower()
        default_url = "http://localhost:1234" if provider == "openai" else "http://localhost:11434"
        url = (str(values.get("llm_url") or "").strip() or default_url).rstrip("/")
        try:
            health = f"{url}/v1/models" if provider == "openai" else f"{url}/api/tags"
            resp = requests.get(health, timeout=5)
            if resp.status_code == 200:
                return True, f"Reachable at {url}"
            return False, f"Server at {url} returned HTTP {resp.status_code}"
        except Exception as e:
            return False, f"Unreachable at {url}: {e}"
