"""Size guard for what the Live session sends on EVERY turn.

Every tool declaration (main.py's inline TOOL_DECLARATIONS, every actions/*.py
TOOL, every tool_connectors capability) plus core/prompt.txt is part of the
system instruction / tool config of each Gemini Live connect, and the model
re-reads all of it on each turn. Telemetry (memory/telemetry.db) showed a
median of ~27k input tokens per turn before the token diet; these limits stop
the budget creeping back up one "helpful" sentence at a time.

Behavioural rules ("say nothing after shutdown", "ask one question at a time")
belong in core/prompt.txt ONCE — a tool description is the contract for the
call, not a place for a second copy of the protocol.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from core.action_loader import discover_actions  # noqa: E402
from tool_connectors.registry import ToolRegistry  # noqa: E402

# Per-declaration limits.
MAX_TOOL_DESCRIPTION = 200
MAX_PARAM_DESCRIPTION = 80
# `action`-style enum parameters legitimately list every accepted value
# (computer_settings has 56) — they get a looser cap instead of an exemption.
MAX_ENUM_PARAM_DESCRIPTION = 700

# Whole-config limits (compact JSON, the closest proxy for what is serialized).
# 40 tools × ~150 parameters have a schema floor of roughly 18k characters
# ({"type":...,"description":...} per parameter); 24k leaves ~30% headroom for
# new tools while staying ~10k characters under the pre-diet 32k.
MAX_TOTAL_TOOL_JSON = 24_000
MAX_PROMPT_CHARS = 5_000


def _compact(obj) -> int:
    return len(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))


def _all_declarations() -> list[dict]:
    decls = list(main.TOOL_DECLARATIONS)
    reg = discover_actions(ROOT / "actions", logger=lambda _m: None)
    decls += reg.get_tool_declarations()
    conn = ToolRegistry(logger=lambda _m: None).discover()
    decls += conn.get_tool_declarations()
    return decls


def _is_enum_param(name: str, desc: str) -> bool:
    return name == "action" or desc.count("|") >= 3


class ToolBudgetTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.decls = _all_declarations()

    def test_something_was_discovered(self):
        names = {d["name"] for d in self.decls}
        # A discovery failure would make the size tests pass vacuously.
        self.assertGreaterEqual(len(names), 25, names)
        self.assertIn("computer_settings", names)
        self.assertIn("screen_process", names)

    def test_tool_descriptions_are_short(self):
        too_long = [
            (d["name"], len(d["description"]))
            for d in self.decls
            if len(d.get("description", "")) > MAX_TOOL_DESCRIPTION
        ]
        self.assertEqual(too_long, [], f"tool descriptions over {MAX_TOOL_DESCRIPTION} chars: {too_long}")

    def test_parameter_descriptions_are_short(self):
        too_long = []
        for d in self.decls:
            for pname, pspec in (d.get("parameters") or {}).get("properties", {}).items():
                desc = pspec.get("description", "") or ""
                cap = MAX_ENUM_PARAM_DESCRIPTION if _is_enum_param(pname, desc) else MAX_PARAM_DESCRIPTION
                if len(desc) > cap:
                    too_long.append((d["name"], pname, len(desc), cap))
        self.assertEqual(too_long, [], f"parameter descriptions over their cap: {too_long}")

    def test_total_tool_json_within_budget(self):
        total = _compact(self.decls)
        self.assertLessEqual(
            total, MAX_TOTAL_TOOL_JSON,
            f"tool declarations total {total} chars (> {MAX_TOTAL_TOOL_JSON}); "
            f"largest: {sorted(((_compact(d), d['name']) for d in self.decls), reverse=True)[:5]}",
        )

    def test_prompt_within_budget(self):
        text = (ROOT / "core" / "prompt.txt").read_text(encoding="utf-8")
        self.assertLessEqual(len(text), MAX_PROMPT_CHARS, f"core/prompt.txt is {len(text)} chars")

    def test_no_shouting_in_tool_descriptions(self):
        # "ALWAYS"/"NEVER"/"MUST" in caps is a tell that a behavioural rule has
        # leaked into a contract; those belong in prompt.txt.
        offenders = [
            d["name"] for d in self.decls
            if any(w in d.get("description", "") for w in ("ALWAYS", "NEVER", "MUST", "CRITICAL"))
        ]
        self.assertEqual(offenders, [], offenders)


class LiveContextLimitsTests(unittest.TestCase):
    """_live_context_limits() feeds ContextWindowCompressionConfig; a bad
    config value must clamp, never disable compression."""

    def _bare(self):
        return object.__new__(main.JarvisLive)

    def test_defaults(self):
        from unittest.mock import patch
        with patch.object(main, "_load_api_config", return_value={}):
            self.assertEqual(self._bare()._live_context_limits(), (12_000, 6_000))

    def test_override_and_clamp(self):
        from unittest.mock import patch
        cfg = {"live_context": {"trigger_tokens": 20_000, "target_tokens": 8_000}}
        with patch.object(main, "_load_api_config", return_value=cfg):
            self.assertEqual(self._bare()._live_context_limits(), (20_000, 8_000))
        cfg = {"live_context": {"trigger_tokens": "garbage", "target_tokens": 999_999}}
        with patch.object(main, "_load_api_config", return_value=cfg):
            trigger, target = self._bare()._live_context_limits()
            self.assertEqual(trigger, 12_000)
            self.assertEqual(target, 11_000)   # clamped to trigger - 1000


if __name__ == "__main__":
    unittest.main()
