"""Guard rails for core/prompt.txt — the persona and protocol every Live
session is built on.

The previous prompt grew to 12k characters of rules that contradicted each
other ("No fluff" next to "slightly witty"; "Never guess" next to "Assume and
proceed"; "speed is your number 1 priority" three times). A speech model with
25 tools and a shouting, self-contradicting rulebook picks the wrong tool
more often, not less. These tests keep the rewrite honest: short, one copy of
each rule, persona first, and the handful of protocol markers the code
depends on still present.
"""
from __future__ import annotations

import re
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROMPT = (ROOT / "core" / "prompt.txt").read_text(encoding="utf-8")

MAX_CHARS = 4_500
MAX_SHOUTS = 3                 # ALL-CAPS CRITICAL / ALWAYS / NEVER
# Markers main.py / tools emit that the prompt must still teach the model.
REQUIRED_MARKERS = [
    "[CONFIRMATION_PENDING]",
    "[ALSO REMEMBERED]",
    "[SYSTEM_ALERT]",
    "[STARTUP_BRIEFING]",
    "[PROACTIVE_CHECK]",
    "shutdown_jarvis",
    "screen_process",
    "recall_memory",
    "save_memory",
    "study_progress",
]
FILLER = ["Certainly", "Of course", "Great question", "I'd be happy to"]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n", text)
    return [p.strip().lower() for p in parts if len(p.strip()) > 20]


class PromptQualityTests(unittest.TestCase):

    def test_length(self):
        self.assertLessEqual(len(PROMPT), MAX_CHARS, f"prompt.txt is {len(PROMPT)} chars")

    def test_persona_comes_first(self):
        self.assertTrue(PROMPT.lstrip().startswith("PERSONA"), PROMPT[:40])

    def test_has_examples_section(self):
        self.assertIn("EXAMPLES", PROMPT)
        self.assertGreaterEqual(PROMPT.count("User:"), 5, "expected at least five example exchanges")

    def test_not_shouting(self):
        shouts = re.findall(r"\b(CRITICAL|ALWAYS|NEVER)\b", PROMPT)
        self.assertLessEqual(len(shouts), MAX_SHOUTS, shouts)

    def test_no_duplicate_sentences(self):
        dupes = [s for s, n in Counter(_sentences(PROMPT)).items() if n > 1]
        self.assertEqual(dupes, [], dupes)

    def test_required_markers_present(self):
        missing = [m for m in REQUIRED_MARKERS if m not in PROMPT]
        self.assertEqual(missing, [], missing)

    def test_filler_is_banned_not_used(self):
        # The filler phrases may appear only inside the rule that bans them.
        for phrase in FILLER:
            positions = [m.start() for m in re.finditer(re.escape(phrase), PROMPT)]
            self.assertLessEqual(len(positions), 1, f"{phrase!r} appears {len(positions)}x")

    def test_no_stale_contradictions(self):
        # The old prompt's mutually exclusive instructions must not creep back.
        for bad in ("Assume and proceed", "number 1 priority", "One-Call Policy"):
            self.assertNotIn(bad, PROMPT)


class IdentityBlockTests(unittest.TestCase):
    """main.py's _assemble_system_prompt() injects identity/address lines
    ahead of prompt.txt; they must read in the same register."""

    def test_identity_block(self):
        from unittest.mock import patch
        import main
        from tests.live_harness import patch_everywhere
        j = object.__new__(main.JarvisLive)
        cfg = {"assistant_name": "JARVIS", "user_name": ""}
        with patch_everywhere("_load_api_config", return_value=cfg), \
             patch_everywhere("load_memory", return_value={}), \
             patch_everywhere("format_memory_for_prompt", return_value=""):
            text = j._assemble_system_prompt()
        self.assertIn("You are JARVIS.", text)
        self.assertIn("at most once per reply", text)
        self.assertNotIn("Always refer to yourself", text)
        self.assertTrue(text.rstrip().endswith(PROMPT.rstrip()))

    def test_identity_block_with_user_name(self):
        from unittest.mock import patch
        import main
        from tests.live_harness import patch_everywhere
        j = object.__new__(main.JarvisLive)
        cfg = {"assistant_name": "FRIDAY", "user_name": "Raj"}
        with patch_everywhere("_load_api_config", return_value=cfg), \
             patch_everywhere("load_memory", return_value={}), \
             patch_everywhere("format_memory_for_prompt", return_value=""):
            text = j._assemble_system_prompt()
        self.assertIn("You are FRIDAY.", text)
        self.assertIn("Call the user 'Raj'", text)


if __name__ == "__main__":
    unittest.main()
