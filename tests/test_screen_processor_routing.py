"""actions/screen_processor.py's one-shot helpers (_vision_query / _text_query)
go through core/backend_router.py instead of a hardcoded Flash-Lite client:
VISION for images (Flash first, Flash-Lite fallback), CHAT for text (the
strongest configured text backend). study_mode's timed loop is the one
caller that opts back into Flash-Lite, via its own policy."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actions import screen_processor as sp  # noqa: E402
from core.backend_router import TaskKind  # noqa: E402


class VisionQueryRoutingTests(unittest.TestCase):

    def test_vision_query_routes_as_vision_with_image(self):
        seen = {}

        def fake_complete(kind, messages, tools=None, images=None, timeout=60, policy=None):
            seen.update(kind=kind, messages=messages, images=images, policy=policy)
            return {"content": "  notes  ", "backend": "gemini"}

        with patch("core.backend_router.complete", fake_complete), \
             patch.object(sp, "_routing_policy", return_value={"p": 1}):
            out = sp._vision_query(b"png", "image/png", "Extract notes")

        self.assertEqual(out, "notes")
        self.assertIs(seen["kind"], TaskKind.VISION)
        self.assertEqual(seen["images"], [(b"png", "image/png")])
        self.assertEqual(seen["messages"], [{"role": "user", "content": "Extract notes"}])
        self.assertEqual(seen["policy"], {"p": 1})

    def test_explicit_policy_wins_over_saved_routing(self):
        seen = {}

        def fake_complete(kind, messages, tools=None, images=None, timeout=60, policy=None):
            seen["policy"] = policy
            return {"content": "x"}

        lite = {TaskKind.VISION: ["gemini_lite", "gemini"]}
        with patch("core.backend_router.complete", fake_complete), \
             patch.object(sp, "_routing_policy", return_value={"saved": True}):
            sp._vision_query(b"png", "image/png", "p", policy=lite)
        self.assertEqual(seen["policy"], lite)

    def test_text_query_routes_as_chat_without_images(self):
        seen = {}

        def fake_complete(kind, messages, tools=None, images=None, timeout=60, policy=None):
            seen.update(kind=kind, images=images)
            return {"content": "[]"}

        with patch("core.backend_router.complete", fake_complete), \
             patch.object(sp, "_routing_policy", return_value=None):
            self.assertEqual(sp._text_query("make a quiz"), "[]")
        self.assertIs(seen["kind"], TaskKind.CHAT)
        self.assertIsNone(seen["images"])

    def test_router_failure_propagates(self):
        with patch("core.backend_router.complete", side_effect=RuntimeError("all failed")), \
             patch.object(sp, "_routing_policy", return_value=None):
            with self.assertRaises(RuntimeError):
                sp._text_query("x")


class StudyModeUsesLitePolicyTests(unittest.TestCase):

    def test_background_loop_policy_prefers_flash_lite(self):
        from actions import study_mode
        self.assertEqual(study_mode._BACKGROUND_VISION_POLICY,
                         {TaskKind.VISION: ["gemini_lite", "gemini"]})

    def test_quiz_generation_is_chat_kind(self):
        from actions import study_quiz
        seen = {}

        def fake_text_query(prompt, kind=None, policy=None):
            seen["kind"] = kind
            return '[{"question":"q","answer":"a","explanation":"e"}]'

        with patch.object(study_quiz, "_text_query", fake_text_query), \
             patch.object(study_quiz, "get_last_notes", return_value={"text": "notes", "topic": "t"}):
            out = study_quiz.study_quiz({"question_count": 1})
        self.assertIn("Q1: q", out)
        self.assertIs(seen["kind"], TaskKind.CHAT)


if __name__ == "__main__":
    unittest.main()
