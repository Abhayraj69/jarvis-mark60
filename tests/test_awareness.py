"""Awareness (core/awareness.py): activity summary, break nudges, the
stuck-on-an-error offer, and how main.py delivers them."""
import asyncio
import unittest
from unittest.mock import patch

import main
from core import awareness
from core.awareness import Awareness
from tests.live_harness import make_jarvis, patch_everywhere

MIN = 60.0


class Clock:
    def __init__(self):
        self.t = 10_000.0

    def __call__(self):
        return self.t


def run_for(a: Awareness, clock: Clock, minutes: float, app: str, title: str = "", idle: float = 0.0):
    steps = int(minutes * MIN / awareness.SAMPLE_SECONDS)
    for _ in range(max(steps, 1)):
        a.sample(app, title, idle)
        clock.t += awareness.SAMPLE_SECONDS


class ActivityTest(unittest.TestCase):
    def test_summary_lists_recent_windows(self):
        c = Clock()
        a = Awareness(c)
        run_for(a, c, 5, "Google Chrome", "Docs")
        run_for(a, c, 25, "Code", "main.py — Mark-LV")
        s = a.summary()
        self.assertTrue(s.startswith("Code — main.py — Mark-LV (25 min)"), s)
        self.assertIn("before that Google Chrome — Docs", s)

    def test_same_app_minutes_spans_its_windows(self):
        c = Clock()
        a = Awareness(c)
        run_for(a, c, 4, "Code", "a.py")
        run_for(a, c, 4, "Code", "b.py")
        self.assertAlmostEqual(a.same_app_minutes(), 8, delta=0.5)


class BreakTest(unittest.TestCase):
    def test_nudge_after_long_stretch_once(self):
        c = Clock()
        a = Awareness(c)
        run_for(a, c, awareness.BREAK_AFTER_MINUTES - 5, "Code")
        self.assertIsNone(a.break_nudge())
        run_for(a, c, 10, "Code")
        n = a.break_nudge()
        self.assertEqual(n.kind, "break")
        a.mark_nudged(n)
        run_for(a, c, 30, "Code")
        self.assertIsNone(a.break_nudge(), "once per stretch")

    def test_stepping_away_resets_the_stretch(self):
        c = Clock()
        a = Awareness(c)
        run_for(a, c, awareness.BREAK_AFTER_MINUTES - 10, "Code")
        a.sample("Code", "", idle=awareness.IDLE_RESET_MINUTES * MIN + 1)
        run_for(a, c, 20, "Code")
        self.assertIsNone(a.break_nudge())


class StuckTest(unittest.TestCase):
    ERR = "ModuleNotFoundError: No module named 'sherpa_onnx'"

    def _in_terminal(self, minutes=awareness.ERROR_CHECK_AFTER_MINUTES + 1):
        c = Clock()
        a = Awareness(c)
        run_for(a, c, minutes, "Terminal", "zsh")
        return a, c

    def test_only_checks_dev_apps_after_a_while(self):
        c = Clock()
        a = Awareness(c)
        run_for(a, c, 30, "Safari")
        self.assertFalse(a.wants_error_check())
        a, c = self._in_terminal(minutes=2)
        self.assertFalse(a.wants_error_check())
        a, c = self._in_terminal()
        self.assertTrue(a.wants_error_check())

    def test_same_error_twice_means_stuck(self):
        a, c = self._in_terminal()
        self.assertIsNone(a.error_seen(self.ERR))
        c.t += awareness.ERROR_CHECK_EVERY_MINUTES * MIN
        n = a.error_seen(self.ERR + " (line 12)")
        self.assertEqual(n.kind, "stuck")
        self.assertIn("ModuleNotFoundError", n.text)

    def test_a_new_error_is_not_stuck(self):
        a, c = self._in_terminal()
        a.error_seen(self.ERR)
        self.assertIsNone(a.error_seen("TypeError: unsupported operand type(s) for +"))

    def test_offered_once_per_error(self):
        a, c = self._in_terminal()
        a.error_seen(self.ERR)
        a.mark_nudged(a.error_seen(self.ERR))
        c.t += awareness.NUDGE_COOLDOWN_MINUTES * MIN + 1
        self.assertIsNone(a.error_seen(self.ERR))

    def test_parse_error_reply(self):
        self.assertEqual(awareness.parse_error('```json\n{"error": "KeyError: \'x\'"}\n```'), "KeyError: 'x'")
        self.assertEqual(awareness.parse_error('{"error": ""}'), "")
        self.assertEqual(awareness.parse_error("no json here"), "")


class DeliveryTest(unittest.TestCase):
    def _tick(self, j, cfg, app=("Code", "main.py")):
        calls = {"n": 0}

        async def one_pass(_):
            calls["n"] += 1
            if calls["n"] > 1:
                raise asyncio.CancelledError

        async def run():
            j.session = FakeSink()
            with patch.object(main.asyncio, "sleep", one_pass), \
                 patch_everywhere("get_plugin_config", return_value=cfg), \
                 patch("core.input_guard.frontmost", return_value=app), \
                 patch.object(awareness, "idle_seconds", return_value=0.0):
                try:
                    await j._run_awareness()
                except asyncio.CancelledError:
                    pass
            return j.session.sent

        return asyncio.run(run())

    def _due_for_break(self):
        j = make_jarvis()
        j._last_user_speech -= 60
        j._awareness._active_since = j._awareness._clock() - (awareness.BREAK_AFTER_MINUTES + 1) * MIN
        return j

    def test_break_nudge_is_sent(self):
        sent = self._tick(self._due_for_break(), {})
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0].startswith("[AWARENESS]"))

    def test_not_while_the_user_is_talking(self):
        j = self._due_for_break()
        j._last_user_speech = main.time.monotonic()
        self.assertEqual(self._tick(j, {}), [])

    def test_not_while_asleep(self):
        j = self._due_for_break()
        j._awake = False
        self.assertEqual(self._tick(j, {}), [])

    def test_switched_off(self):
        self.assertEqual(self._tick(self._due_for_break(), {"enabled": False}), [])
        self.assertEqual(self._tick(self._due_for_break(), {"break_reminders": False}), [])

    def test_screen_check_is_opt_in(self):
        j = make_jarvis()
        j._last_user_speech -= 60
        for _ in range(int((awareness.ERROR_CHECK_AFTER_MINUTES + 1) * MIN / awareness.SAMPLE_SECONDS)):
            j._awareness.sample("Terminal", "zsh", 0)
            j._awareness._clock = (lambda t=j._awareness._clock() + awareness.SAMPLE_SECONDS: t)
        with patch.object(main.JarvisLive, "_screen_error", side_effect=AssertionError("captured!")):
            self._tick(j, {}, app=("Terminal", "zsh"))   # default: screen_errors off


class FakeSink:
    def __init__(self):
        self.sent = []

    async def send_client_content(self, turns=None, turn_complete=True):
        self.sent.append(turns["parts"][0]["text"])


if __name__ == "__main__":
    unittest.main()
