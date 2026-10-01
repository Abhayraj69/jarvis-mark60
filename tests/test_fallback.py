"""Automatic Local Mode while Gemini is down (core/fallback.py and
JarvisLive._maybe_fall_back / _wait_for_cloud)."""
import asyncio
import unittest
from unittest.mock import patch

import main
from core import fallback, tts
from core.fallback import CloudHealth
from tests.live_harness import make_jarvis


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class CloudHealthTest(unittest.TestCase):
    def test_three_failures_in_two_minutes(self):
        c = Clock()
        h = CloudHealth(c)
        for _ in range(fallback.FAIL_COUNT - 1):
            h.failed("network")
        self.assertFalse(h.should_fall_back(models_resting=False))
        h.failed("other")
        self.assertTrue(h.should_fall_back(models_resting=False))

    def test_old_failures_expire(self):
        c = Clock()
        h = CloudHealth(c)
        for _ in range(fallback.FAIL_COUNT):
            h.failed("network")
            c.t += fallback.FAIL_WINDOW / 2
        self.assertFalse(h.should_fall_back(models_resting=False))

    def test_self_healing_errors_do_not_count(self):
        h = CloudHealth(Clock())
        for kind in ("idle_drop", "bad_handle", "drop_tuning", "drop_proactive", "bad_key") * 3:
            h.failed(kind)
        self.assertFalse(h.should_fall_back(models_resting=False))

    def test_all_models_resting_is_enough(self):
        self.assertTrue(CloudHealth(Clock()).should_fall_back(models_resting=True))

    def test_connecting_clears_it(self):
        h = CloudHealth(Clock())
        for _ in range(fallback.FAIL_COUNT):
            h.failed("network")
        h.connected()
        self.assertFalse(h.should_fall_back(models_resting=False))


class ReadinessTest(unittest.TestCase):
    def test_reports_what_is_missing(self):
        with patch.object(fallback, "_have", return_value=False), \
             patch.object(fallback, "mac_say_available", return_value=False):
            ok, missing = fallback.local_readiness({}, lambda: False)
        self.assertFalse(ok)
        self.assertEqual(len(missing), 3)

    def test_mac_say_counts_as_a_voice(self):
        with patch.object(fallback, "_have", side_effect=lambda m: m == "faster_whisper"), \
             patch.object(fallback, "mac_say_available", return_value=True):
            ok, missing = fallback.local_readiness({}, lambda: True)
        self.assertTrue(ok, missing)

    def test_tts_falls_back_to_say(self):
        with patch.object(tts, "_installed", return_value=False), \
             patch.object(tts.sys, "platform", "darwin"):
            player = tts.create_tts_player({"tts_engine": "edgetts"})
        self.assertIsInstance(player._engine, tts.MacSayEngine)


class SwitchTest(unittest.TestCase):
    """_maybe_fall_back with the local loop and the cloud probe stubbed."""

    def _jarvis(self, *, ready=True, cloud_back_after=0.05, local_exits=False):
        j = make_jarvis()
        j.calls = []

        async def local(fallback_mode=False):
            j.calls.append(("local", fallback_mode))
            if local_exits:
                return
            await asyncio.sleep(3600)

        async def probe():
            await asyncio.sleep(cloud_back_after)

        j._run_local_loop = local
        j._wait_for_cloud = probe
        self.patches = [
            patch.object(main, "get_plugin_config", return_value={}),
            patch.object(fallback, "local_readiness",
                         return_value=(ready, [] if ready else ["a local LLM"])),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        return j

    def test_switches_to_local_and_back(self):
        j = self._jarvis()
        j.session = object()
        back = asyncio.run(j._maybe_fall_back(models_resting=True))
        self.assertTrue(back)
        self.assertEqual(j.calls, [("local", True)])
        self.assertIsNone(j.session, "the dead session must not be used meanwhile")
        self.assertFalse(j._in_fallback)
        self.assertTrue(any("switching to Local Mode" in l for l in j.ui.logs))
        self.assertTrue(any("switching back to the cloud" in l for l in j.ui.logs))

    def test_not_set_up_keeps_retrying_cloud_and_says_why_once(self):
        j = self._jarvis(ready=False)
        self.assertFalse(asyncio.run(j._maybe_fall_back(models_resting=True)))
        self.assertFalse(asyncio.run(j._maybe_fall_back(models_resting=True)))
        warnings = [l for l in j.ui.logs if "can't take over" in l]
        self.assertEqual(len(warnings), 1)
        self.assertIn("a local LLM", warnings[0])
        self.assertEqual(j.calls, [])

    def test_switched_off(self):
        j = self._jarvis()
        with patch.object(main, "get_plugin_config", return_value={"auto_fallback": False}):
            self.assertFalse(asyncio.run(j._maybe_fall_back(models_resting=True)))
        self.assertEqual(j.calls, [])

    def test_local_failing_to_start_returns_to_cloud_retries(self):
        j = self._jarvis(local_exits=True, cloud_back_after=3600)
        self.assertFalse(asyncio.run(j._maybe_fall_back(models_resting=False)))
        self.assertFalse(j._in_fallback)

    def test_vision_is_refused_during_fallback(self):
        j = make_jarvis()
        j._in_fallback = True
        j._dispatch_tool = main.JarvisLive._dispatch_tool.__get__(j)
        result = asyncio.run(j._dispatch_tool("screen_process", {}))
        self.assertIn("not available in Local Mode", result)


class ProbeTest(unittest.TestCase):
    def test_waits_for_models_and_network(self):
        j = make_jarvis()
        j._fallback_streak = 1
        resting = iter([True, False, False])
        reachable = iter([False, True])
        with patch.object(fallback, "PROBE_SECONDS", 0.01), \
             patch.object(main._gemini, "all_live_models_resting", side_effect=lambda: next(resting)), \
             patch.object(fallback, "cloud_reachable", side_effect=lambda: next(reachable)):
            asyncio.run(asyncio.wait_for(j._wait_for_cloud(), 2))

    def test_repeat_fallbacks_wait_longer(self):
        j = make_jarvis()
        j._fallback_streak = 3
        waits = []

        async def fake_sleep(s):
            waits.append(s)
            raise asyncio.CancelledError

        with patch.object(main.asyncio, "sleep", fake_sleep):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(j._wait_for_cloud())
        self.assertEqual(waits[0], fallback.PROBE_SECONDS * 4)


if __name__ == "__main__":
    unittest.main()
