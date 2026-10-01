"""Fixes for what the first live run turned up (2026-10-01)."""
import asyncio
import time
import unittest
from datetime import datetime
from unittest.mock import patch

import main
from tests.live_harness import heard, make_jarvis, play, tool_call


class TimeTest(unittest.TestCase):
    def test_get_time_reads_the_clock(self):
        j = make_jarvis()
        j._dispatch_tool = main.JarvisLive._dispatch_tool.__get__(j)
        result = asyncio.run(j._dispatch_tool("get_time", {}))
        self.assertIn(datetime.now().strftime("%A"), result)
        self.assertIn(datetime.now().strftime("%I:%M %p")[:4], result)

    def test_declared(self):
        self.assertIn("get_time", {t["name"] for t in main.TOOL_DECLARATIONS})


class WakeNotByeTest(unittest.TestCase):
    def test_hey_jarvis_heard_as_bye_is_ignored(self):
        # 12:20:33 — "Hey Jarvis" said while awake, transcribed "Bye, Jarvis".
        j = make_jarvis()
        s = play(j, [lambda j: j._on_wake_detected(),
                     heard("Bye, Jarvis. Bye, Jarvis."), tool_call("shutdown_jarvis")])
        self.assertTrue(j._awake)
        self.assertIn("ignored", s.tool_responses[0].response["result"])
        self.assertTrue(any("not 'bye Jarvis'" in l for l in j.ui.logs))

    def test_real_goodbye_still_works(self):
        j = make_jarvis()
        j._wake_heard_at = time.monotonic() - main.WAKE_NOT_BYE_SECONDS - 1
        play(j, [heard("bye jarvis"), tool_call("shutdown_jarvis")])
        self.assertFalse(j._awake)

    def test_wake_word_while_awake_does_not_rewake_or_brief(self):
        j = make_jarvis()
        j._on_wake_detected()
        self.assertTrue(j._awake)
        self.assertNotEqual(j._wake_heard_at, -1e9)

    def test_end_session_is_a_goodbye(self):
        j = make_jarvis()
        play(j, [heard("End session"), tool_call("shutdown_jarvis")])
        self.assertFalse(j._awake)


class SearchFailureTest(unittest.TestCase):
    def test_both_backends_down_is_a_failure_not_no_results(self):
        from actions import web_search as ws

        class DownDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def text(self, *a, **k):
                raise TimeoutError("operation timed out")

        with patch.object(ws, "_gemini_search", side_effect=RuntimeError("every Gemini model failed")), \
             patch.object(ws, "_get_ddgs", return_value=DownDDGS):
            out = ws.web_search({"query": "song with the name nail"})
        self.assertTrue(out.startswith("Search failed"), out)
        from core import result_contract
        self.assertFalse(result_contract.classify("web_search", out).ok)

    def test_genuinely_nothing_found_is_still_no_results(self):
        from actions import web_search as ws

        class EmptyDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def text(self, *a, **k):
                return []

        with patch.object(ws, "_gemini_search", side_effect=RuntimeError("quota")), \
             patch.object(ws, "_get_ddgs", return_value=EmptyDDGS):
            self.assertTrue(ws.web_search({"query": "zzqx"}).startswith("No results found"))


class OpenAppTest(unittest.TestCase):
    def test_camera_means_photo_booth_on_a_mac(self):
        from actions import open_app as oa
        with patch.object(oa, "_SYSTEM", "Darwin"):
            self.assertEqual(oa._normalize("Camera"), "Photo Booth")

    def test_spotlight_guess_is_reported_as_a_guess(self):
        from actions import open_app as oa
        with patch.dict(oa._OS_LAUNCHERS, {oa._SYSTEM: lambda name, path="": "spotlight"}):
            out = oa.open_app({"app_name": "Frobnicator"})
        self.assertIn("No app called Frobnicator", out)
        self.assertNotIn("Opened Frobnicator", out)

    def test_normal_launch_unchanged(self):
        from actions import open_app as oa
        with patch.dict(oa._OS_LAUNCHERS, {oa._SYSTEM: lambda name, path="": True}):
            self.assertEqual(oa.open_app({"app_name": "Safari"}), "Opened Safari.")


if __name__ == "__main__":
    unittest.main()


class ReplyAudioHealthTest(unittest.TestCase):
    def _report(self, j):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            j._report_reply_audio()
        return buf.getvalue()

    def test_slow_arrival_is_reported(self):
        j = make_jarvis()
        speech_bytes = 3 * main.RECEIVE_SAMPLE_RATE * 2          # 3 s of speech
        j._reply_audio = [100.0, 127.0, speech_bytes]             # arrived over 27 s
        self.assertIn("slower than real time", self._report(j))
        self.assertIsNone(j._reply_audio)

    def test_healthy_reply_is_quiet(self):
        j = make_jarvis()
        j._reply_audio = [100.0, 101.0, 3 * main.RECEIVE_SAMPLE_RATE * 2]
        self.assertEqual(self._report(j), "")

    def test_underruns_are_reported_and_reset(self):
        j = make_jarvis()
        j._reply_audio = [100.0, 101.0, 48000]
        j._underruns = 4
        self.assertIn("ran out of audio 4 time(s)", self._report(j))
        self.assertEqual(j._underruns, 0)

    def test_receive_loop_tracks_audio(self):
        from tests.live_harness import audio, turn_complete
        j = make_jarvis()
        seen = {}
        play(j, [audio(48000), lambda j: seen.update(stats=list(j._reply_audio)), turn_complete()])
        self.assertEqual(seen["stats"][2], 48000)
        self.assertIsNone(j._reply_audio, "reset at turn_complete")


class ClockNoteTest(unittest.TestCase):
    class Sink:
        def __init__(self):
            self.sent = []

        async def send_client_content(self, turns=None, turn_complete=True):
            self.sent.append((turns["parts"][0]["text"], turn_complete))

    def _jarvis(self):
        j = make_jarvis()
        j.session = self.Sink()
        j._last_user_speech -= 60
        return j

    def test_sent_once_a_minute_without_asking_for_a_reply(self):
        j = self._jarvis()
        asyncio.run(j._maybe_send_clock())
        asyncio.run(j._maybe_send_clock())
        self.assertEqual(len(j.session.sent), 1, "same minute: once")
        text, turn_complete = j.session.sent[0]
        self.assertTrue(text.startswith("[CLOCK] "))
        self.assertIn(datetime.now().strftime("%A"), text)
        self.assertFalse(turn_complete, "must not make JARVIS talk")

    def test_never_while_anyone_is_talking(self):
        for setup in (lambda j: setattr(j, "_is_speaking", True),
                      lambda j: setattr(j, "_generating", True),
                      lambda j: setattr(j, "_last_user_speech", time.monotonic()),
                      lambda j: setattr(j, "_tools_running", 1)):
            j = self._jarvis()
            setup(j)
            asyncio.run(j._maybe_send_clock())
            self.assertEqual(j.session.sent, [])

    def test_new_minute_sends_again(self):
        j = self._jarvis()
        asyncio.run(j._maybe_send_clock())
        j._clock_sent = "00:00"
        asyncio.run(j._maybe_send_clock())
        self.assertEqual(len(j.session.sent), 2)
