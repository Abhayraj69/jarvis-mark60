"""Fixes for what the first live run turned up (2026-10-01)."""
import asyncio
import time
import unittest
from datetime import datetime
from unittest.mock import patch

import live.constants
import main
from tests.live_harness import heard, make_jarvis, play, tool_call, patch_everywhere


class TimeTest(unittest.TestCase):
    def test_get_time_reads_the_clock(self):
        j = make_jarvis()
        j._dispatch_tool = main.JarvisLive._dispatch_tool.__get__(j)
        result = asyncio.run(j._dispatch_tool("get_time", {}))
        self.assertIn(datetime.now().strftime("%A"), result)
        self.assertIn(datetime.now().strftime("%I:%M %p")[:4], result)

    def test_declared(self):
        self.assertIn("get_time", {t["name"] for t in main.TOOL_DECLARATIONS})


class SleepCheckTest(unittest.TestCase):
    """A sleep request is settled by a local transcript of the user's last
    seconds, because Gemini sometimes hears "Hey Jarvis" as "Bye, Jarvis"
    and the wake detector scores both the same."""

    def _jarvis(self, local):
        j = make_jarvis()
        j._local_transcript = lambda: local
        return j

    def test_hey_jarvis_heard_as_bye_is_ignored(self):
        j = self._jarvis("Hey Jarvis.")
        s = play(j, [heard("Bye, Jarvis."), tool_call("shutdown_jarvis")])
        self.assertTrue(j._awake)
        self.assertIn("ignored", s.tool_responses[0].response["result"])
        self.assertTrue(any('I heard "Hey Jarvis."' in l for l in j.ui.logs))

    def test_real_goodbye_sleeps(self):
        # 19:32:51 — "Bye, Jarvis." was refused by the old wake-word check.
        j = self._jarvis("Bye, Jarvis.")
        play(j, [heard("Bye, Jarvis."), tool_call("shutdown_jarvis")])
        self.assertFalse(j._awake)

    def test_sleep_jarvis_sleeps(self):
        j = self._jarvis("Sleep, Jarvis.")
        play(j, [heard("Sleep, Jarvis."), tool_call("shutdown_jarvis")])
        self.assertFalse(j._awake)

    def test_without_local_check_gemini_transcript_decides(self):
        j = self._jarvis(None)
        play(j, [heard("bye jarvis"), tool_call("shutdown_jarvis")])
        self.assertFalse(j._awake)
        j = self._jarvis(None)
        play(j, [heard("open youtube"), tool_call("shutdown_jarvis")])
        self.assertTrue(j._awake)

    def test_misspelled_sleep_jarvis(self):
        for text in ("Sleep Javas.", "Jarvas, go to sleep"):
            self.assertTrue(live.constants._FAREWELL_RE.search(text), text)
        for text in ("Hey Jarvis.", "I can't sleep", "sleep apnea just"):
            self.assertFalse(live.constants._FAREWELL_RE.search(text), text)

    def test_end_session_is_a_goodbye(self):
        j = self._jarvis("End session.")
        play(j, [heard("End session"), tool_call("shutdown_jarvis")])
        self.assertFalse(j._awake)

    def test_ring_buffer_keeps_only_the_last_seconds(self):
        import numpy as np
        j = make_jarvis()
        for _ in range(400):                      # ~25 s of 1024-sample blocks
            j._remember_mic(np.ones(1024, dtype=np.int16))
        self.assertLessEqual(j._recent_mic_len, live.constants.SLEEP_CHECK_SECONDS * live.constants.SEND_SAMPLE_RATE + 1024)


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


class FollowUpWindowTest(unittest.TestCase):
    def setUp(self):
        p = patch_everywhere("get_plugin_config", return_value={})
        p.start()
        self.addCleanup(p.stop)

    def test_sleeps_after_follow_up_window(self):
        from tests.live_harness import elapse, tick_sleep_watch
        j = make_jarvis()
        elapse(live.constants.FOLLOW_UP_SECONDS + 1)(j)
        tick_sleep_watch(j)
        self.assertFalse(j._awake)

    def test_still_awake_inside_the_window(self):
        from tests.live_harness import elapse, tick_sleep_watch
        j = make_jarvis()
        elapse(live.constants.FOLLOW_UP_SECONDS - 10)(j)
        tick_sleep_watch(j)
        self.assertTrue(j._awake)


class LocalGoodbyeTest(unittest.TestCase):
    """19:41 — the backup model heard "By javas" and never called
    shutdown_jarvis. Goodbyes are now handled locally."""

    def setUp(self):
        for p in (patch_everywhere("FAST_VOICE_SETTLE_SECONDS", 0.05),
                  patch_everywhere("get_plugin_config", return_value={})):
            p.start()
            self.addCleanup(p.stop)

    def _run(self, transcript, local, extra=()):
        from tests.live_harness import pause
        j = make_jarvis()
        j._local_transcript = lambda: local
        play(j, [heard(transcript), *extra, pause(0.2)])
        return j

    def test_garbled_goodbye_without_a_tool_call_sleeps(self):
        self.assertFalse(self._run("By javas", "Bye, Jarvis.")._awake)

    def test_hey_jarvis_misheard_as_bye_does_not(self):
        self.assertTrue(self._run("Bye, Jarvis.", "Hey Jarvis.")._awake)

    def test_no_local_check_trusts_the_transcript(self):
        self.assertFalse(self._run("Sleep, Jarvis.", None)._awake)

    def test_more_speech_cancels_it(self):
        self.assertTrue(self._run("bye", "bye, I'll call you later",
                                  extra=[heard("I'll call you later")])._awake)

    def test_ordinary_sentences_never_trigger(self):
        for text in ("by the way", "Jazz", "what are you doing"):
            self.assertTrue(self._run(text, "Bye Jarvis")._awake, text)


class QuietAfterGoodbyeTest(unittest.TestCase):
    def test_reply_after_goodbye_is_not_played(self):
        from tests.live_harness import audio, said, queued_audio_bytes, turn_complete
        j = make_jarvis()
        j._local_transcript = lambda: "Bye, Jarvis."
        play(j, [heard("Bye, Jarvis."), tool_call("shutdown_jarvis"),
                 said("Goodbye, sir. Call me when you need me."), audio(), audio(), turn_complete()])
        self.assertFalse(j._awake)
        self.assertEqual(queued_audio_bytes(j), 0)

    def test_awake_replies_still_play(self):
        from tests.live_harness import audio, said, queued_audio_bytes, turn_complete
        j = make_jarvis()
        play(j, [heard("hello"), said("Hello, sir."), audio(), turn_complete()])
        self.assertGreater(queued_audio_bytes(j), 0)


class SleepingDisplayTest(unittest.TestCase):
    def test_hud_keeps_showing_sleeping_after_goodbye(self):
        from tests.live_harness import audio, said, turn_complete
        j = make_jarvis()
        j._local_transcript = lambda: "Bye, Jarvis."
        play(j, [heard("Bye, Jarvis."), tool_call("shutdown_jarvis"),
                 said("Goodbye."), audio(), turn_complete()])
        j.set_speaking(True)
        j.set_speaking(False)           # what playback does when the reply ends
        self.assertFalse(j._awake)
        self.assertEqual(j.ui.state, "SLEEPING")

    def test_awake_still_shows_listening(self):
        j = make_jarvis()
        j.set_speaking(True)
        j.set_speaking(False)
        self.assertEqual(j.ui.state, "LISTENING")
