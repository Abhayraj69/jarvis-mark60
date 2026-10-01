"""Scripted conversations replayed through the real JarvisLive receive loop.

Each scenario is a bug that already shipped once: features interfering with
each other (sleep vs. reconnects, interrupts vs. playback, the model putting
itself to sleep). See tests/live_harness.py for how scripts work.
"""
import unittest

import main
from tests.live_harness import (
    audio, elapse, heard, make_jarvis, play, queued_audio_bytes, said,
    tick_sleep_watch, tool_call, turn_complete,
)


def _jarvis_lines(j):
    return [l for l in j.ui.logs if l.startswith(f"{j._asst_name}: ")]


class TurnTest(unittest.TestCase):
    def test_plain_turn_is_logged_and_played(self):
        j = make_jarvis()
        play(j, [heard("what time is it"), said("It's ten past nine, sir."),
                 audio(), turn_complete()])
        self.assertIn("You: what time is it", j.ui.logs)
        self.assertEqual(_jarvis_lines(j), [f"{j._asst_name}: It's ten past nine, sir."])
        self.assertEqual(queued_audio_bytes(j), 4800)

    def test_repeated_transcript_tail_is_logged_once(self):
        # The API re-sends the transcript tail across a tool call's
        # turn_completes; it used to be logged (and mouthed) twice.
        j = make_jarvis()
        play(j, [heard("open chrome"), said("Opening Chrome."), turn_complete(),
                 said("Opening Chrome."), turn_complete()])
        self.assertEqual(len(_jarvis_lines(j)), 1)


class InterruptTest(unittest.TestCase):
    def test_interrupt_mid_reply_drops_the_rest_of_it(self):
        j = make_jarvis()
        play(j, [heard("tell me a story"), said("Once upon a time"), audio(),
                 lambda j: j.interrupt(),
                 audio(), audio(), turn_complete()])
        self.assertEqual(queued_audio_bytes(j), 0)
        self.assertFalse(j._interrupted, "flag must clear on that reply's turn_complete")

    def test_next_reply_after_interrupt_is_heard(self):
        j = make_jarvis()
        play(j, [said("Once upon a time"), audio(),
                 lambda j: j.interrupt(), audio(), turn_complete(),
                 heard("what's the weather"), said("Sunny."), audio(), turn_complete()])
        self.assertEqual(queued_audio_bytes(j), 4800)
        self.assertIn(f"{j._asst_name}: Sunny.", j.ui.logs)

    def test_interrupt_when_reply_already_finished_does_not_eat_the_next(self):
        # Esc / stop pressed while only the buffered tail was playing: no
        # turn_complete is coming, so the flag must not be left set.
        j = make_jarvis()
        play(j, [said("Done."), audio(), turn_complete(),
                 lambda j: j.interrupt(),
                 heard("thanks"), said("Anytime."), audio(), turn_complete()])
        self.assertIn(f"{j._asst_name}: Anytime.", j.ui.logs)
        self.assertGreater(queued_audio_bytes(j), 0)


class ToolTest(unittest.TestCase):
    def test_tool_call_runs_and_answers(self):
        j = make_jarvis(tools={"open_app": "Opened Chrome."})
        s = play(j, [heard("open chrome"), tool_call("open_app", {"app_name": "chrome"}),
                     said("Chrome is open."), turn_complete()])
        self.assertEqual(j.dispatched, [("open_app", {"app_name": "chrome"})])
        self.assertEqual(len(s.tool_responses), 1)
        self.assertTrue(s.tool_responses[0].response["ok"])
        self.assertEqual(j._tools_running, 0)

    def test_failed_tool_is_reported_as_failed(self):
        j = make_jarvis(tools={"open_app": "Tool 'open_app' failed: not installed"})
        s = play(j, [heard("open photoshop"), tool_call("open_app", {"app_name": "photoshop"})])
        self.assertFalse(s.tool_responses[0].response["ok"])
        self.assertTrue(any(l.startswith("SYS: ✗ open_app") for l in j.ui.logs))


class SelfSleepTest(unittest.TestCase):
    def test_model_cannot_sleep_without_a_goodbye(self):
        j = make_jarvis()
        s = play(j, [heard("open youtube"), tool_call("shutdown_jarvis")])
        self.assertTrue(j._awake)
        self.assertIn("ignored", s.tool_responses[0].response["result"])

    def test_model_cannot_sleep_on_stale_speech(self):
        j = make_jarvis()
        play(j, [heard("bye jarvis"), turn_complete(), elapse(120),
                 tool_call("shutdown_jarvis")])
        self.assertTrue(j._awake)

    def test_goodbye_puts_it_to_sleep(self):
        j = make_jarvis()
        play(j, [heard("bye jarvis"), tool_call("shutdown_jarvis")])
        self.assertFalse(j._awake)
        self.assertEqual(j.ui.state, "SLEEPING")


class AutoSleepTest(unittest.TestCase):
    def test_sleeps_after_two_quiet_minutes(self):
        j = make_jarvis()
        elapse(main.WAKE_SLEEP_TIMEOUT + 1)(j)
        tick_sleep_watch(j)
        self.assertFalse(j._awake)

    def test_does_not_sleep_while_a_tool_is_running(self):
        j = make_jarvis()
        elapse(main.WAKE_SLEEP_TIMEOUT + 1)(j)
        j._tools_running = 1
        tick_sleep_watch(j)
        self.assertTrue(j._awake)

    def test_finishing_a_reply_restarts_the_clock(self):
        j = make_jarvis()
        elapse(main.WAKE_SLEEP_TIMEOUT + 1)(j)
        j.set_speaking(True)
        j.set_speaking(False)
        j._tail_until = 0.0
        tick_sleep_watch(j)
        self.assertTrue(j._awake)

    def test_never_sleeps_with_wake_word_off(self):
        j = make_jarvis(wake_word=False)
        elapse(main.WAKE_SLEEP_TIMEOUT + 1)(j)
        tick_sleep_watch(j)
        self.assertTrue(j._awake)


class ReconnectTest(unittest.TestCase):
    def test_first_connect_with_wake_word_comes_up_asleep(self):
        j = make_jarvis(awake=True)
        j._has_connected = False
        j._on_session_connected()
        self.assertFalse(j._awake)
        self.assertEqual(j.ui.state, "SLEEPING")

    def test_reconnect_while_awake_stays_awake(self):
        j = make_jarvis(awake=True)
        j._on_session_connected()
        self.assertTrue(j._awake)
        self.assertEqual(j.ui.state, "LISTENING")

    def test_reconnect_while_asleep_stays_asleep(self):
        j = make_jarvis(awake=False)
        j._on_session_connected()
        self.assertFalse(j._awake)

    def test_wake_then_drop_then_reconnect_keeps_listening(self):
        # The run.log case: "Hey Jarvis", then a 1008 drop, then the user
        # speaks — they must not have to say the wake word again.
        j = make_jarvis(awake=False)
        j.wake()
        j._on_session_connected()
        play(j, [heard("what's on my screen"), said("Your editor."), turn_complete()])
        self.assertTrue(j._awake)
        self.assertIn("You: what's on my screen", j.ui.logs)


class LiveErrorTest(unittest.TestCase):
    def _kind(self, err, top=None, *, resumed=False, uptime=120.0, tuned=False, enhanced=False):
        return main._classify_live_error(err, top if top is not None else err,
                                         resumed_with=resumed, uptime=uptime,
                                         tuned=tuned, enhanced=enhanced)

    def test_idle_drop_on_working_session(self):
        self.assertEqual(self._kind("x | 1008 None. The operation was aborted."), "idle_drop")

    def test_1008_at_connect_backs_off(self):
        self.assertEqual(self._kind("1008 None. The operation was aborted.", uptime=1.0), "other")

    def test_rejected_resume_handle(self):
        self.assertEqual(self._kind("INVALID_ARGUMENT: bad handle", resumed=True), "bad_handle")

    def test_tuning_dropped_before_proactive(self):
        self.assertEqual(self._kind("INVALID_ARGUMENT", tuned=True, enhanced=True), "drop_tuning")
        self.assertEqual(self._kind("INVALID_ARGUMENT", tuned=False, enhanced=True), "drop_proactive")

    def test_bad_key_and_network(self):
        self.assertEqual(self._kind("API key not valid"), "bad_key")
        self.assertEqual(self._kind("getaddrinfo failed"), "network")

    def test_internal_error_is_left_to_the_model_ladder(self):
        self.assertEqual(self._kind("1011 Internal error"), "other")


if __name__ == "__main__":
    unittest.main()
