"""Spoken fast commands: a transcript that is exactly a device command runs
locally after a short pause, and the model's own call for it is not run a
second time."""
import unittest
from unittest.mock import patch

import live.constants
import main
from core import fast_intent
from tests.live_harness import (
    patch_everywhere,
    heard, make_jarvis, pause, play, said, tool_call, turn_complete,
)

SETTLE = 0.05


class FastVoiceTest(unittest.TestCase):
    def setUp(self):
        patches = [
            patch_everywhere("FAST_VOICE_SETTLE_SECONDS", SETTLE),
            patch_everywhere("get_plugin_config", return_value={}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_spoken_command_runs_before_the_model_answers(self):
        j = make_jarvis()
        play(j, [heard("volume up"), pause(SETTLE * 3)])
        self.assertEqual(j.dispatched, [("computer_settings", {"action": "volume_up"})])

    def test_model_call_for_the_same_thing_is_not_repeated(self):
        # "pause" is a toggle: running it twice would resume the video.
        j = make_jarvis()
        s = play(j, [heard("pause the video"), pause(SETTLE * 3),
                     tool_call("computer_settings", {"action": "pause_video"}),
                     said("Paused."), turn_complete()])
        self.assertEqual(len(j.dispatched), 1)
        self.assertTrue(s.tool_responses[0].response["ok"])
        self.assertIn("Already done", s.tool_responses[0].response["summary"])

    def test_more_speech_cancels_the_match(self):
        j = make_jarvis()
        play(j, [heard("open chrome"), heard("and go to gmail"), pause(SETTLE * 3)])
        self.assertEqual(j.dispatched, [])

    def test_model_tool_call_first_wins(self):
        j = make_jarvis()
        play(j, [heard("open chrome"),
                 tool_call("open_app", {"app_name": "Google Chrome"}),
                 pause(SETTLE * 3)])
        self.assertEqual(j.dispatched, [("open_app", {"app_name": "Google Chrome"})])

    def test_questions_and_unsafe_commands_go_to_the_model(self):
        for text in ("what's the volume", "close this window", "press enter"):
            j = make_jarvis()
            play(j, [heard(text), pause(SETTLE * 3)])
            self.assertEqual(j.dispatched, [], text)

    def test_once_per_turn(self):
        j = make_jarvis()
        # "louder" ... "please" still matches "louder" once filler is stripped.
        play(j, [heard("louder"), pause(SETTLE * 3), heard("please"), pause(SETTLE * 3)])
        self.assertEqual(len(j.dispatched), 1)

    def test_next_turn_can_run_again(self):
        j = make_jarvis()
        play(j, [heard("volume up"), pause(SETTLE * 3), turn_complete(),
                 heard("volume up"), pause(SETTLE * 3), turn_complete()])
        self.assertEqual(len(j.dispatched), 2)

    def test_switched_off(self):
        j = make_jarvis()
        with patch_everywhere("get_plugin_config", return_value={"voice": False}):
            play(j, [heard("volume up"), pause(SETTLE * 3)])
        self.assertEqual(j.dispatched, [])

    def test_stale_command_does_not_swallow_a_later_request(self):
        j = make_jarvis()
        play(j, [heard("volume up"), pause(SETTLE * 3), turn_complete()])
        j._fast_voice_done = (j._fast_voice_done[0], j._fast_voice_done[1] - live.constants.FAST_VOICE_DEDUPE_SECONDS - 1)
        play(j, [tool_call("computer_settings", {"action": "volume_up"})])
        self.assertEqual(len(j.dispatched), 2)


class SameCallTest(unittest.TestCase):
    def test_loose_app_names(self):
        i = fast_intent.detect("open chrome")
        self.assertTrue(fast_intent.same_call(i, "open_app", {"app_name": "Google Chrome"}))
        self.assertFalse(fast_intent.same_call(i, "open_app", {"app_name": "Safari"}))
        self.assertFalse(fast_intent.same_call(i, "youtube_video", {"app_name": "chrome"}))

    def test_settings_action_must_match(self):
        i = fast_intent.detect("volume up")
        self.assertFalse(fast_intent.same_call(i, "computer_settings", {"action": "volume_down"}))


if __name__ == "__main__":
    unittest.main()
