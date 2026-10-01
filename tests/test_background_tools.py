"""Slow tools run beside the conversation: JARVIS keeps hearing and
answering while a web search works, and the result arrives afterwards."""
import unittest

from google.genai import types

import main
from tests.live_harness import (
    FakeSession, Slow, heard, make_jarvis, pause, play, said, tick_sleep_watch,
    tool_call, turn_complete,
)

SEARCH = Slow(0.2, "Top result: the match starts at 7 pm.")
WHEN_IDLE = types.FunctionResponseScheduling.WHEN_IDLE
SILENT = types.FunctionResponseScheduling.SILENT


class BackgroundToolTest(unittest.TestCase):
    def test_search_does_not_block_the_next_turn(self):
        j = make_jarvis(tools={"web_search": SEARCH})
        seen_mid = {}

        def check_mid(j):
            seen_mid["responses"] = [fr.name for fr in j.session.tool_responses]
            seen_mid["running"] = j._tools_running
            seen_mid["logs"] = list(j.ui.logs)

        s = play(j, [
            heard("when does the match start"),
            tool_call("web_search", {"query": "match start time"}),
            said("Looking that up."), turn_complete(),
            heard("and turn the volume up"),
            tool_call("computer_settings", {"action": "volume_up"}, call_id="call-2"),
            said("Done."), turn_complete(),
            check_mid,
            pause(0.4),
        ])
        self.assertEqual(seen_mid["responses"], ["computer_settings"])
        self.assertEqual(seen_mid["running"], 1)
        self.assertIn("You: and turn the volume up", seen_mid["logs"])
        # …and then the search result arrives, timed for a gap.
        self.assertEqual([fr.name for fr in s.tool_responses], ["computer_settings", "web_search"])
        search = s.tool_responses[1]
        self.assertTrue(search.response["ok"])
        self.assertEqual(search.scheduling, WHEN_IDLE)
        self.assertEqual(j._tools_running, 0)
        self.assertTrue(any("web_search finished" in l for l in j.ui.logs))

    def test_quick_tools_still_answer_inline(self):
        j = make_jarvis()
        s = play(j, [tool_call("open_app", {"app_name": "chrome"})])
        self.assertEqual([fr.name for fr in s.tool_responses], ["open_app"])
        self.assertFalse(any("background" in l for l in j.ui.logs))

    def test_no_auto_sleep_while_it_runs(self):
        j = make_jarvis(tools={"web_search": Slow(0.3, "ok")})

        def sleep_check(j):
            j._last_activity -= main.WAKE_SLEEP_TIMEOUT + 1

        play(j, [tool_call("web_search", {"query": "x"}), sleep_check, pause(0.4)])
        # The tool finishing counts as activity, so the clock restarted.
        tick_sleep_watch(j)
        self.assertTrue(j._awake)

    def test_failure_is_reported_as_failed(self):
        j = make_jarvis(tools={"web_search": Slow(0.05, "Could not reach the search API")})
        s = play(j, [tool_call("web_search", {"query": "x"}), pause(0.2)])
        self.assertFalse(s.tool_responses[0].response["ok"])
        self.assertTrue(any("✗ web_search finished" in l for l in j.ui.logs))

    def test_result_after_bye_is_kept_quiet(self):
        j = make_jarvis(tools={"web_search": Slow(0.1, "Top result.")})
        s = play(j, [tool_call("web_search", {"query": "x"}),
                     lambda j: j.sleep(reason="bye jarvis"), pause(0.3)])
        self.assertEqual(s.tool_responses[0].scheduling, SILENT)

    def test_result_after_reconnect_goes_to_the_new_session(self):
        j = make_jarvis(tools={"web_search": Slow(0.1, "Top result.")})
        new = FakeSession(j, [])

        async def no_wait(timeout):
            return None

        j._wait_until_quiet = no_wait
        old = play(j, [tool_call("web_search", {"query": "x"}),
                       lambda j: setattr(j, "session", new), pause(0.3)])
        self.assertEqual(old.tool_responses, [])
        self.assertEqual(len(new.client_content), 1)
        self.assertIn("[BACKGROUND RESULT] web_search", new.client_content[0]["parts"][0]["text"])


class DeclarationTest(unittest.TestCase):
    def test_slow_tools_are_declared_non_blocking(self):
        j = make_jarvis()
        decls = {d["name"]: d for d in j._action_registry.get_tool_declarations()}
        for name in ("web_search", "flight_finder", "code_helper", "dev_agent",
                     "file_processor", "game_updater", "study_notes", "study_quiz"):
            self.assertEqual(decls[name].get("behavior"), "NON_BLOCKING", name)
        for name in ("open_app", "computer_control", "computer_settings", "send_message"):
            self.assertNotIn("behavior", decls[name], name)


if __name__ == "__main__":
    unittest.main()
