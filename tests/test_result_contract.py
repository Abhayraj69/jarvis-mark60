"""core/result_contract.py — every tool result becomes {ok, summary, detail};
FalseSuccessTracker catches a spoken "done" after a failed tool; telemetry
records it. Plus main.py's thin wiring around both."""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import result_contract as rc  # noqa: E402
from core import telemetry  # noqa: E402


class ClassifyTests(unittest.TestCase):

    def test_plain_success(self):
        o = rc.classify("open_app", "Opened Spotify")
        self.assertTrue(o.ok)
        self.assertEqual(o.summary, "Opened Spotify")
        self.assertEqual(o.detail, "Opened Spotify")

    def test_failure_prefixes(self):
        for text in ("Tool 'file_controller' failed: boom", "Could not capture the screen",
                     "Couldn't find that file", "Unknown action. Use start | stop",
                     "Error: no such device", "Action 'x' is not available.",
                     "Unable to reach the server", "Invalid path"):
            self.assertFalse(rc.classify("file_controller", text).ok, text)

    def test_case_insensitive_and_whitespace(self):
        self.assertFalse(rc.classify("x", "   COULD NOT do it").ok)

    def test_status_reports_are_not_failures(self):
        for text in ("[CONFIRMATION_PENDING] press confirm on the HUD",
                     "No missed questions on record.",
                     "Not enough quiz history yet to identify weak topics.",
                     "No topics are being monitored."):
            self.assertTrue(rc.classify("study_progress", text).ok, text)

    def test_failure_word_mid_text_is_content(self):
        o = rc.classify("file_processor", "Summary: the author could not prove the lemma; error bars are wide.")
        self.assertTrue(o.ok)

    def test_empty_and_none(self):
        self.assertEqual(rc.classify("x", "").detail, "Done.")
        self.assertTrue(rc.classify("x", None).ok)

    def test_summary_is_first_line_and_capped(self):
        long = "A" * 400 + "\nsecond line"
        o = rc.classify("x", long)
        self.assertLessEqual(len(o.summary), rc.SUMMARY_MAX)
        self.assertTrue(o.summary.endswith("…"))
        o = rc.classify("x", "\n\nMoved 3 files\ndetails...")
        self.assertEqual(o.summary, "Moved 3 files")

    def test_dict_with_ok_passes_through(self):
        o = rc.classify("x", {"ok": False, "detail": "nope", "summary": "n"})
        self.assertFalse(o.ok); self.assertEqual(o.detail, "nope"); self.assertEqual(o.summary, "n")

    def test_as_response_shape(self):
        self.assertEqual(set(rc.classify("x", "hi").as_response()), {"ok", "summary", "detail"})


class AdmitsFailureTests(unittest.TestCase):

    def test_english(self):
        self.assertTrue(rc.admits_failure("I couldn't move it, sir."))
        self.assertTrue(rc.admits_failure("That didn't work — Documents is read-only."))
        self.assertFalse(rc.admits_failure("Done, sir. Anything else?"))

    def test_other_languages(self):
        self.assertTrue(rc.admits_failure("Maalesef dosyayı bulamadım."))
        self.assertTrue(rc.admits_failure("Lo siento, no pude abrirlo."))
        self.assertTrue(rc.admits_failure("Désolé, impossible de l'ouvrir."))

    def test_empty(self):
        self.assertFalse(rc.admits_failure(""))


class TrackerTests(unittest.TestCase):

    def test_nothing_pending(self):
        t = rc.FalseSuccessTracker()
        self.assertIsNone(t.conclude()); self.assertIsNone(t.poll())
        t.note_output("Done.")     # ignored when nothing is pending
        self.assertFalse(t.pending)

    def test_honest_reply(self):
        t = rc.FalseSuccessTracker()
        t.register_failure("file_controller", now=100.0)
        t.note_output("Moving it now.")            # the late-arriving acknowledgement
        t.note_output("It refused — read-only. Downloads instead?")
        v = t.conclude()
        self.assertFalse(v.false_success)
        self.assertEqual(v.tools, ["file_controller"])
        self.assertFalse(t.pending)

    def test_lie_is_caught(self):
        t = rc.FalseSuccessTracker()
        t.register_failure("file_controller", now=100.0)
        t.note_output("Moving it now.")
        t.note_output("Done, sir.")
        v = t.conclude()
        self.assertTrue(v.false_success)
        self.assertIn("Done, sir.", v.spoken)

    def test_multiple_failures_one_verdict(self):
        t = rc.FalseSuccessTracker()
        t.register_failure("a", now=1.0); t.register_failure("b", now=2.0)
        self.assertEqual(t.since, 1.0)
        v = t.conclude()
        self.assertEqual(v.tools, ["a", "b"])

    def test_poll_respects_deadline(self):
        t = rc.FalseSuccessTracker(deadline_s=10)
        t.register_failure("x", now=100.0)
        self.assertIsNone(t.poll(now=105.0))
        self.assertTrue(t.pending)
        v = t.poll(now=110.0)
        self.assertIsNotNone(v); self.assertTrue(v.false_success)
        self.assertFalse(t.pending)

    def test_speech_before_failure_is_not_counted(self):
        t = rc.FalseSuccessTracker()
        t.note_output("Earlier: it failed badly.")      # before any failure → dropped
        t.register_failure("x", now=1.0)
        t.note_output("Done.")
        self.assertTrue(t.conclude().false_success)


class TelemetryFalseSuccessTests(unittest.TestCase):

    def test_record_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            telemetry.record_false_success(["file_controller", "undo"], "Done, sir.", db_path=db)
            telemetry.record_false_success(["file_controller"], "All set.", db_path=db)
            for _ in range(50):                       # writes are off-thread
                s = telemetry.summary(days=1, db_path=db)
                if s["false_successes"]["count"] == 2:
                    break
                time.sleep(0.02)
            self.assertEqual(s["false_successes"]["count"], 2)
            self.assertEqual(s["false_successes"]["by_tool"], {"file_controller": 2, "undo": 1})

    def test_add_tool_span(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            t = telemetry.start_turn("gemini", db_path=db)
            t.add_tool_span("think:claude", 1234.5)
            t.finish()
            for _ in range(50):
                s = telemetry.summary(days=1, db_path=db)
                if "think:claude" in s["tools"]:
                    break
                time.sleep(0.02)
            self.assertAlmostEqual(s["tools"]["think:claude"]["avg_ms"], 1234.5)

    def test_add_tool_span_after_finish_attaches_to_latest_turn(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            t = telemetry.start_turn("gemini", db_path=db)
            t.finish()
            for _ in range(50):
                if telemetry.summary(days=1, db_path=db)["backends"]:
                    break
                time.sleep(0.02)
            self.assertTrue(t.finished)
            t.add_tool_span("think:gemini", 900.0)      # late: turn already written
            for _ in range(50):
                s = telemetry.summary(days=1, db_path=db)
                if "think:gemini" in s["tools"]:
                    break
                time.sleep(0.02)
            self.assertEqual(s["tools"]["think:gemini"]["count"], 1)


class MainWiringTests(unittest.TestCase):
    """main.py's _apply_result_contract / _conclude_false_success on a bare
    JarvisLive (no Qt, no audio)."""

    def _bare(self):
        import main
        j = object.__new__(main.JarvisLive)
        j.ui = MagicMock()
        j._false_success = rc.FalseSuccessTracker()
        return j

    def test_failure_registers_and_logs(self):
        j = self._bare()
        out = j._apply_result_contract("file_controller", "Could not move: read-only")
        self.assertFalse(out.ok)
        self.assertTrue(j._false_success.pending)
        j.ui.write_log.assert_called()
        self.assertIn("✗ file_controller", j.ui.write_log.call_args[0][0])

    def test_success_does_not_register(self):
        j = self._bare()
        self.assertTrue(j._apply_result_contract("open_app", "Opened Chrome").ok)
        self.assertFalse(j._false_success.pending)

    def test_conclude_records_false_success(self):
        import main
        j = self._bare()
        j._apply_result_contract("file_controller", "Could not move it")
        j._note_spoken_for_contract("Done, sir.")
        with patch.object(main.telemetry, "record_false_success") as rec:
            j._conclude_false_success(force=True)
        rec.assert_called_once()
        self.assertEqual(rec.call_args[0][0], ["file_controller"])
        self.assertFalse(j._false_success.pending)

    def test_conclude_honest_records_nothing(self):
        import main
        j = self._bare()
        j._apply_result_contract("file_controller", "Could not move it")
        j._note_spoken_for_contract("I couldn't move it — it's read-only.")
        with patch.object(main.telemetry, "record_false_success") as rec:
            j._conclude_false_success(force=True)
        rec.assert_not_called()

    def test_poll_without_force_waits_for_deadline(self):
        import main
        j = self._bare()
        j._apply_result_contract("x", "Error: nope")
        with patch.object(main.telemetry, "record_false_success") as rec:
            j._conclude_false_success()          # not forced, deadline not reached
        rec.assert_not_called()
        self.assertTrue(j._false_success.pending)


if __name__ == "__main__":
    unittest.main()
