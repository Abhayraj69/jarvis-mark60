"""Calendar and Mail (core/mac_apps.py, plugins/calendar_events.py,
plugins/email_inbox.py) and the morning briefing that uses them. osascript
is never run here — its output is faked."""
import asyncio
import json
import unittest
from datetime import datetime
from unittest.mock import patch

import main
from core import mac_apps
from tests.live_harness import make_jarvis


def ts(h, m=0):
    return datetime(2026, 10, 1, h, m).timestamp()


EVENTKIT_OUT = json.dumps({"events": [
    {"title": "Standup", "start": ts(9, 30), "end": ts(9, 45), "all_day": False,
     "location": "Zoom", "calendar": "Work"},
    {"title": "Holiday", "start": ts(0), "end": ts(23, 59), "all_day": True,
     "location": "", "calendar": "Home"},
    {"title": "Lunch with Sam", "start": ts(13), "end": ts(14), "all_day": False,
     "location": "", "calendar": "Home"},
]})


class CalendarTest(unittest.TestCase):
    def test_eventkit_events_sorted_all_day_first(self):
        with patch.object(mac_apps, "_osascript", return_value=EVENTKIT_OUT):
            evs = mac_apps.events()
        self.assertEqual([e.title for e in evs], ["Holiday", "Standup", "Lunch with Sam"])
        self.assertEqual(mac_apps.describe_events(evs),
                         "Today: all day: Holiday; 09:30 Standup (Zoom); 13:00 Lunch with Sam.")

    def test_falls_back_to_calendar_app(self):
        app_out = "Dentist\t2026-10-01T16:00:00\t2026-10-01T17:00:00\tfalse\tHome\n"
        with patch.object(mac_apps, "_osascript", return_value=json.dumps({"error": "status 2"})), \
             patch.object(mac_apps, "_run_as_with_args", return_value=app_out):
            evs = mac_apps.events()
        self.assertEqual([(e.title, e.start.hour) for e in evs], [("Dentist", 16)])

    def test_both_refused_says_where_to_allow_it(self):
        with patch.object(mac_apps, "_osascript", side_effect=RuntimeError("not allowed")), \
             patch.object(mac_apps, "_run_as_with_args", side_effect=RuntimeError("-1743")):
            with self.assertRaises(mac_apps.Unavailable) as ctx:
                mac_apps.events()
        self.assertIn("Privacy & Security", str(ctx.exception))

    def test_empty_day(self):
        self.assertEqual(mac_apps.describe_events([], "tomorrow"), "Nothing on your calendar tomorrow.")

    def test_eventkit_script_asks_the_right_range(self):
        js = mac_apps._wrap_js(1, 7)
        self.assertTrue(js.rstrip().endswith("fetchEvents(1, 7);"))
        self.assertNotIn("function run(", js, "osascript would call run() a second time")

    def test_plugin(self):
        from plugins import calendar_events
        with patch.object(mac_apps, "_osascript", return_value=EVENTKIT_OUT):
            out = calendar_events.run({"day": "week"})
        self.assertIn("Thursday:", out)
        with patch.object(mac_apps, "events", side_effect=mac_apps.Unavailable("allow it")):
            self.assertTrue(calendar_events.run({}).startswith("Could not read the calendar"))


class MailTest(unittest.TestCase):
    OUT = ("12\n"
           "Alice Smith <alice@x.com>\tInvoice due\t2026-10-01T08:00:00\n"
           "\"GitHub\" <noreply@github.com>\tCI failed on main\t2026-10-01T09:15:00\n")

    def test_parse_newest_first(self):
        total, mails = mac_apps.parse_mail(self.OUT)
        self.assertEqual(total, 12)
        self.assertEqual([m.subject for m in mails], ["CI failed on main", "Invoice due"])
        self.assertEqual(mac_apps.describe_mail(total, mails),
                         '12 unread emails. Latest — GitHub: "CI failed on main"; Alice Smith: "Invoice due".')

    def test_mail_not_open(self):
        with self.assertRaises(mac_apps.Unavailable):
            mac_apps.parse_mail("NOT_RUNNING")

    def test_nothing_unread(self):
        self.assertEqual(mac_apps.describe_mail(*mac_apps.parse_mail("0\n")), "No unread email.")

    def test_script_never_launches_mail(self):
        self.assertIn('if application "Mail" is not running then return "NOT_RUNNING"', mac_apps._MAIL_AS)
        for verb in ("delete", "send", "move", "set read status"):
            self.assertNotRegex(mac_apps._MAIL_AS, rf"\b{verb}\b")


class BriefingTest(unittest.TestCase):
    def test_agenda_added_to_brief(self):
        j = make_jarvis()
        j._proactive.get_morning_brief = lambda m, d: "News: rain later."
        replies = {"check_calendar": "Today: 09:30 Standup.", "check_email": "2 unread emails."}
        with patch.object(main, "get_plugin_config", return_value={}), \
             patch.object(j._plugin_registry, "has", return_value=True), \
             patch.object(j._plugin_registry, "run", side_effect=lambda t, a: replies[t]):
            brief = j._compose_brief({}, 0)
        self.assertEqual(brief, "News: rain later.\n\nToday's agenda: Today: 09:30 Standup. 2 unread emails.")

    def test_unreadable_agenda_is_left_out(self):
        j = make_jarvis()
        j._proactive.get_morning_brief = lambda m, d: "News."
        with patch.object(main, "get_plugin_config", return_value={}), \
             patch.object(j._plugin_registry, "has", return_value=True), \
             patch.object(j._plugin_registry, "run", return_value="Could not check email: Mail isn't open"):
            self.assertEqual(j._compose_brief({}, 0), "News.")

    def _wake_and_wait(self, j, then=None):
        async def run():
            j._loop = asyncio.get_running_loop()
            j.session = object()
            briefed = []

            async def fake_brief():
                briefed.append(True)
            j._send_startup_briefing = fake_brief
            with patch.object(main, "get_brief_enabled", return_value=True), \
                 patch.object(main, "BRIEF_AFTER_WAKE_SECONDS", 0.05):
                j.wake()
                if then:
                    then(j)
                await asyncio.sleep(0.2)
            return briefed
        return asyncio.run(run())

    def test_first_wake_of_the_day_gets_the_briefing(self):
        j = make_jarvis(awake=False)
        self.assertEqual(self._wake_and_wait(j), [True])
        j._awake = False
        self.assertEqual(self._wake_and_wait(j), [], "only once a day")

    def test_wake_for_a_command_postpones_it(self):
        j = make_jarvis(awake=False)

        def speaks(j):
            j._last_user_speech += 1.0
        self.assertEqual(self._wake_and_wait(j, then=speaks), [])
        self.assertEqual(j._briefing_day, "", "tries again on the next wake")


if __name__ == "__main__":
    unittest.main()
