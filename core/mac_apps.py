"""
Read-only access to the Mac's own Calendar and Mail.

No passwords, no OAuth app to register: whatever accounts are already added
to macOS (Google, iCloud, Exchange…) are read through the system. macOS asks
once for permission — Calendars for the events, Automation for Mail.

Calendar goes through EventKit (via JavaScript for Automation), which expands
recurring events properly — the standup that repeats every weekday is there
every weekday. If EventKit access is refused it falls back to scripting the
Calendar app, which only sees the first occurrence of a recurring event.

Mail is read only if the Mail app is already running; JARVIS never launches
it. Nothing here sends, moves, deletes or marks anything.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime

TIMEOUT = 90          # the first call can sit on the permission prompt


class Unavailable(Exception):
    """The app or permission is not there; the message says what to do."""


@dataclass
class Event:
    title: str
    start: datetime
    end: datetime
    all_day: bool = False
    location: str = ""
    calendar: str = ""


@dataclass
class Mail:
    sender: str
    subject: str
    received: datetime | None


def _osascript(args: list[str], script: str) -> str:
    if sys.platform != "darwin":
        raise Unavailable("Calendar and Mail access is only built for macOS so far.")
    r = subprocess.run(["osascript", *args, "-e", script], capture_output=True,
                       text=True, timeout=TIMEOUT)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-300:] or "osascript failed")
    return r.stdout.strip()


# ── Calendar ─────────────────────────────────────────────────────────────────

_EVENTKIT_JS = r"""
ObjC.import('EventKit');
ObjC.import('Foundation');
function fetchEvents(offset, days) {
  const store = $.EKEventStore.alloc.init;
  let status = $.EKEventStore.authorizationStatusForEntityType($.EKEntityTypeEvent);
  if (status === 0) {
    let done = false;
    const cb = (granted, err) => { done = true; };
    if (store.respondsToSelector('requestFullAccessToEventsWithCompletion:'))
      store.requestFullAccessToEventsWithCompletion(cb);
    else
      store.requestAccessToEntityTypeCompletion($.EKEntityTypeEvent, cb);
    const until = Date.now() + 80000;
    while (!done && Date.now() < until)
      $.NSRunLoop.currentRunLoop.runUntilDate($.NSDate.dateWithTimeIntervalSinceNow(0.25));
    status = $.EKEventStore.authorizationStatusForEntityType($.EKEntityTypeEvent);
  }
  if (status !== 3) return JSON.stringify({error: 'status ' + status});
  const start = $.NSCalendar.currentCalendar.startOfDayForDate($.NSDate.date)
                  .dateByAddingTimeInterval(offset * 86400);
  const end = start.dateByAddingTimeInterval(days * 86400);
  const evs = store.eventsMatchingPredicate(
      store.predicateForEventsWithStartDateEndDateCalendars(start, end, $()));
  const out = [];
  for (let i = 0; i < evs.count; i++) {
    const e = evs.objectAtIndex(i);
    out.push({title: ObjC.unwrap(e.title) || '', start: e.startDate.timeIntervalSince1970,
              end: e.endDate.timeIntervalSince1970, all_day: !!e.allDay,
              location: ObjC.unwrap(e.location) || '', calendar: ObjC.unwrap(e.calendar.title) || ''});
  }
  return JSON.stringify({events: out});
}
"""

_CALENDAR_APP_AS = r"""
on run argv
  set offsetDays to (item 1 of argv) as integer
  set numDays to (item 2 of argv) as integer
  set d0 to (current date) - (time of (current date)) + offsetDays * days
  set d1 to d0 + numDays * days
  set out to ""
  tell application "Calendar"
    repeat with c in calendars
      set evs to (every event of c whose start date ≥ d0 and start date < d1)
      repeat with e in evs
        set out to out & (summary of e) & tab & ((start date of e) as «class isot» as string) & tab & ¬
          ((end date of e) as «class isot» as string) & tab & (allday event of e) & tab & (name of c) & linefeed
      end repeat
    end repeat
  end tell
  return out
end run
"""


def parse_eventkit(out: str) -> list[Event]:
    data = json.loads(out or "{}")
    if "error" in data:
        raise PermissionError(data["error"])
    return [Event(e["title"], datetime.fromtimestamp(e["start"]), datetime.fromtimestamp(e["end"]),
                  bool(e.get("all_day")), e.get("location", ""), e.get("calendar", ""))
            for e in data.get("events", [])]


def parse_calendar_app(out: str) -> list[Event]:
    events = []
    for line in (out or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        title, start, end, all_day, cal = parts[:5]
        try:
            events.append(Event(title, datetime.fromisoformat(start), datetime.fromisoformat(end),
                                all_day.strip() == "true", "", cal))
        except ValueError:
            continue
    return events


def events(offset_days: int = 0, days: int = 1) -> list[Event]:
    """Events from the start of today+offset_days, for `days` days, sorted."""
    try:
        evs = parse_eventkit(_osascript(["-l", "JavaScript"], _wrap_js(offset_days, days)))
    except Exception as e:
        print(f"[Calendar] EventKit unavailable ({e}) — asking the Calendar app instead.")
        try:
            evs = parse_calendar_app(_run_as_with_args(_CALENDAR_APP_AS, offset_days, days))
        except Exception as e2:
            raise Unavailable(
                "I couldn't read your calendar. Allow access in System Settings → "
                f"Privacy & Security → Calendars (and Automation). ({e2})") from e2
    return sorted(evs, key=lambda e: (not e.all_day, e.start))


def _wrap_js(offset_days: int, days: int) -> str:
    # Not named run(): osascript would call that itself, with no arguments.
    return _EVENTKIT_JS + f"\nfetchEvents({int(offset_days)}, {int(days)});"


def _run_as_with_args(script: str, offset_days: int, days: int) -> str:
    if sys.platform != "darwin":
        raise Unavailable("Calendar access is only built for macOS so far.")
    r = subprocess.run(["osascript", "-", str(int(offset_days)), str(int(days))], input=script,
                       capture_output=True, text=True, timeout=TIMEOUT)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-300:] or "osascript failed")
    return r.stdout.strip()


def describe_events(evs: list[Event], label: str = "today") -> str:
    if not evs:
        return f"Nothing on your calendar {label}."
    parts = []
    for e in evs[:12]:
        if e.all_day:
            parts.append(f"all day: {e.title}")
        else:
            where = f" ({e.location})" if e.location else ""
            parts.append(f"{e.start.strftime('%H:%M')} {e.title}{where}")
    more = f" …and {len(evs) - 12} more" if len(evs) > 12 else ""
    return f"{label.capitalize()}: " + "; ".join(parts) + more + "."


# ── Mail ─────────────────────────────────────────────────────────────────────

_MAIL_AS = r"""
on run argv
  set lim to (item 1 of argv) as integer
  if application "Mail" is not running then return "NOT_RUNNING"
  tell application "Mail"
    set msgs to (messages of inbox whose read status is false)
    set n to count of msgs
    set out to (n as text) & linefeed
    set k to n
    if k > lim then set k to lim
    repeat with i from 1 to k
      set m to item i of msgs
      set out to out & (sender of m) & tab & (subject of m) & tab & ¬
        ((date received of m) as «class isot» as string) & linefeed
    end repeat
  end tell
  return out
end run
"""


def parse_mail(out: str) -> tuple[int, list[Mail]]:
    if out.strip() == "NOT_RUNNING":
        raise Unavailable("The Mail app isn't open, so I can't check your email. Open Mail and ask again.")
    lines = out.splitlines()
    try:
        total = int(lines[0].strip())
    except (IndexError, ValueError):
        return 0, []
    mails = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            when = datetime.fromisoformat(parts[2])
        except ValueError:
            when = None
        mails.append(Mail(parts[0], parts[1], when))
    mails.sort(key=lambda m: m.received or datetime.min, reverse=True)
    return total, mails


def unread_mail(limit: int = 5) -> tuple[int, list[Mail]]:
    if sys.platform != "darwin":
        raise Unavailable("Mail access is only built for macOS so far.")
    r = subprocess.run(["osascript", "-", str(int(limit))], input=_MAIL_AS,
                       capture_output=True, text=True, timeout=TIMEOUT)
    if r.returncode != 0:
        raise Unavailable("I couldn't read Mail. Allow JARVIS to control Mail in System Settings → "
                          f"Privacy & Security → Automation. ({r.stderr.strip()[-200:]})")
    return parse_mail(r.stdout)


def _short_sender(sender: str) -> str:
    name = sender.split("<")[0].strip().strip('"')
    return name or sender


def describe_mail(total: int, mails: list[Mail]) -> str:
    if total == 0:
        return "No unread email."
    head = f"{total} unread email{'s' if total != 1 else ''}."
    if not mails:
        return head
    items = [f"{_short_sender(m.sender)}: \"{m.subject[:80]}\"" for m in mails[:5]]
    return head + " Latest — " + "; ".join(items) + "."
