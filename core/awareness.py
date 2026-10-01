"""
Awareness: JARVIS notices what you are doing.

The proactive check-in already existed, but it only knew the clock and the
memory file — so it could say "good afternoon" and nothing about the build
error you have been staring at for ten minutes. This module is the missing
input. Three things, all decided locally:

  1. Activity. Every SAMPLE_SECONDS the focused app and window title are read
     (core/input_guard.frontmost) and kept as focus segments. A short summary
     of them is added to the proactive check-in, so a check-in can be about
     what you are actually doing.
  2. Breaks. After BREAK_AFTER_MINUTES of continuous use (no keyboard/mouse
     gap of IDLE_RESET_MINUTES) JARVIS suggests a break, once per stretch.
  3. Stuck on an error (opt-in, sends screenshots). While a terminal or code
     editor has had focus for ERROR_CHECK_AFTER_MINUTES, the screen is
     checked every ERROR_CHECK_EVERY_MINUTES by a small vision model for a
     visible error. The same error still there on the next check means you
     are stuck, and JARVIS offers to help — once per error.

Nothing here talks by itself: it returns a Nudge, and main.py decides when it
is polite to say it (awake, not speaking, the user not mid-sentence).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Optional

SAMPLE_SECONDS = 15
BREAK_AFTER_MINUTES = 90
IDLE_RESET_MINUTES = 5
ERROR_CHECK_AFTER_MINUTES = 5
ERROR_CHECK_EVERY_MINUTES = 5
# Same error if the two one-line summaries are at least this similar.
SAME_ERROR_RATIO = 0.6
NUDGE_COOLDOWN_MINUTES = 20

DEV_APPS = ("Terminal", "iTerm", "Warp", "Code", "Visual Studio Code", "Cursor", "Xcode",
            "PyCharm", "IntelliJ", "Android Studio", "Sublime", "PowerShell", "cmd")

ERROR_PROMPT = (
    "Look at this screenshot of a developer's screen. Is an error, exception, "
    "stack trace or failing build/test output visible? Reply with JSON only: "
    '{"error": "<one line: the error type and message>"} or {"error": ""} if none.'
)


@dataclass
class Segment:
    app: str
    title: str
    start: float
    last: float

    @property
    def minutes(self) -> float:
        return (self.last - self.start) / 60.0


@dataclass
class Nudge:
    kind: str       # "break" | "stuck"
    text: str       # instruction for the model, never read aloud verbatim


def idle_seconds() -> float:
    """Seconds since the last keyboard/mouse input (macOS); 0 if unknown."""
    if sys.platform != "darwin":
        return 0.0
    try:
        out = subprocess.run(["ioreg", "-c", "IOHIDSystem"], capture_output=True,
                             text=True, timeout=3).stdout
        m = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', out)
        return int(m.group(1)) / 1e9 if m else 0.0
    except Exception:
        return 0.0


def _is_dev(app: str) -> bool:
    a = app.lower()
    return any(d.lower() in a for d in DEV_APPS)


def parse_error(reply: str) -> str:
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return ""
    try:
        return str(json.loads(m.group(0)).get("error") or "").strip()[:200]
    except Exception:
        return ""


def same_error(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= SAME_ERROR_RATIO


class Awareness:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self.segments: list[Segment] = []
        self._active_since: Optional[float] = None     # start of the current stretch of use
        self._break_nudged = False
        self._last_error_check = 0.0
        self._last_error = ""
        self._nudged_errors: list[str] = []
        self._last_nudge: dict[str, float] = {}

    # ── activity ──────────────────────────────────────────────────────────────

    def sample(self, app: str, title: str, idle: float) -> None:
        now = self._clock()
        if idle >= IDLE_RESET_MINUTES * 60:
            self._active_since = None           # away from the keyboard: stretch over
            self._break_nudged = False
        elif self._active_since is None:
            self._active_since = now
        if not app:
            return
        cur = self.segments[-1] if self.segments else None
        if cur and cur.app == app and cur.title == title:
            cur.last = now
        else:
            self.segments.append(Segment(app, title, now, now))
            del self.segments[:-20]

    def current(self) -> Optional[Segment]:
        return self.segments[-1] if self.segments else None

    def same_app_minutes(self) -> float:
        """How long the current app has had focus (any of its windows)."""
        if not self.segments:
            return 0.0
        app, end = self.segments[-1].app, self.segments[-1].last
        start = self.segments[-1].start
        for seg in reversed(self.segments[:-1]):
            if seg.app != app:
                break
            start = seg.start
        return (end - start) / 60.0

    def summary(self) -> str:
        """'Visual Studio Code — main.py (25 min); before that Google Chrome (5 min)'."""
        parts, seen = [], 0
        for seg in reversed(self.segments):
            if seg.minutes < 1 and seen:
                continue
            label = seg.app + (f" — {seg.title[:60]}" if seg.title and seg.title != seg.app else "")
            parts.append(f"{label} ({max(seg.minutes, 0):.0f} min)")
            seen += 1
            if seen == 3:
                break
        if not parts:
            return ""
        return parts[0] + ("; before that " + ", ".join(parts[1:]) if len(parts) > 1 else "")

    # ── nudges ────────────────────────────────────────────────────────────────

    def _cooled(self, kind: str) -> bool:
        return self._clock() - self._last_nudge.get(kind, -1e9) >= NUDGE_COOLDOWN_MINUTES * 60

    def mark_nudged(self, nudge: Nudge) -> None:
        self._last_nudge[nudge.kind] = self._clock()

    def break_nudge(self) -> Optional[Nudge]:
        if self._active_since is None or self._break_nudged or not self._cooled("break"):
            return None
        minutes = (self._clock() - self._active_since) / 60.0
        if minutes < BREAK_AFTER_MINUTES:
            return None
        self._break_nudged = True
        return Nudge("break",
                     f"The user has been working for about {minutes:.0f} minutes without a "
                     "break. In one short, friendly sentence, suggest a quick break. No tools.")

    def wants_error_check(self) -> bool:
        cur = self.current()
        if cur is None or not _is_dev(cur.app):
            self._last_error = ""
            return False
        if self.same_app_minutes() < ERROR_CHECK_AFTER_MINUTES:
            return False
        return self._clock() - self._last_error_check >= ERROR_CHECK_EVERY_MINUTES * 60

    def error_seen(self, error: str) -> Optional[Nudge]:
        """Feed one screen check's result. A Nudge when the same error has
        survived a whole check interval and has not been offered before."""
        self._last_error_check = self._clock()
        prev, self._last_error = self._last_error, error
        if not error or not same_error(prev, error):
            return None
        if any(same_error(error, e) for e in self._nudged_errors) or not self._cooled("stuck"):
            return None
        self._nudged_errors = (self._nudged_errors + [error])[-10:]
        mins = ERROR_CHECK_EVERY_MINUTES
        return Nudge("stuck",
                     f"The user has had this error on screen for at least {mins} minutes: "
                     f"\"{error}\". In one short sentence, mention you noticed it and offer to "
                     "take a look. Don't try to solve it unless they say yes. No tools.")
