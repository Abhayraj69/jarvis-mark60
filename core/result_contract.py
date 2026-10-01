"""
core/result_contract.py — one shape for every tool result, and a detector for
the lie that shape exists to prevent.

WHY THIS EXISTS
    Tool handlers return free text: "Moved: a.txt -> Documents/", "Could not
    capture the screen: ...", "Tool 'x' failed: ...", "Unknown action". The
    Live model got that string and nothing else, and a speech model under
    time pressure reads "Could not find file" and says "Done, sir" often
    enough that users stopped trusting it. Two fixes, both here:

    1. classify() wraps a handler's string as {"ok", "summary", "detail"}.
       The model no longer has to infer success from prose — ok=false is a
       field it can see, and core/prompt.txt tells it what to do with one.

    2. FalseSuccessTracker watches what JARVIS *says* after a tool returned
       ok=false. If none of the speech that follows admits the failure (no
       "couldn't", "failed", "didn't" ... in any configured language), that
       is a hallucinated success, and core/telemetry.py records it so the
       rate can be counted instead of felt.

    The tracker is deliberately not tied to a single turn: with the Live
    API the model's acknowledgement ("Moving it now") often arrives as its
    own turn_complete, sometimes *after* the tool has already failed, because
    the receive loop was busy running the tool. Judging that first fragment
    alone would flag every acknowledgement as a lie. So the tracker collects
    everything spoken after the failure and decides only when the user
    speaks again or a deadline passes.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

# A handler result starting with one of these (case-insensitive) is a failure.
# Kept as prefixes on purpose: "Could not" at the start is a failure report,
# "could not" in the middle of a summary ("the author could not prove...")
# is content.
FAILURE_PREFIXES = (
    "tool '",            # _dispatch_tool's own "Tool 'x' failed: ..."
    "could not", "couldn't", "cannot", "can't", "unable to",
    "unknown", "error", "failed", "failure",
    "not available", "no such", "not found", "invalid",
    "action '",          # ActionRegistry: "Action 'x' is not available."
)
# ...unless it is one of these, which are status reports, not failures.
NON_FAILURE_PREFIXES = (
    "[confirmation_pending]",
    "no missed questions",      # study_progress: empty history is fine
    "no quiz history",
    "not enough quiz history",
    "no topics are being monitored",
    "nothing new was captured",
)

SUMMARY_MAX = 160

# Words/phrases in JARVIS's spoken reply that count as "admitting" a failure.
# Lower-case, substring match. English first; a few common languages so a
# Turkish or Spanish reply that owns the failure is not flagged as a lie.
# Extend freely — a false negative here only means one lie goes uncounted,
# a false positive means an honest reply is counted as a lie.
FAILURE_WORDS: dict[str, tuple[str, ...]] = {
    "en": ("couldn't", "could not", "can't", "cannot", "unable", "failed", "failure",
           "didn't", "did not", "wasn't able", "not able", "problem", "error",
           "refused", "denied", "no luck", "not found", "isn't available", "not available",
           "unfortunately", "afraid"),
    "tr": ("olmadı", "başarısız", "hata", "yapamadım", "bulamadım", "mümkün değil", "maalesef"),
    "es": ("no pude", "no puedo", "error", "falló", "fallo", "imposible", "lo siento"),
    "fr": ("impossible", "erreur", "échoué", "je n'ai pas pu", "désolé", "malheureusement"),
    "de": ("fehler", "konnte nicht", "kann nicht", "fehlgeschlagen", "leider"),
    "hi": ("नहीं हो", "नहीं कर", "विफल", "त्रुटि", "माफ़"),
}

FALSE_SUCCESS_DEADLINE_S = 45.0


@dataclass
class ToolOutcome:
    ok: bool
    summary: str
    detail: str

    def as_response(self) -> dict:
        return {"ok": self.ok, "summary": self.summary, "detail": self.detail}


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


def classify(tool_name: str, result) -> ToolOutcome:
    """Wrap a handler's raw result. Strings are inspected for failure
    prefixes; dicts that already carry "ok" pass through; anything else is
    stringified and treated as success (the handler did not complain)."""
    if isinstance(result, ToolOutcome):
        return result
    if isinstance(result, dict) and "ok" in result:
        detail = str(result.get("detail") or result.get("result") or result.get("summary") or "")
        summary = str(result.get("summary") or _first_line(detail))[:SUMMARY_MAX]
        return ToolOutcome(bool(result["ok"]), summary, detail)

    text = "" if result is None else str(result)
    stripped = text.strip()
    lowered = stripped.lower()

    ok = True
    if not stripped:
        text, stripped = "Done.", "Done."
    elif lowered.startswith(NON_FAILURE_PREFIXES):
        ok = True
    elif lowered.startswith(FAILURE_PREFIXES):
        ok = False
    elif lowered.startswith(f"tool '{tool_name.lower()}'"):
        ok = False

    summary = _first_line(stripped)
    if len(summary) > SUMMARY_MAX:
        summary = summary[:SUMMARY_MAX - 1].rstrip() + "…"
    return ToolOutcome(ok, summary, text)


def admits_failure(spoken: str) -> bool:
    """True if the spoken text contains any failure word in any language."""
    low = (spoken or "").lower()
    if not low:
        return False
    return any(word in low for words in FAILURE_WORDS.values() for word in words)


@dataclass
class FalseSuccessVerdict:
    tools: list[str]
    spoken: str
    false_success: bool


@dataclass
class FalseSuccessTracker:
    """Collects JARVIS's speech after a failed tool and decides later.

    register_failure(tool)  — a tool returned ok=false
    note_output(text)       — JARVIS said something (any turn_complete)
    conclude()              — the user spoke again: decide now
    poll()                  — call periodically: decides once the deadline passes

    conclude()/poll() return a FalseSuccessVerdict (or None if nothing was
    pending) and reset. A verdict with false_success=True means the tool
    failed and nothing JARVIS said afterwards admitted it.
    """
    deadline_s: float = FALSE_SUCCESS_DEADLINE_S
    _tools: list[str] = field(default_factory=list)
    _spoken: list[str] = field(default_factory=list)
    _since: Optional[float] = None

    @property
    def pending(self) -> bool:
        return bool(self._tools)

    @property
    def since(self) -> Optional[float]:
        """Monotonic time the oldest pending failure was registered."""
        return self._since

    def register_failure(self, tool: str, now: Optional[float] = None) -> None:
        if not self._tools:
            self._since = time.monotonic() if now is None else now
            self._spoken = []
        self._tools.append(tool)

    def note_output(self, text: str) -> None:
        if self._tools and text and text.strip():
            self._spoken.append(text.strip())

    def conclude(self) -> Optional[FalseSuccessVerdict]:
        if not self._tools:
            return None
        spoken = " ".join(self._spoken)
        verdict = FalseSuccessVerdict(
            tools=list(self._tools), spoken=spoken,
            false_success=not admits_failure(spoken),
        )
        self._tools, self._spoken, self._since = [], [], None
        return verdict

    def poll(self, now: Optional[float] = None) -> Optional[FalseSuccessVerdict]:
        if not self._tools or self._since is None:
            return None
        now = time.monotonic() if now is None else now
        if now - self._since >= self.deadline_s:
            return self.conclude()
        return None


_WS = re.compile(r"\s+")


def snippet(text: str, limit: int = 240) -> str:
    text = _WS.sub(" ", text or "").strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"
