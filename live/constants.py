"""Tunables and fixed values shared by the Live session (main.py and live/*)."""

import re
import sys
from pathlib import Path

# Spoken fast commands (core/fast_intent.py on the live transcript): how long
# the transcript must stay unchanged before a match runs — long enough that
# "open chrome" is not acted on when the user is still saying "...and go to
# gmail", short enough to beat the model by a couple of seconds.
FAST_VOICE_SETTLE_SECONDS = 0.6
# A model tool call this soon after a spoken fast command, for the same thing,
# is answered "already done" instead of running it again (pause is a toggle).
FAST_VOICE_DEDUPE_SECONDS = 10.0

# First wake of the day: give the morning briefing if the user says nothing
# else for this long after "Hey Jarvis" (if they do, it was for a command).
BRIEF_AFTER_WAKE_SECONDS = 4.0

# How long the assistant stays awake with no user speech before it auto-sleeps
# again (wake-word mode only).
WAKE_SLEEP_TIMEOUT = 120.0   # seconds (2 minutes)
# Default listening window after the last exchange (Plugin Settings →
# LISTENING). Two minutes awake and idle is where the random replies came
# from — the room's chatter, a video, a TV — so it goes back to needing
# "Hey Jarvis" much sooner.
FOLLOW_UP_SECONDS = 45.0
# 100 ms of 16 kHz int16 silence, sent whenever nothing else has gone to the
# Live session for KEEPALIVE_IDLE_SECONDS, to keep it open (see _run_sleep_watch).
_KEEPALIVE_SILENCE = bytes(3200)
# The server drops a session after ~30 s with no input (1008). Asleep is not
# the only time the mic sends nothing: muted, push-to-talk released, and a
# long reply playing all gate it too.
KEEPALIVE_IDLE_SECONDS = 10.0
# A 1008 on a session that had been up at least this long is the idle drop
# above: reconnect at once. One sooner than that is more likely the server
# rejecting the session itself, so it takes the normal backoff instead of a
# tight reconnect loop.
_IDLE_DROP_MIN_UPTIME = 20.0

# _enter_standby() debounce: the mic keeps capturing in real time right through
# the awake->asleep transition, so the wake detector's very first frames are
# the acoustic tail of the "bye jarvis" the user just finished saying — room
# reverb and decay, not silence. Without a settle window, that tail alone can
# score as "Hey Jarvis" (same "...jarvis" sound) and re-wake JARVIS instantly.
WAKE_SETTLE_SECONDS = 1.0
# After a genuine wake, ignore a second shutdown_jarvis call for this long.
# Backstops the same echo/tail problem from the other direction: if a stray
# "jarvis"-ish sound both re-wakes the detector AND reaches the model as
# audio, the model can reflexively re-call shutdown_jarvis a moment later
# with no new, deliberate goodbye from the user. Short enough that a real,
# fast second "bye jarvis" from the user is not a realistic case it blocks.
STANDBY_REENTRY_GUARD_SECONDS = 2.5
# The model decides when to call shutdown_jarvis, and with proactive audio on
# it can do so with no request at all (a goodbye said to someone else, TV
# audio, its own voice leaking back from the speakers). Only honour it when
# the user actually spoke or typed within this window.
SHUTDOWN_USER_WINDOW_SECONDS = 10.0
# ...and only when what they said actually reads as a goodbye. On Sept 30 the
# model called shutdown_jarvis 14 times, mostly on remarks that were not one.
# English and Hindi/Hinglish; the HUD's sleep button covers anything else.
_FAREWELL_RE = re.compile(
    r"\b(bye|goodbye|good ?night|see you|that'?s all|go to sleep|sleep now|"
    r"stop listening|shut ?down|end (?:the |this )?session|sleep,? j\w{3,6}|j\w{3,6},? (?:go to )?sleep|"
    r"alvida|so ja(o)?|chalo bye)\b"
    r"|अलविदा|बाय|शुभ रात्रि|सो जाओ",
    re.IGNORECASE,
)
# "Hey Jarvis" said while already awake reaches the model as "Bye, Jarvis"
# often enough to put it to sleep. The wake-word detector cannot settle it — it
# scores "Bye Jarvis" as high as "Hey Jarvis" (0.97 on 2026-10-01), so using it
# blocked every real goodbye. Instead a sleep request is checked against a
# local Whisper transcript of the last SLEEP_CHECK_SECONDS of the user's audio.
SLEEP_CHECK_SECONDS = 8.0
# A whole utterance that is only a goodbye — handled locally, not left to the
# model: the backup Live model heard "Bye Jarvis" as "By javas" and simply did
# not call shutdown_jarvis (2026-10-01, 19:41). Strict on purpose: "by the way"
# or "bye, I'll call you" do not match.
_GOODBYE_UTTERANCE = re.compile(
    r"^\W*(?:(?:ok(?:ay)?|alright|right)\W+)?"
    r"(?:bye(?:\W+bye)?|by|goodbye|good\W*night|sleep|go\W+to\W+sleep|see\W+you|"
    r"that'?s\W+all|end\W+(?:the\W+|this\W+)?session)"
    r"(?:\W+(?:j\w{2,7}|now|for\W+now|then|later))*\W*$",
    re.IGNORECASE,
)
# Laptop speakers keep playing for a moment after the last chunk is handed to
# PortAudio. Re-opening the mic the instant playback "ends" streams JARVIS's
# own trailing words back to Gemini, which reads them as the user talking —
# a phantom turn, a barge-in, or (if the tail was "…bye") a phantom goodbye.
ECHO_TAIL_SECONDS = 0.8

def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent

BASE_DIR        = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"

CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000 
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024
