"""
Event-loop stall detector.

Everything in the Live session — sending mic audio, receiving replies,
playback — shares one asyncio loop. If any code on it blocks (a synchronous
call that should have been in a thread), all of it stops at once: the mic
queue fills, audio is dropped, and the log only shows the symptom.

A background thread pings the loop every CHECK_SECONDS. When a ping goes
unanswered for STALL_SECONDS it prints the loop thread's current stack — the
exact line that is holding everything up — once per stall, and how long the
stall lasted when the loop comes back. It costs one no-op callback per tick.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import traceback

CHECK_SECONDS = 0.5
STALL_SECONDS = 1.0


class LoopWatchdog:
    def __init__(self, loop: asyncio.AbstractEventLoop, log=print):
        self._loop = loop
        self._log = log
        self._loop_thread_id = threading.get_ident()   # construct on the loop's thread
        self._last_beat = time.monotonic()
        self._stop = threading.Event()

    def _beat(self) -> None:
        self._last_beat = time.monotonic()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="loop-watchdog").start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        stalled_since = None
        while not self._stop.wait(CHECK_SECONDS):
            try:
                self._loop.call_soon_threadsafe(self._beat)
            except RuntimeError:
                return                              # loop closed
            lag = time.monotonic() - self._last_beat
            if lag >= STALL_SECONDS + CHECK_SECONDS:
                if stalled_since is None:
                    stalled_since = self._last_beat
                    frame = sys._current_frames().get(self._loop_thread_id)
                    stack = "".join(traceback.format_stack(frame)[-12:]) if frame else "(no frame)"
                    self._log(f"[Watchdog] ⚠️ Event loop blocked for {lag:.1f}s — it is stuck here:\n{stack}")
            elif stalled_since is not None:
                self._log(f"[Watchdog] Event loop free again after "
                          f"{time.monotonic() - stalled_since:.1f}s.")
                stalled_since = None
