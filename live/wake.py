"""Wake word, sleep and auto-sleep: when JARVIS listens. Mixed into JarvisLive
(main.py).
"""

import asyncio
import time
import numpy as np
from datetime import datetime
from memory.config_manager import (
    get_brief_enabled, save_wake_word_enabled, get_plugin_config,
)
from core.wake_word import (
    WakeWordDetector, is_ready as wake_is_ready,
    install_and_download as wake_install,
)
from live.constants import (
    BRIEF_AFTER_WAKE_SECONDS, FOLLOW_UP_SECONDS, KEEPALIVE_IDLE_SECONDS,
    SEND_SAMPLE_RATE, SHUTDOWN_USER_WINDOW_SECONDS, SLEEP_CHECK_SECONDS,
    STANDBY_REENTRY_GUARD_SECONDS, WAKE_SETTLE_SECONDS, _FAREWELL_RE,
    _KEEPALIVE_SILENCE,
)


class WakeMixin:
    """Wake word, sleep and auto-sleep: when JARVIS listens."""

    # ── Wake word: state machine ─────────────────────────────────────────────

    def _wake_state(self) -> dict:
        # A loaded, running detector is definitively ready; otherwise fall back
        # to the cheap on-disk model-file check (no Model construction).
        ready = bool(self._wake_detector and self._wake_detector.ready) or wake_is_ready()
        return {"enabled": self._wake_enabled, "awake": self._awake, "ready": ready}

    def _ensure_wake_detector(self) -> bool:
        """Load the detector once (model loads on first start). Idempotent."""
        if self._wake_detector is None:
            self._wake_detector = WakeWordDetector(
                on_detect=self._on_wake_detected,
                logger=lambda m: print(f"[Wake] {m}"),
                notify=lambda m: self.ui.write_log(f"SYS: {m}"),
            )
        if not self._wake_detector.ready:
            return self._wake_detector.start()
        return True

    def _on_wake_detected(self) -> None:
        """Called from the detector thread when 'Hey Jarvis' is heard."""
        if self._awake:
            # Already awake: nothing to wake, but remember it — the model may
            # be about to hear the same words as "bye Jarvis" (see _standby_allowed).
            self._wake_heard_at = time.monotonic()
            return
        self.wake(reason="wake word")

    def wake(self, reason: str = "wake word") -> None:
        if self._awake:
            return
        self._awake = True
        # See STANDBY_REENTRY_GUARD_SECONDS: a stray echo of the acoustic
        # tail that just caused this wake can otherwise read as a second,
        # genuine "bye jarvis" a moment later and immediately re-sleep.
        self._standby_reentry_guard_until = time.monotonic() + STANDBY_REENTRY_GUARD_SECONDS
        if self._standby_forced_wake:
            # Undo _enter_standby()'s temporary override now that its job is
            # done, so a user who never turned wake-word mode on goes back to
            # always-listening instead of being left in permanent sleep-until-
            # "Hey Jarvis" mode by a single "bye jarvis".
            self._wake_enabled       = self._standby_restore_wake_enabled
            self._standby_forced_wake = False
        self._last_user_speech = time.monotonic()   # start the auto-sleep clock now
        self._touch_activity()
        if not self.ui.muted:
            self._show_idle()
        self.ui.write_log(f"SYS: Awake — {reason}.")
        self._maybe_brief_on_wake()

    def _maybe_brief_on_wake(self) -> None:
        """Wake-word users never got the startup briefing (JARVIS comes up
        asleep). Give it on the first wake of the day instead — unless the
        wake was for a command, in which case try again next wake."""
        today = datetime.now().strftime("%Y-%m-%d")
        loop = self._loop
        if self._briefing_day == today or not get_brief_enabled() or loop is None or not self.session:
            return
        self._briefing_day = today
        woke_at = self._last_user_speech

        async def _later():
            await asyncio.sleep(BRIEF_AFTER_WAKE_SECONDS)
            if self._last_user_speech > woke_at + 0.5 or not self._awake or not self.session:
                self._briefing_day = ""        # they had something to say; next time
                return
            await self._send_startup_briefing()
        try:
            asyncio.run_coroutine_threadsafe(_later(), loop)
        except Exception:
            self._briefing_day = ""

    def _enter_standby(self, reason: str = "bye jarvis") -> None:
        """'bye jarvis' (or any explicit goodbye) no longer ends the process —
        it mutes. Speech is cut off immediately, JARVIS stops treating mic
        audio or typed text as commands, and it waits silently for "Hey
        Jarvis" to resume — the exact same _awake gate that already drives
        the wake-word auto-sleep timeout (see sleep()/_listen_audio), so
        "muted" means one consistent thing everywhere in the app rather than
        a second, parallel mechanism.

        Works even if the user never turned wake-word mode on: it force-arms
        the detector and flips _wake_enabled on for the duration of standby
        (wake() restores it) so the existing awake/asleep gates, the WAKE NOW
        button, and a live reconnect all treat this session correctly either
        way.

        Guarded against the two ways this loop was observed re-triggering
        itself right after "bye jarvis": (1) STANDBY_REENTRY_GUARD_SECONDS
        ignores a second call arriving immediately after a wake — that's an
        echo, not a new goodbye; (2) WAKE_SETTLE_SECONDS (applied in the mic
        callback / _local_wake_wait, not here) keeps the just-armed detector
        from hearing the tail of THIS utterance as a fresh "Hey Jarvis"."""
        if time.monotonic() < self._standby_reentry_guard_until:
            self.ui.write_log(
                "SYS: Ignoring an immediate repeat 'bye jarvis' — likely an "
                "echo of the last one."
            )
            return
        self.interrupt()   # stop mid-sentence speech now — no trailing audio
        if not self._wake_enabled:
            self._standby_restore_wake_enabled = self._wake_enabled
            self._wake_enabled        = True
            self._standby_forced_wake = True
        detector_ready = self._ensure_wake_detector()
        if detector_ready and self._wake_detector is not None:
            # Without this, "bye jarvis" only behaves correctly the FIRST
            # time per process: the detector's internal audio buffer is left
            # frozen mid-"hey jarvis" from whatever wake last fed it, and a
            # few fresh frames layered on top of that stale buffer can score
            # as an immediate false "Hey Jarvis" the moment feeding resumes
            # (see WakeWordDetector.start()'s docstring for the mechanism).
            self._wake_detector.reset()
        self.sleep(reason=reason)
        self._wake_feed_gate_open_at = time.monotonic() + WAKE_SETTLE_SECONDS
        if not detector_ready:
            self.ui.write_log(
                "SYS: Note — the 'Hey Jarvis' wake model isn't downloaded, so "
                "I won't hear you to wake back up. Use the WAKE NOW button in "
                "the HUD, or set up wake word once in ⚙ → WAKE WORD."
            )

    def _touch_activity(self) -> None:
        """Reset the auto-sleep clock. Called for anything that means the
        conversation is live: user speech, typed text, JARVIS finishing a
        reply, a tool starting or finishing. The clock used to move only on
        user speech, so JARVIS fell asleep right after finishing a long answer
        or tool run, and never counted typed commands at all."""
        self._last_activity = time.monotonic()

    def _note_user_input(self, text: str = "") -> None:
        self._last_user_speech = time.monotonic()
        if text:
            self._last_user_text = text
        self._touch_activity()

    def _sleep_reason(self) -> str:
        """Say what the model heard when it chose to sleep, so a surprise
        sleep in the log shows its trigger instead of a generic 'bye jarvis'."""
        heard = self._last_user_text.strip()
        return f'you said "{heard[:80]}"' if heard else "bye jarvis"

    def _remember_mic(self, block) -> None:
        b = np.array(block, dtype=np.int16).reshape(-1)
        self._recent_mic.append(b)
        self._recent_mic_len += len(b)
        limit = int(SLEEP_CHECK_SECONDS * SEND_SAMPLE_RATE)
        while self._recent_mic and self._recent_mic_len - len(self._recent_mic[0]) >= limit:
            self._recent_mic_len -= len(self._recent_mic.popleft())

    def _warm_sleep_check(self) -> None:
        try:
            if self._sleep_stt is None:
                from core.stt import WhisperSTT
                self._sleep_stt = WhisperSTT("tiny", "en")
        except Exception as e:
            print(f"[JARVIS] local sleep check unavailable: {e}")
            self._sleep_stt_failed = True

    def _local_transcript(self) -> str | None:
        """Whisper (tiny, on this Mac) over the user's last few seconds. None
        when faster-whisper is missing or fails — the caller then falls back
        to Gemini's own transcript."""
        if self._sleep_stt_failed or not self._recent_mic:
            return None
        try:
            if self._sleep_stt is None:
                from core.stt import WhisperSTT
                self._sleep_stt = WhisperSTT("tiny", "en")
            audio = np.concatenate(list(self._recent_mic)).astype(np.float32) / 32768.0
            text = self._sleep_stt.transcribe(audio).strip()
            return text or None     # heard nothing (e.g. spoke over JARVIS): use Gemini's
        except Exception as e:
            print(f"[JARVIS] local sleep check unavailable: {e}")
            self._sleep_stt_failed = True
            return None

    def _standby_allowed(self, local_text: str | None = None) -> bool:
        """Guard for the model's shutdown_jarvis call — see
        SHUTDOWN_USER_WINDOW_SECONDS. `local_text`: a local Whisper transcript
        of the last few seconds, which settles "hey" vs "bye" when Gemini's
        transcript got it wrong."""
        print(f"[JARVIS] 🛡️ sleep request — Gemini heard {self._last_user_text[:80]!r}, "
              f"local check heard {local_text!r}")
        if local_text is not None:
            if _FAREWELL_RE.search(local_text):
                return True
            self.ui.write_log(f'SYS: Ignored a sleep request — I heard "{local_text[:60]}", '
                              "not a goodbye.")
            print("[JARVIS] 🛡️ shutdown_jarvis ignored: local transcript is not a goodbye")
            return False
        if (time.monotonic() - self._last_user_speech) > SHUTDOWN_USER_WINDOW_SECONDS:
            self.ui.write_log(
                "SYS: Ignored a sleep request — you hadn't said anything just before it."
            )
            print("[JARVIS] 🛡️ shutdown_jarvis ignored: no recent user input")
            return False
        if not _FAREWELL_RE.search(self._last_user_text or ""):
            self.ui.write_log(
                f'SYS: Ignored a sleep request — "{(self._last_user_text or "")[:60]}" '
                "isn't a goodbye. Say 'bye Jarvis' or tap sleep."
            )
            print("[JARVIS] 🛡️ shutdown_jarvis ignored: last input was not a goodbye")
            return False
        return True

    def sleep(self, reason: str = "timeout") -> None:
        if not self._awake:
            return
        self._awake = False
        self.set_speaking(False)
        self.ui.set_state("SLEEPING")
        self.ui.write_log(f"SYS: Sleeping — {reason}. Say 'Hey Jarvis' to wake me.")

    @staticmethod
    def _follow_up_seconds() -> float:
        try:
            v = float(get_plugin_config("listening").get("follow_up_seconds", FOLLOW_UP_SECONDS))
            return max(10.0, v)
        except (TypeError, ValueError):
            return FOLLOW_UP_SECONDS

    async def _maybe_send_clock(self) -> None:
        """Once a minute, at a quiet moment, tell the model the time without
        asking for a reply — so "what time is it?" is answered at once from
        context instead of costing a get_time round trip (~2 s of silence)."""
        now = datetime.now()
        minute = now.strftime("%H:%M")
        if minute == self._clock_sent or self.session is None:
            return
        with self._speaking_lock:
            speaking = self._is_speaking
        if (speaking or self._generating or self._tools_running
                or time.monotonic() - self._last_user_speech < 3):
            return
        try:
            await self.session.send_client_content(
                turns={"role": "user", "parts": [{"text":
                    f"[CLOCK] {now.strftime('%I:%M %p, %A %B %d, %Y')}"}]},
                turn_complete=False,
            )
            self._clock_sent = minute
        except Exception as e:
            print(f"[JARVIS] clock update failed: {e}")

    async def _run_sleep_watch(self) -> None:
        """Auto-sleep after the configured silence window (wake-word mode only),
        and keep the Live session alive while asleep."""
        while True:
            await asyncio.sleep(5)
            await self._maybe_send_clock()
            # The Live server drops a session that has had no input for ~30 s
            # ("1008 The operation was aborted"). The failure only surfaced on
            # the next thing the user said, eating it. Whenever nothing has
            # been sent for a while — asleep, muted, push-to-talk released, a
            # long reply playing — a tenth of a second of digital silence
            # keeps the session open. It is never microphone audio.
            if (self.session is not None and self.out_queue is not None
                    and time.monotonic() - self._last_realtime_send > KEEPALIVE_IDLE_SECONDS):
                try:
                    self.out_queue.put_nowait(
                        {"data": _KEEPALIVE_SILENCE, "mime_type": "audio/pcm"})
                except Exception:
                    pass
            if self._wake_enabled and not self._awake:
                continue
            if not self._wake_enabled:
                continue
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking or self._tools_running > 0 or self._tail_active():
                continue
            timeout = self._follow_up_seconds()
            if (time.monotonic() - self._last_activity) > timeout:
                self.sleep(reason=f"no conversation for {timeout:.0f} seconds")

    # ── Wake word: UI callbacks (called from the Qt thread) ──────────────────

    def _ui_wake_toggle(self, enable: bool) -> str:
        """Enable/disable wake word from the settings UI. Returns a status token:
        'enabled' | 'disabled' | 'need_download'."""
        if enable:
            if not wake_is_ready():
                return "need_download"
            self._wake_enabled = True
            save_wake_word_enabled(True)
            self._ensure_wake_detector()
            self.sleep(reason="wake word enabled")
            return "enabled"
        else:
            self._wake_enabled = False
            save_wake_word_enabled(False)
            self.wake(reason="wake word disabled")
            return "disabled"

    def _ui_wake_manual(self) -> None:
        """Manual sleep/wake button in the UI."""
        if not self._wake_enabled:
            return
        if self._awake:
            self.sleep(reason="you tapped sleep")
        else:
            self.wake(reason="you tapped wake")

    def _ui_wake_install(self) -> tuple[bool, str]:
        """Download openwakeword + the model (runs in a UI worker thread)."""
        # Triggered by the user pressing the button, so its progress is exactly
        # what they are waiting to see.
        return wake_install(logger=lambda m: print(f"[Wake] {m}"),
                            notify=lambda m: self.ui.write_log(f"SYS: {m}"))
