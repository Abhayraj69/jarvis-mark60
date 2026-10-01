"""Speaking state, push-to-talk, interrupts, and text sent for JARVIS to say.
Mixed into JarvisLive (main.py).
"""

import asyncio
import time
from live.audio_analysis import _TAIL_MARGIN
from live.constants import ECHO_TAIL_SECONDS


class SpeechMixin:
    """Speaking state, push-to-talk, interrupts, and text sent for JARVIS to say."""

    def _tail_active(self) -> bool:
        """True while the speakers may still be finishing our last sentence."""
        return time.monotonic() < self._tail_until

    def set_speaking(self, value: bool, echo_guard: bool = True):
        with self._speaking_lock:
            was_speaking = self._is_speaking
            self._is_speaking = value
            if was_speaking and not value:
                self._mic_open_at = (time.monotonic() + ECHO_TAIL_SECONDS) if echo_guard else 0.0
        if was_speaking and not value:
            self._touch_activity()
        if value:
            self._tail_until = 0.0
        else:
            # Hold the guard open across the device's own output latency plus a
            # margin for the room. The microphone is NOT muted during it — the
            # guard still lets a genuine reply through, so answering instantly
            # still works. Only our own echo is dropped.
            self._tail_until = time.monotonic() + self._out_latency + _TAIL_MARGIN
        if not value:
            # The echo history is deliberately NOT cleared here: the tail above
            # still needs it to recognise our own voice. It is dropped when the
            # tail expires. What the guard learned about the room always stays.
            self._out_level = 0.0
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self._show_idle()

    def _show_idle(self) -> None:
        """The HUD's resting state. Every "done speaking / tool finished" path
        used to set LISTENING unconditionally, so JARVIS looked awake again a
        second after "bye Jarvis" while it was in fact asleep (2026-10-01)."""
        self.ui.set_state("SLEEPING" if (self._wake_enabled and not self._awake) else "LISTENING")

    def set_push_to_talk(self, enabled: bool) -> str:
        """Turn hold-to-talk on or off. Returns the scope actually achieved."""
        from core.hotkey import PushToTalk

        self._ptt_enabled = bool(enabled)
        self._ptt_held = False
        if not enabled:
            if self._ptt is not None:
                self._ptt.stop()
                self._ptt = None
            return "off"

        if self._ptt is None:
            self._ptt = PushToTalk(self._on_ptt)
        scope = self._ptt.start()
        # A window-scoped chord is a real limitation, not a detail — say it once
        # in the log so nobody wonders why it does nothing while another app is
        # focused. Reporting it must never be able to undo the thing it reports.
        try:
            self.ui.write_log(
                f"SYS: Push-to-talk on — hold {self._ptt.label}"
                + ("." if scope == "global"
                   else " (works while this window is focused)."))
        except Exception:
            pass
        return scope

    def _on_ptt(self, held: bool) -> None:
        """Chord pressed or released — may arrive on the hotkey thread."""
        self._ptt_held = held
        if held:
            # Holding the key is also a way to wake it, so push-to-talk works
            # without having to say the wake word first.
            if self._wake_enabled and not self._awake:
                self._awake = True
                self._last_user_speech = time.monotonic()
        try:
            self.ui.set_state("LISTENING" if held else "SLEEPING")
        except Exception:
            pass
    def _local_control_state(self) -> dict:
        """Polled by the Arc Sentinel widget (core/local_control.py) to keep
        its mic/interrupt buttons in sync with the real session — called from
        the control server's own thread, so only cheap, already-thread-safe
        reads belong here (a lock-guarded bool, a plain property)."""
        with self._speaking_lock:
            speaking = self._is_speaking
        return {
            "muted":    self.ui.muted,
            "speaking": speaking,
            "awake":    self._awake,
        }

    def interrupt(self) -> None:
        """Stop JARVIS mid-speech: drain queued audio and open mic immediately."""
        # Only discard incoming audio if the server is still generating this
        # reply — its turn_complete is what clears the flag again. Interrupting
        # while just the locally buffered tail was playing (the reply already
        # complete), or while idle, used to leave the flag set with no
        # turn_complete coming, so the NEXT answer was silently thrown away:
        # JARVIS "randomly stopped responding" after Esc / the stop button.
        self._interrupted = self._generating
        q = self.audio_in_queue
        if q:
            drained = 0
            while True:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break
            if drained:
                print(f"[JARVIS] ✋ Interrupted — {drained} audio chunks discarded")
        self.set_speaking(False, echo_guard=False)
        # The words we were about to mouth are never going to be spoken now.
        self._visemes.reset()
        self._play_cursor = 0.0     # next batch starts a fresh timeline
        if self._turn_done_event:
            self._turn_done_event.clear()

        # Local Mode barge-in: tell the streaming consumer thread to stop
        # reading from the LLM, drop any sentences already queued for TTS,
        # and cut audio that's playing right now.
        self._local_stream_cancel.set()
        tts_q = self._local_tts_queue
        if tts_q is not None:
            while True:
                try:
                    tts_q.get_nowait()
                except Exception:
                    break
            try:
                tts_q.put_nowait(None)
            except Exception:
                pass
        if self._local_tts_player is not None:
            self._local_tts_player.stop()

        self.ui.write_log("SYS: Interrupted — listening...")

    def speak(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    def plugin_say(self, instruction: str) -> None:
        """
        Thread-safe speech channel for plugins: lets a plugin ask JARVIS to
        say something short WHILE its run() is still executing (plugins block
        their executor thread, so they can't speak through the tool response
        until they finish). The instruction is injected into the Live session
        exactly like a proactive check-in; Gemini phrases it naturally in
        English. Silently a no-op when no session is connected.
        """
        loop = getattr(self, "_loop", None)
        if not loop or not self.session:
            return

        async def _say():
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": instruction}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[PluginSay] {e}")

        try:
            asyncio.run_coroutine_threadsafe(_say(), loop)
        except Exception as e:
            print(f"[PluginSay] {e}")
