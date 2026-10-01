"""Device commands run locally (typed or spoken) and the local goodbye. Mixed
into JarvisLive (main.py).
"""

import asyncio
import time
from memory.config_manager import get_plugin_config, get_plugin_setting
from core import fast_intent, telemetry
from live.constants import FAST_VOICE_SETTLE_SECONDS, _FAREWELL_RE, _GOODBYE_UTTERANCE


class FastCommandsMixin:
    """Device commands run locally (typed or spoken) and the local goodbye."""

    # ── fast local commands ─────────────────────────────────────────────────

    def _try_fast_intent(self, text: str) -> bool:
        """Run `text` locally if it is an unambiguous device command, skipping
        the model round trip entirely. Returns True if it was handled here.

        Thread-safe: called from the Qt thread (typed box) and from the async
        loop (phone). Detection itself is a handful of regex matches, so it is
        cheap enough to attempt on every typed command; a miss returns False
        and the caller takes the normal path unchanged."""
        if not self._loop:
            return False
        if not get_plugin_setting("fast_commands", "enabled", True):
            return False
        intent = fast_intent.detect(text)
        if intent is None:
            return False
        asyncio.run_coroutine_threadsafe(self._run_fast_intent(intent), self._loop)
        return True

    def _schedule_local_goodbye(self, text: str) -> None:
        """A transcript that is nothing but a goodbye puts JARVIS to sleep
        here, after the same short pause as a fast command and a local
        Whisper check that it was not "Hey Jarvis" — without waiting for the
        model to decide to call shutdown_jarvis (which it sometimes doesn't)."""
        self._goodbye_seq += 1
        if not _GOODBYE_UTTERANCE.match(text or ""):
            return
        seq = self._goodbye_seq

        async def _later():
            await asyncio.sleep(FAST_VOICE_SETTLE_SECONDS)
            if seq != self._goodbye_seq or not self._awake:
                return                      # more speech, or already asleep
            local = await asyncio.to_thread(self._local_transcript)
            if local is not None and not _FAREWELL_RE.search(local) \
                    and not _GOODBYE_UTTERANCE.match(local):
                print(f"[JARVIS] 🛡️ goodbye {text!r} not confirmed — local check heard {local!r}")
                return
            if seq != self._goodbye_seq or not self._awake:
                return
            print(f"[JARVIS] 👋 Goodbye heard ({text!r}, local {local!r}) — going to sleep.")
            self._enter_standby(reason=f'you said "{text.strip()[:40]}"')
        asyncio.create_task(_later())

    def _schedule_fast_voice(self, text: str) -> None:
        """Called on every live transcript update. If the whole utterance so far
        is a fast command and nothing more is heard for
        FAST_VOICE_SETTLE_SECONDS, run it locally instead of waiting ~3 s for
        the model to decide to call the same tool."""
        self._fast_voice_seq += 1
        if self._fast_voice_ran_turn:
            return
        intent = fast_intent.detect(text)
        if intent is None or not fast_intent.voice_safe(intent):
            return
        cfg = get_plugin_config("fast_commands")
        if not cfg.get("enabled", True) or not cfg.get("voice", True):
            return
        seq = self._fast_voice_seq

        async def _later():
            await asyncio.sleep(FAST_VOICE_SETTLE_SECONDS)
            if seq != self._fast_voice_seq or self._fast_voice_ran_turn:
                return              # more speech, or the model called a tool
            self._fast_voice_ran_turn = True
            self._fast_voice_done = (intent, time.monotonic())
            print(f"[JARVIS] ⚡ Spoken fast command: {text!r} → {intent.tool}{intent.args}")
            await self._run_fast_intent(intent, tell_model=False)
        asyncio.create_task(_later())

    async def _run_fast_intent(self, intent: "fast_intent.Intent", tell_model: bool = True) -> None:
        turn = telemetry.start_turn("fast_intent")
        turn.set_fast_intent()
        self.ui.set_state("THINKING")
        with turn.tool_span(intent.tool):
            result = await self._dispatch_tool(intent.tool, dict(intent.args))
        turn.mark("model_done")   # no model call on this path — marks the shortcut's own latency
        turn.finish()
        if not self.ui.muted:
            self._show_idle()

        failed = result.startswith(("Tool '", "Unknown tool:", "Action '"))
        self.ui.write_log(f"ERR: {result}" if failed else f"JARVIS: {intent.reply}")

        # Tell the model what just happened so follow-ups ("do that again",
        # "put it back") still make sense. turn_complete=False appends to the
        # conversation WITHOUT asking for a reply — the action has already run
        # and been confirmed on screen, so a spoken answer here would only add
        # back the latency this path exists to remove.
        # Not for speech: the model is mid-turn on that same audio, and a
        # client message now would cut across it. It hears the request itself
        # and any call it makes for it is answered "already done".
        if self.session and tell_model:
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text":
                        f"[Executed locally] {intent.tool}({intent.args}) → {result}"}]},
                    turn_complete=False,
                )
            except Exception:
                pass

    def _on_text_command(self, text: str):
        if not self._loop or not self.session:
            return
        # Respect wake-word sleep: a typed command must not be answered while
        # asleep either (the sleep gate is not just for the mic). Wake first with
        # "Hey Jarvis" or the WAKE NOW button.
        if self._wake_enabled and not self._awake:
            self.ui.write_log("SYS: I'm asleep — say 'Hey Jarvis' or tap WAKE NOW first.")
            return
        self._note_user_input(text)
        if self._try_fast_intent(text):
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )
