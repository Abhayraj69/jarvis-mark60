"""Background loops: session summaries, system/topic monitors, proactive mode,
awareness. Mixed into JarvisLive (main.py).
"""

import asyncio
import time
from memory.memory_manager import (
    load_memory, save_session_summary, peek_recent_sessions, get_session_count,
)
from actions.background_monitor import list_monitors, check_all as monitor_check_all
from memory.config_manager import get_plugin_config
from core import awareness


class BackgroundMixin:
    """Background loops: session summaries, system/topic monitors, proactive mode, awareness."""

    # ── Session memory ──────────────────────────────────────────────────────────

    async def _save_session_summary(self) -> None:
        """Summarise the current session in 1-2 sentences and save to long_term.json."""
        log = self._session_log
        if len(log) < 3:          # need at least one exchange to be worth saving
            return
        if self._mode == "local":
            # This calls Gemini's API to write the summary — sending the local,
            # offline conversation to the cloud is exactly the privacy
            # regression Local Mode exists to avoid, so it's skipped rather
            # than silently done anyway. The morning-briefing callback this
            # feeds just won't have anything to reference after a local
            # session; the conversation itself isn't lost, only its summary.
            self._session_log = []
            self.ui.write_log("SYS: Session summary skipped — Local Mode makes no cloud calls.")
            return
        self._session_log = []    # reset immediately so the next session starts clean

        lang = "English"

        convo = "\n".join(log[-40:])   # cap at last 40 turns to stay within token budget
        prompt = (
            f"Summarize this conversation in 1-2 sentences in {lang}. "
            "Focus on what the user accomplished or discussed. "
            "Output ONLY the summary text, nothing else:\n\n" + convo
        )
        try:
            from core import gemini
            summary = await asyncio.to_thread(
                gemini.text, prompt, gemini.SMART, None, 30_000,
            )
            if summary:
                save_session_summary(summary, lang)
        except Exception as e:
            print(f"[Memory] ⚠️ Session summary failed: {e}")

    # ── System monitor ──────────────────────────────────────────────────────────

    async def _run_system_monitor(self) -> None:
        """Background task: voice alerts when metrics exceed thresholds."""
        while True:
            await asyncio.sleep(10)
            alert = await asyncio.to_thread(self._sys_monitor.check)
            if not alert or not self.session or not self._awake:
                continue
            # Don't interrupt an active conversation
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking or (time.monotonic() - self._last_user_speech) < 10:
                continue
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": alert}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[Monitor] ⚠️ Could not send alert: {e}")

    # ── Background monitor ──────────────────────────────────────────────────────

    async def _run_background_monitor(self) -> None:
        """Check user-configured topics once per day; speak alerts when new headlines appear."""
        await asyncio.sleep(300)          # wait 5 min after startup before first check
        while True:
            if self.session and self._awake:
                # Don't interrupt if user spoke recently or JARVIS is mid-sentence
                with self._speaking_lock:
                    speaking = self._is_speaking
                recent_speech = (time.monotonic() - self._last_user_speech) < 30
                if not speaking and not recent_speech:
                    try:
                        alerts = await asyncio.to_thread(monitor_check_all)
                        for alert in alerts:
                            msg = (
                                f"{alert}\n\n"
                                f"Inform the user about this development naturally in English. "
                                "One brief sentence only."
                            )
                            await self.session.send_client_content(
                                turns={"role": "user", "parts": [{"text": msg}]},
                                turn_complete=True,
                            )
                            print("[JARVIS] Monitor alert sent.")
                            await asyncio.sleep(6)   # gap between consecutive alerts
                    except Exception as e:
                        print(f"[Monitor] ⚠️ Background check error: {e}")
            await asyncio.sleep(1800)     # check every 30 minutes

    # ── Proactive mode ──────────────────────────────────────────────────────────

    async def _run_proactive_mode(self) -> None:
        """
        Background task: periodically checks if the user has been silent long enough,
        then hands time + memory context to Gemini so it can decide what (if anything)
        to say proactively. No hardcoded rules — Gemini makes the call.
        """
        while True:
            await asyncio.sleep(60)   # evaluate once per minute

            if not self.session or not self._awake:
                continue

            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue

            if not self._proactive.should_trigger(self._last_user_speech):
                continue

            self._proactive.mark_triggered()

            try:
                memory        = await asyncio.to_thread(load_memory)
                monitors      = await asyncio.to_thread(list_monitors)
                recent_turns  = self._session_log[-8:] if self._session_log else []
                past_sessions = await asyncio.to_thread(peek_recent_sessions, 2)
                depth         = await asyncio.to_thread(get_session_count)
                prompt = self._proactive.build_prompt(
                    memory             = memory,
                    monitors           = monitors or None,
                    recent_turns       = recent_turns or None,
                    past_sessions      = past_sessions or None,
                    relationship_depth = depth,
                )
                # Added after build_prompt, which caches for hours — this changes by the minute.
                activity = self._awareness.summary() \
                    if get_plugin_config("awareness").get("enabled", True) else ""
                if activity:
                    prompt += (f"\n[ACTIVITY] On screen right now: {activity}. "
                               "Use it if it makes the check-in more relevant.")
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": prompt}]},
                    turn_complete=True,
                )
                print("[JARVIS] Proactive check-in.")
            except Exception as e:
                print(f"[Proactive] ⚠️ {e}")

    # ── Awareness (core/awareness.py) ─────────────────────────────────────────

    def _can_nudge(self) -> bool:
        """A moment when speaking up unprompted is polite."""
        if not self.session or not self._awake or self._tools_running:
            return False
        with self._speaking_lock:
            if self._is_speaking:
                return False
        return time.monotonic() - self._last_user_speech > 20

    async def _run_awareness(self) -> None:
        from core import input_guard
        while True:
            await asyncio.sleep(awareness.SAMPLE_SECONDS)
            cfg = get_plugin_config("awareness")
            if not cfg.get("enabled", True):
                continue
            try:
                app, title = await asyncio.to_thread(input_guard.frontmost)
                idle = await asyncio.to_thread(awareness.idle_seconds)
                self._awareness.sample(app, title, idle)
                if not self._can_nudge():
                    continue
                nudge = self._awareness.break_nudge() if cfg.get("break_reminders", True) else None
                if nudge is None and cfg.get("screen_errors", False) \
                        and self._awareness.wants_error_check():
                    error = await asyncio.to_thread(self._screen_error)
                    nudge = self._awareness.error_seen(error)
                if nudge is None or not self._can_nudge():
                    continue
                self._awareness.mark_nudged(nudge)
                print(f"[Awareness] {nudge.kind}: {nudge.text[:80]}")
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": f"[AWARENESS] {nudge.text}"}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[Awareness] ⚠️ {e}")

    @staticmethod
    def _screen_error() -> str:
        """One cheap vision read of the screen: the visible error, or ''."""
        from actions.screen_processor import _capture_screen, _vision_query
        from core.backend_router import TaskKind
        img, mime = _capture_screen()
        reply = _vision_query(img, mime, awareness.ERROR_PROMPT, kind=TaskKind.VISION,
                              policy={TaskKind.VISION: ["gemini_lite", "gemini"]})
        return awareness.parse_error(reply)

    def _awareness_settings_section(self) -> dict:
        return {
            "plugin":    "awareness",
            "namespace": "awareness",
            "title":     "👁 AWARENESS — notice what I'm doing",
            "fields": [
                {"key": "enabled", "type": "toggle",
                 "label": "Track the focused app/window (stays on this computer; "
                          "a short summary goes into check-ins)",
                 "default": True},
                {"key": "break_reminders", "type": "toggle",
                 "label": f"Suggest a break after {awareness.BREAK_AFTER_MINUTES} minutes of non-stop use",
                 "default": True},
                {"key": "briefing_agenda", "type": "toggle",
                 "label": "Include today's calendar and unread email in the morning briefing",
                 "default": True},
                {"key": "screen_errors", "type": "toggle",
                 "label": "Offer help when the same error stays on screen in a terminal/editor "
                          "(sends a screenshot to Gemini every few minutes)",
                 "default": False},
            ],
            "values": get_plugin_config("awareness"),
        }
