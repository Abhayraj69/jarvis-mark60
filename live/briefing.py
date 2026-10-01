"""Reply-audio reports, session connect handling and the startup/morning
briefing. Mixed into JarvisLive (main.py).
"""

import concurrent.futures
import asyncio
from datetime import datetime
from memory.memory_manager import load_memory, pop_last_session, get_session_count
from memory.config_manager import get_plugin_config
from live.constants import RECEIVE_SAMPLE_RATE


class BriefingMixin:
    """Reply-audio reports, session connect handling and the startup/morning briefing."""

    # ── Morning briefing ────────────────────────────────────────────────────────

    def _report_reply_audio(self) -> None:
        """At turn_complete: say plainly if this reply's audio came in slower
        than it plays (network/server) or the speaker ran dry (local). Quiet
        when all is well."""
        stats, self._reply_audio = self._reply_audio, None
        underruns, self._underruns = self._underruns, 0
        if stats is None:
            return
        first, last, nbytes = stats
        speech = nbytes / (RECEIVE_SAMPLE_RATE * 2)
        span = last - first
        if speech >= 1.0 and span > speech * 1.1:
            print(f"[JARVIS] ⚠️ Reply audio arrived slower than real time: {speech:.1f}s of "
                  f"speech took {span:.1f}s to arrive (network or Gemini is slow).")
        if underruns:
            print(f"[JARVIS] ⚠️ Speaker ran out of audio {underruns} time(s) during the reply.")

    def _compose_brief(self, memory: dict, depth: int) -> str:
        """Morning brief (news, patterns…) plus today's calendar and unread
        mail when those plugins are installed and enabled. Runs in a worker."""
        brief = self._proactive.get_morning_brief(memory, depth) or ""
        if not get_plugin_config("awareness").get("briefing_agenda", True):
            return brief
        agenda = []
        for tool, args in (("check_calendar", {"day": "today"}), ("check_email", {"limit": 3})):
            if not self._plugin_registry.has(tool):
                continue
            # Not `with`: its exit would wait for a call stuck on a permission prompt.
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                text = ex.submit(self._plugin_registry.run, tool, args).result(timeout=6)
            except Exception:
                continue          # slow or refused: the brief goes out without it
            finally:
                ex.shutdown(wait=False)
            if text and not text.startswith("Could not"):
                agenda.append(text)
        if agenda:
            brief = (brief + "\n\n" if brief else "") + "Today's agenda: " + " ".join(agenda)
        return brief

    def _on_session_connected(self) -> None:
        """Awake/asleep state for a session that just connected."""
        # Wake word: if enabled, come up ASLEEP (mic gated, silent)
        # until the user says "Hey Jarvis" or taps wake in the UI.
        # Only the FIRST connect of a run comes up asleep. Live
        # sessions drop and reconnect on their own (network blips,
        # server GoAway, a 503, changing mic or voice); forcing
        # sleep on every one of those put JARVIS to sleep
        # mid-conversation with no command from the user.
        reconnect = self._has_connected
        self._has_connected = True
        self._go_away_pending = False
        if reconnect and self._awake:
            self._touch_activity()
            if not self.ui.muted:
                self._show_idle()
            self.ui.write_log("SYS: Reconnected — still awake.")
        elif self._wake_enabled:
            self._ensure_wake_detector()
            self._awake = False
            self.ui.set_state("SLEEPING")
            self.ui.write_log("SYS: JARVIS online — sleeping. Say 'Hey Jarvis' to wake me.")
        else:
            self._awake = True
            self._show_idle()
            self.ui.write_log("SYS: JARVIS online.")

    async def _send_startup_briefing(self) -> None:
        """
        Two-phase briefing optimized for speed:
          Phase 1 — instant greeting (no tools) → speech starts in <1s
          Phase 2 — the consolidated morning brief (build_morning_brief: monitored-
                    topic news, habit suggestions, recent-session context, and
                    causal patterns) is assembled in a background thread while
                    Phase 1 plays, then delivered as ready text (no Gemini tool-call
                    round-trip) and shown on the UI content panel. Waits for
                    turn_complete instead of a fixed sleep so there is no
                    unnecessary gap.
        """
        memory   = load_memory()
        identity = memory.get("identity", {})

        def _val(k: str) -> str:
            e = identity.get(k, {})
            return (e.get("value", "") if isinstance(e, dict) else str(e)).strip()

        name = _val("name")
        time_str = datetime.now().strftime("%H:%M")

        # Start assembling the morning brief immediately — runs in a background
        # thread (news lookups + DB queries) in parallel while phase 1 plays
        loop  = asyncio.get_event_loop()
        depth = await asyncio.to_thread(get_session_count)
        brief_future = loop.run_in_executor(None, self._compose_brief, memory, depth)

        await asyncio.sleep(0.3)
        if not self.session:
            return

        # ── Phase 1: instant greeting ─────────────────────────────────────────
        lang_clause = " Speak in English."
        name_clause = f" Address the user as {name}." if name else ""

        # Inject last session context if available — pop removes it so it's never repeated
        last = await asyncio.to_thread(pop_last_session)
        session_clause = ""
        if last:
            try:
                _delta = (datetime.now() - datetime.strptime(last["date"], "%Y-%m-%d")).days
                _when  = "earlier today" if _delta == 0 else ("yesterday" if _delta == 1 else f"{_delta} days ago")
            except Exception:
                _when = "last time"
            session_clause = (
                f" Also briefly and naturally mention that {_when}: {last['summary']}"
            )

        p1 = (
            f"Greet the user warmly, mention it is {time_str}, and say you are fetching today's news now.{session_clause} "
            f"Keep it to 2 short sentences max. Do not call any tools.{lang_clause}{name_clause}"
        )

        # Clear the turn-done event so we can wait for Phase 1 to finish
        if self._turn_done_event:
            self._turn_done_event.clear()

        await self.session.send_client_content(
            turns={"role": "user", "parts": [{"text": p1}]},
            turn_complete=True,
        )
        print("[JARVIS] Briefing phase 1 (greeting) sent.")

        # ── Phase 2: fire as soon as Phase 1 audio is done ───────────────────
        async def _deliver_brief():
            try:
                lang_str = " Speak in English."

                # Wait for the brief to finish assembling (already running) and
                # Phase 1 turn-complete in parallel — whichever takes longer
                # determines the wait time
                brief_done  = asyncio.wrap_future(brief_future)
                turn_waited = False
                if self._turn_done_event:
                    try:
                        await asyncio.wait_for(self._turn_done_event.wait(), timeout=6.0)
                        turn_waited = True
                    except asyncio.TimeoutError:
                        pass

                # Extra buffer: turn_complete fires when Gemini finishes *generating*
                # Phase 1, but audio may still be playing.  Waiting a beat here
                # prevents Phase 2 audio from arriving while Phase 1 is mid-sentence
                # (which sounds like a "repeated first response" to the user).
                if turn_waited:
                    await asyncio.sleep(0.8)
                else:
                    await asyncio.sleep(1.0)

                try:
                    brief_text = await asyncio.wait_for(brief_done, timeout=8.0)
                except Exception:
                    brief_text = ""

                if not self.session:
                    return

                if brief_text:
                    # Show on UI content panel immediately
                    self.ui.show_content("MORNING BRIEF", brief_text)
                    p2 = f"{brief_text}{lang_str}"
                else:
                    p2 = (
                        "Nothing new to report right now — no news, patterns, or "
                        f"context worth mentioning. Let the user know briefly.{lang_str}"
                    )

                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": p2}]},
                    turn_complete=True,
                )
                self.ui.write_log("SYS: Briefing phase 2 (morning brief) sent.")
            except Exception as e:
                print(f"[Briefing] Phase 2 error: {e}")
                print(f"[JARVIS] Briefing phase 2 failed: {e}")
                self.ui.write_log("SYS: Could not fetch the news for the briefing.")

        asyncio.create_task(_deliver_brief())
