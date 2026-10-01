"""Phone dashboard: remote keys, confirm/undo/suggestion parity, phone audio
and commands. Mixed into JarvisLive (main.py).
"""

import asyncio
import time
from core import undo as undo_stack, confirm as confirm_gate, predictive_assistant


class DashboardBridgeMixin:
    """Phone dashboard: remote keys, confirm/undo/suggestion parity, phone audio and commands."""

    def _log_dashboard_unavailable(self):
        """The dashboard can be absent for two very different reasons, and
        telling the user to pip-install packages they already have sends them
        down the wrong path. If it died while starting, show what actually
        killed it."""
        why = getattr(self, "_dashboard_error", "")
        self.ui.write_log(
            f"SYS: Dashboard unavailable — {why}" if why else
            "SYS: Dashboard unavailable. "
            "Run: pip install fastapi \"uvicorn[standard]\" cryptography"
        )

    def _make_remote_key(self):
        """Called from Qt main thread when user presses Remote Control."""
        if self._dashboard is None:
            self._log_dashboard_unavailable()
            return None
        key    = self._dashboard.new_key()
        url    = self._dashboard.get_url()
        manual = self._dashboard.get_manual_url()
        return url, key, f"{url}/auto-login?key={key}", manual

    def _make_remote_key_tailscale(self):
        """Reads this PC's Tailscale address and pairs the phone dashboard
        against it — same login flow (QR/6-digit key) as the LAN Remote
        Control, just reachable from anywhere with internet via the private
        tailnet instead of only this Wi-Fi. Unlike the Cloudflare Tunnel
        this replaced, there's no process to start or wait on here: once
        Tailscale reports an address it's already routable, so this only
        ever shells out to read status. Still called off the Qt thread (see
        ui.py's threaded button handler) since that's still a subprocess call."""
        if self._dashboard is None:
            self._log_dashboard_unavailable()
            return None
        from dashboard.server import PORT
        from dashboard import tailscale as ts

        state = ts.login_state()
        if state == "missing":
            self.ui.write_log(
                "SYS: Tailscale isn't installed. Install it once with:\n"
                f"    {ts.install_hint()}\n"
                "then sign in on this PC (tailscale up) and install/sign in "
                "on your phone too, then click this button again."
            )
            return None
        if state == "needs_login":
            self.ui.write_log(
                "SYS: Tailscale is installed but not signed in. Run 'tailscale up' "
                "and follow the login link, then click this button again."
            )
            return None
        if state == "unknown":
            self.ui.write_log("SYS: Could not read Tailscale status — is the service running?")
            return None

        ip, dns = ts.get_address()
        if not ip:
            self.ui.write_log("SYS: Tailscale has no address for this device yet — try again shortly.")
            return None

        host = dns or ip
        base = f"http://{host}:{PORT}"
        key  = self._dashboard.new_key()
        return base, key, f"{base}/auto-login?key={key}", base, True

    # ── Phone-dashboard parity: confirm / undo / suggestions ──────────────────
    # Gives the phone the same three safety controls the HUD has, over the
    # dashboard's existing /ws channel plus three POST endpoints — all routed
    # through the exact functions the HUD's own buttons call
    # (core.confirm.resolve, core.undo.undo_last, _on_suggestion_decision), so
    # behaviour is identical no matter which surface acted, and a confirmation
    # or suggestion resolved on one surface is invalid on the other because
    # core.confirm/core.undo's pending state is a single shared slot/stack.

    async def _broadcast_remote_state(self) -> None:
        if not self._dashboard:
            return
        try:
            await self._dashboard.broadcast({
                "type":       "state",
                "confirm":    confirm_gate.pending_info(),
                "undo":       undo_stack.history(),
                "suggestion": self._active_suggestion,
            })
        except Exception as e:
            print(f"[Dashboard] state broadcast failed: {e}")

    def _broadcast_remote_state_threadsafe(self) -> None:
        """Safe to call from any thread (a Qt button handler, an executor
        thread running a tool) — hops onto the asyncio loop to actually send."""
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._broadcast_remote_state(), self._loop)

    def _dashboard_confirm(self, confirm_id: str, accepted: bool) -> str | bool:
        """Called off the dashboard's own event loop thread (see
        dashboard/server.py's /api/confirm — it runs this in an executor).
        Rejects a stale/already-resolved id instead of blindly resolving,
        since core.confirm's pending slot is single-use across both surfaces."""
        current = confirm_gate.pending_info()
        if current is None or current["key"] != confirm_id:
            return False
        confirm_gate.resolve(accepted)
        return True

    def _dashboard_undo(self) -> str:
        result = undo_stack.undo_last()
        self._broadcast_remote_state_threadsafe()
        return result

    def _dashboard_suggestion(self, accepted: bool) -> None:
        if self._active_suggestion is None:
            return
        self._on_suggestion_decision(accepted, self._active_suggestion)

    def _maybe_show_suggestion(self) -> None:
        """Throttled check for a proactive hint worth surfacing. Runs the
        (cheap, frequency-based) pattern detectors off the Qt/asyncio thread
        and, if something above threshold turns up and isn't the same pattern
        already on screen, raises it as a dismissible SuggestionHint — never
        auto-executed, see _on_suggestion_decision below."""
        now = time.monotonic()
        if now < self._suggestion_cooldown_until:
            return
        # However this resolves, don't check again for a while — a hit keeps
        # the hint from being spammed, a miss avoids re-running the detectors
        # on every single tool call.
        self._suggestion_cooldown_until = now + 120.0

        def _check():
            try:
                return predictive_assistant.get_proactive_suggestions(
                    current_context=self.ui.current_file or ""
                )
            except Exception as e:
                print(f"[PredictiveAssistant] ⚠️ suggestion check failed: {e}")
                return []

        async def _run():
            suggestions = await asyncio.get_event_loop().run_in_executor(None, _check)
            if not suggestions:
                return
            top = suggestions[0]
            if top.pattern_key == self._last_suggested_pattern:
                return  # already shown (and presumably dismissed/ignored) recently
            self._last_suggested_pattern = top.pattern_key
            suggestion = {
                "action": top.action,
                "confidence_score": top.confidence_score,
                "reasoning": top.reasoning,
                "one_click_command": top.one_click_command,
                "pattern_key": top.pattern_key,
            }
            self.ui.show_suggestion(suggestion)
            self._active_suggestion = suggestion
            await self._broadcast_remote_state()

        asyncio.ensure_future(_run())

    def _spawn(self, coro) -> None:
        """Schedule a coroutine on JARVIS's asyncio loop from ANY thread.
        asyncio.ensure_future() from the Qt thread (the suggestion buttons)
        made a task on a loop that never runs — "There is no current event
        loop" — so RUN on a suggestion silently did nothing."""
        loop = self._loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and running is loop:
            asyncio.ensure_future(coro)
        elif loop is not None:
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            coro.close()

    def _on_suggestion_decision(self, accepted: bool, suggestion: dict) -> None:
        """UI callback for the SuggestionHint's RUN/DISMISS buttons. RUN is the
        explicit human confirmation the spec requires — nothing here ever
        fires without it. Runs the underlying tool through the exact same
        registry _dispatch_tool already uses, so an accepted suggestion behaves
        identically to the user asking for it out loud."""
        pattern_key = suggestion.get("pattern_key", "")
        action      = suggestion.get("action", "")

        if self._active_suggestion is suggestion or (
            self._active_suggestion and self._active_suggestion.get("pattern_key") == pattern_key
        ):
            self._active_suggestion = None
        self._broadcast_remote_state_threadsafe()

        async def _record():
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: predictive_assistant.record_suggestion_feedback(
                    pattern_key, action, accepted,
                    context=self.ui.current_file or "",
                ),
            )
        self._spawn(_record())

        if not accepted:
            return

        if self._action_registry.has(action):
            self._spawn(self._dispatch_tool(action, {}))
        elif pattern_key.startswith("manual:"):
            # A repeated-manual-steps suggestion names a sequence of steps
            # ("step1 then step2"), not a single registered tool. Reconstruct
            # the real {tool, args} calls behind it from the workflow log and
            # save them as a manage_sequence macro, so accepting the hint
            # gives the user a real one-click replay instead of dead-ending.
            async def _save_as_macro():
                steps = await asyncio.get_event_loop().run_in_executor(
                    None, predictive_assistant.get_manual_sequence_steps, pattern_key,
                )
                if not steps:
                    self.ui.write_log(
                        f"SYS: Couldn't find the steps behind '{action}' to save as a macro."
                    )
                    return
                name = suggestion.get("one_click_command") or "auto_sequence"
                result = await self._dispatch_tool("manage_sequence", {
                    "action": "save",
                    "name": name,
                    "steps": steps,
                    "description": suggestion.get("reasoning", ""),
                })
                self.ui.write_log(f"SYS: {result}")

            self._spawn(_save_as_macro())
        else:
            # Neither a registered tool nor a reconstructable manual-step
            # sequence — nothing to run yet, so say so instead of silently
            # doing nothing.
            self.ui.write_log(f"SYS: '{action}' isn't wired to a runnable action yet.")

    # ── Phone audio relay ────────────────────────────────────────────────────────

    async def _relay_phone_audio(self) -> None:
        """Forward phone mic PCM chunks from dashboard queue into the Gemini Live session."""
        q = self._dashboard._phone_audio_queue
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No audio for 1 s → phone mic inactive, give PC mic back
                self._phone_active = False
                continue
            self._phone_active = True   # phone is streaming — silence PC mic
            with self._speaking_lock:
                speaking = self._is_speaking
            if not speaking and not self.ui.muted:
                try:
                    self.out_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass

    def _on_phone_connected(self) -> None:
        self.ui.write_log("SYS: Phone connected via Remote Dashboard.")
        self.ui.notify_phone_connected()

    # ── dashboard command relay ─────────────────────────────────────────────

    async def _process_dashboard_commands(self) -> None:
        while True:
            try:
                text = await asyncio.wait_for(
                    self._dashboard._command_queue.get(), timeout=0.5
                )
                if not text:
                    continue
                # Wait up to 8s for session to become ready after a wake
                for _ in range(80):
                    if self.session:
                        break
                    await asyncio.sleep(0.1)
                if self.session:
                    # A remote command is deliberate control and the phone user
                    # has no desktop WAKE button — so it wakes JARVIS if asleep.
                    if self._wake_enabled and not self._awake:
                        self.wake(reason="remote command")
                    self.ui.write_log(f"[Web]: {text}")
                    if self._try_fast_intent(text):
                        continue
                    await self.session.send_client_content(
                        turns={"role": "user", "parts": [{"text": text}]},
                        turn_complete=True,
                    )
                else:
                    print(f"[Dashboard] Dropped command (no session): {text}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"[Dashboard] Command error: {e}")
                await asyncio.sleep(0.5)
