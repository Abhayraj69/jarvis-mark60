"""Running tool calls: dispatch, background tools, think, and the result
contract. Mixed into JarvisLive (main.py).
"""

import asyncio
import threading
import time
import traceback
from datetime import datetime
from google.genai import types
from memory.memory_manager import update_memory, search_memory
from actions.screen_processor import _capture_camera, _capture_screen
from actions.system_monitor import get_system_status
from actions.background_monitor import add_monitor, remove_monitor, list_monitors
from memory.config_manager import get_plugin_config
from core import (
    undo as undo_stack, fast_intent, predictive_assistant, causal_reasoning,
    context_manager, sequence_memory, telemetry, result_contract,
    think as think_core,
)
from live.constants import FAST_VOICE_DEDUPE_SECONDS


class ToolsMixin:
    """Running tool calls: dispatch, background tools, think, and the result contract."""

    async def _execute_tool(self, fc, background: bool = False) -> types.FunctionResponse:
        """`background`: running beside the conversation (see
        _start_background_tool) — leave the HUD state to whatever JARVIS is
        doing meanwhile instead of flipping it to THINKING and back."""
        name = fc.name
        args = dict(fc.args or {})

        print(f"[JARVIS] 🔧 {name}  {args}")
        self._touch_activity()
        if not background:
            self.ui.set_state("THINKING")


        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
            if not self.ui.muted:
                self._show_idle()
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True}
            )

        if name == "shutdown_jarvis":
            # `silent: True` (see save_memory above) tells the Live API not to
            # generate a spoken turn for this — a text instruction alone is
            # not a hard enough guarantee for "zero trailing speech".
            local_text = await asyncio.to_thread(self._local_transcript)
            if not self._standby_allowed(local_text):
                return types.FunctionResponse(
                    id=fc.id, name=name,
                    response={"result": "ignored: the user did not ask you to sleep. "
                                        "Stay awake and do not call this again unless they do.",
                              "silent": True}
                )
            self._enter_standby(reason=self._sleep_reason())
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "muted until 'Hey Jarvis'", "silent": True}
            )

        if name == "think":
            # Runs in the background and sends its own FunctionResponse when
            # the answer is ready (the tool is declared NON_BLOCKING). Returning
            # None tells the receive loop not to answer this call itself.
            self._start_think(fc, args)
            return None

        done = self._fast_voice_done
        if done is not None and time.monotonic() - done[1] < FAST_VOICE_DEDUPE_SECONDS \
                and fast_intent.same_call(done[0], name, args):
            # Already run from the transcript a moment ago — running it again
            # would undo a toggle (pause) or double a step (volume).
            self._fast_voice_done = None
            print(f"[JARVIS] ⚡ {name} already done from speech — not repeating")
            if not self.ui.muted:
                self._show_idle()
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"ok": True, "summary": f"Already done: {done[0].reply}",
                          "silent": True})

        result = await self._dispatch_tool(name, args)
        outcome = self._apply_result_contract(name, result)

        if not self.ui.muted and not background:
            self._show_idle()

        print(f"[JARVIS] 📤 {name} → {'ok' if outcome.ok else 'FAILED'}: {outcome.summary[:80]}")
        # A tool that declared itself NON_BLOCKING also says when its answer may
        # re-enter the conversation (e.g. don't talk over a call that is already
        # ringing). Tools that declared nothing get the API default.
        _sched = (self._action_registry.scheduling(name)
                  or self._plugin_registry.scheduling(name))
        _extra = {"scheduling": _sched} if _sched else {}
        return types.FunctionResponse(
            id=fc.id, name=name,
            response=outcome.as_response(),
            **_extra
        )

    # ── Result contract (core/result_contract.py) ─────────────────────────────
    def _apply_result_contract(self, name: str, result) -> result_contract.ToolOutcome:
        """Wrap a raw handler result as {ok, summary, detail} and, when ok is
        false, start watching what JARVIS says next (see _receive_audio's
        turn_complete handling and _run_local_loop) so a reply that reports
        success after a failed tool is counted as a false success."""
        outcome = result_contract.classify(name, result)
        if not outcome.ok:
            self._false_success.register_failure(name)
            self.ui.write_log(f"SYS: ✗ {name} — {outcome.summary[:100]}")
        return outcome

    def _note_spoken_for_contract(self, spoken: str) -> None:
        self._false_success.note_output(spoken)

    def _conclude_false_success(self, force: bool = False) -> None:
        """Decide a pending false-success check: on a new user turn (force)
        or once its deadline has passed. Records to telemetry and the log."""
        verdict = self._false_success.conclude() if force else self._false_success.poll()
        if verdict is None:
            return
        if verdict.false_success:
            telemetry.record_false_success(verdict.tools, verdict.spoken)
            self.ui.write_log(
                f"SYS: ⚠ reported success after failed {', '.join(verdict.tools)}: "
                f"\"{result_contract.snippet(verdict.spoken, 120)}\""
            )
            print(f"[Contract] false success after {verdict.tools}: {result_contract.snippet(verdict.spoken)}")

    # ── think (core/think.py) ─────────────────────────────────────────────────
    # Delivery: the first sentence or two go back as the FunctionResponse the
    # moment they exist (scheduling=WHEN_IDLE, so they follow the model's own
    # acknowledgement instead of cutting it off); anything after that is sent
    # as one follow-up text turn once JARVIS has finished speaking part one.
    # Short answers — the common case — arrive whole in the FunctionResponse.
    _THINK_FIRST_PART_CHARS = 140
    _THINK_FOLLOWUP_WAIT_S  = 25.0

    # ── Background tools ──────────────────────────────────────────────────────
    # A tool that declared behavior NON_BLOCKING (the slow ones: web search,
    # flights, code, file processing, game updates, study notes/quizzes) used
    # to be awaited inside the receive loop like any other, so nothing the
    # user said was processed until it finished — the declaration told the
    # model not to wait, but the client waited anyway. Now it runs as its own
    # task and answers when done, timed for a gap in the conversation.

    def _runs_in_background(self, name: str) -> bool:
        return (self._action_registry.runs_in_background(name)
                or self._plugin_registry.runs_in_background(name))

    def _start_background_tool(self, fc) -> None:
        self._tools_running += 1          # the sleep watch waits for it
        self.ui.write_log(f"SYS: ⏳ {fc.name} is running in the background — keep talking.")
        task = asyncio.get_event_loop().create_task(
            self._run_background_tool(fc, self.session))
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _run_background_tool(self, fc, session) -> None:
        t0 = time.monotonic()
        try:
            fr = await self._execute_tool(fc, background=True)
        except Exception as e:
            print(f"[JARVIS] ❌ background {fc.name}: {e}")
            fr = types.FunctionResponse(id=fc.id, name=fc.name, response={
                "ok": False, "summary": f"{fc.name} failed", "detail": str(e)[:300]})
        finally:
            self._tools_running -= 1
            self._touch_activity()
        secs = time.monotonic() - t0
        ok = bool((fr.response or {}).get("ok", True))
        summary = str((fr.response or {}).get("summary", ""))[:120]
        self.ui.write_log(f"SYS: {'✓' if ok else '✗'} {fc.name} finished ({secs:.1f}s).")

        if not self._awake:
            # Put to sleep meanwhile ("bye jarvis"): keep the result, don't talk.
            fr.scheduling = types.FunctionResponseScheduling.SILENT
            self.ui.write_log(f"SYS: {fc.name}: {summary}")
        elif fr.scheduling is None:
            fr.scheduling = types.FunctionResponseScheduling.WHEN_IDLE

        current = self.session
        if current is None:
            self.ui.write_log(f"SYS: {fc.name}: {summary}")
            return
        try:
            if current is session:
                await current.send_tool_response(function_responses=[fr])
                return
            # The session was rebuilt while this ran; the call it answers is
            # gone, so hand the result over as a message instead.
            if not self._awake:
                return
            await self._wait_until_quiet(timeout=20.0)
            await current.send_client_content(
                turns={"role": "user", "parts": [{"text":
                    f"[BACKGROUND RESULT] {fc.name} finished "
                    f"({'ok' if ok else 'failed'}): {fr.response}. "
                    "Tell the user the outcome briefly."}]},
                turn_complete=True,
            )
        except Exception as e:
            print(f"[JARVIS] could not deliver background {fc.name}: {e}")
            self.ui.write_log(f"SYS: {fc.name}: {summary}")

    def _start_think(self, fc, args: dict) -> None:
        task = asyncio.get_event_loop().create_task(self._run_think(fc, args))
        self._think_tasks.add(task)
        task.add_done_callback(self._think_tasks.discard)

    async def _run_think(self, fc, args: dict) -> None:
        loop  = asyncio.get_event_loop()
        query = str(args.get("query") or "").strip()
        include_screen = bool(args.get("include_screen", False))
        turn  = self._current_turn
        t0    = time.monotonic()

        if not query:
            await self._send_think_response(fc, {"ok": False, "summary": "think needs a query",
                                                 "detail": "No query was given."})
            return

        self.ui.write_log(f"[think] {query[:80]}{' +screen' if include_screen else ''}")

        first_closed = threading.Event()   # decided synchronously in the worker thread
        first_sent   = asyncio.Event()     # loop-side signal that part one can go
        first_part:  list[str] = []
        rest_part:   list[str] = []

        def _on_sentence(sentence: str, idx: int):
            # Worker thread. The part-one/part-two split is decided HERE with
            # a threading.Event, not by peeking at the asyncio.Event: a
            # call_soon_threadsafe(set) hasn't run yet when the next sentence
            # arrives on a fast stream, and every sentence would land in
            # part one.
            if not first_closed.is_set():
                first_part.append(sentence)
                if sum(len(x) for x in first_part) >= self._THINK_FIRST_PART_CHARS or idx >= 1:
                    first_closed.set()
                    loop.call_soon_threadsafe(first_sent.set)
            else:
                rest_part.append(sentence)

        try:
            from core.backend_router import TaskKind, load_policy_from_config
            policy = load_policy_from_config(get_plugin_config("routing"))
            # Flash-Lite first for think: telemetry put it at a 3.2 s median
            # against 6.8 s for full Flash, and full Flash's free tier allows
            # only 20 requests a day — once spent, every think paid for a
            # failed Flash call before falling back to Lite anyway.
            chat = [n for n in policy.get(TaskKind.CHAT, []) if n != "gemini_lite"]
            policy[TaskKind.CHAT] = ["gemini_lite"] + chat
        except Exception:
            policy = None

        async def _deliver_first_when_ready():
            await first_sent.wait()
            text = " ".join(first_part).strip()
            await self._send_think_response(fc, {
                "ok": True, "summary": text[:160], "detail": text,
                "relay": "Speak this now in your own voice; more may follow — do not conclude.",
            })

        deliver_task = asyncio.ensure_future(_deliver_first_when_ready())
        try:
            result = await asyncio.to_thread(
                think_core.run, query, list(self._session_log), include_screen,
                _on_sentence, policy,
            )
        except Exception as e:
            deliver_task.cancel()
            msg = str(e)[:200]
            self.ui.write_log(f"ERR: think — {msg}")
            self._false_success.register_failure("think")
            await self._send_think_response(fc, {"ok": False, "summary": "reasoning failed",
                                                 "detail": f"The reasoning core failed: {msg}"})
            return

        elapsed = (time.monotonic() - t0) * 1000
        label = f"think:{result.backend or 'unknown'}"
        if turn is not None:
            turn.add_tool_span(label, elapsed)
        self.ui.write_log(f"[think] {result.backend} · {elapsed:.0f}ms · {len(result.sentences)} sentence(s)")

        if not first_closed.is_set():
            # Whole answer fits in part one (or nothing came back at all).
            deliver_task.cancel()
            text = result.text.strip()
            if text:
                await self._send_think_response(fc, {"ok": True, "summary": text[:160], "detail": text,
                                                     "relay": "Speak this answer in your own voice."})
            else:
                self._false_success.register_failure("think")
                await self._send_think_response(fc, {"ok": False, "summary": "empty answer",
                                                     "detail": "The reasoning core returned nothing."})
            return

        await deliver_task
        rest = " ".join(rest_part).strip()
        if rest and self.session:
            await self._wait_until_quiet(self._THINK_FOLLOWUP_WAIT_S)
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text":
                        "[THINK, continued] Continue relaying this directly, in your own voice, "
                        "with no preamble and without repeating what you already said: " + rest}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[Think] could not deliver continuation: {e}")

    async def _send_think_response(self, fc, response: dict) -> None:
        if not self.session:
            return
        try:
            await self.session.send_tool_response(function_responses=[
                types.FunctionResponse(
                    id=fc.id, name="think", response=response,
                    scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
                )
            ])
        except Exception as e:
            print(f"[Think] could not deliver response: {e}")

    async def _wait_until_quiet(self, timeout: float) -> None:
        """Wait until JARVIS has stopped speaking (or `timeout` passes) so a
        follow-up text turn doesn't interrupt part one mid-sentence."""
        deadline = time.monotonic() + timeout
        # Give the model a moment to start speaking part one before we
        # start polling for silence.
        await asyncio.sleep(1.5)
        while time.monotonic() < deadline:
            with self._speaking_lock:
                speaking = self._is_speaking
            if not speaking:
                return
            await asyncio.sleep(0.2)

    async def _dispatch_tool(self, name: str, args: dict) -> str:
        """The actual tool router: everything except the Gemini-specific
        `save_memory` fast path above (which returns a `silent` FunctionResponse
        flag that only means something to the Live API and isn't worth
        threading through a second caller).

        Split out of _execute_tool so Local Mode's text tool-calling loop
        (_run_local_loop) can dispatch against the exact same action/plugin
        registry and inline tools as the Gemini Live path, instead of
        duplicating this router. Returns the plain string result — the two
        callers wrap it differently (a Gemini FunctionResponse here, a
        {"role": "tool", ...} message in local mode)."""
        loop    = asyncio.get_event_loop()
        result  = "Done."
        success = True

        # Vision needs a live multimodal session to inject the captured image
        # into (see _receive_audio's turn_complete handling) — Local Mode's
        # text-only LLM has nothing to send that image to, so pretending to
        # capture it would leave the model describing an image it never saw.
        if name in ("screen_process", "close_camera") and (self._mode == "local" or self._in_fallback):
            return ("Vision is not available in Local Mode — it requires the "
                    "Cloud (Gemini Live) engine. Switch engines in Settings "
                    "to use the camera or screen.")

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
            return "ok"

        try:
            if name == "recall_memory":
                # Local file search: no network, no second model. Kept out of
                # the executor deliberately — it is a dictionary scan over a few
                # hundred short strings, and a thread hop would cost more than
                # the work itself.
                result = search_memory(args.get("query", ""), limit=8)

            elif name == "undo":
                if str(args.get("action", "")).lower().strip() == "list":
                    items = undo_stack.history()
                    result = ("Things I can undo, most recent first:\n"
                              + "\n".join(f"{i+1}. {t}" for i, t in enumerate(items))
                              ) if items else "I have not changed anything I can undo yet."
                else:
                    result = await loop.run_in_executor(None, undo_stack.undo_last)

            elif name == "screen_process":
                import time as _t_mod
                _now = _t_mod.monotonic()
                _cooldown = 4.0  # seconds — covers echo window after speaking ends
                if self._vision_busy or (_now - self._vision_last_time) < _cooldown:
                    _wait = max(0, _cooldown - (_now - self._vision_last_time))
                    print(f"[Vision] ⏳ Cooldown active ({_wait:.1f}s remaining) — ignoring duplicate call")
                    result = "Vision is still processing the previous request. I will not call this again."
                else:
                    self._vision_busy      = True
                    self._vision_last_time = _now
                    angle     = args.get("angle", "screen").lower()
                    user_text = args.get("text", "What do you see?")
                    if angle == "camera":
                        img_b, mime_t = await loop.run_in_executor(None, _capture_camera)
                        self.ui.start_camera_stream()
                        self._vision_cam_active = True
                        print(f"[Vision] 📷 Camera: {len(img_b):,} bytes")
                        _stall = "camera"
                    else:
                        img_b, mime_t = await loop.run_in_executor(None, _capture_screen)
                        print(f"[Vision] 🖥️  Screen: {len(img_b):,} bytes")
                        _stall = "screen"
                    self._pending_vision = (img_b, mime_t, user_text, angle)
                    # The image is attached to this same exchange, so there is
                    # nothing to stall for and nothing to announce. Asking for an
                    # acknowledgement here is what produced two spoken answers —
                    # the model filled that turn by answering the question from
                    # imagination, then answered it again once it could see.
                    result = (
                        f"[VISION_ACTIVE] {_stall.capitalize()} captured and attached to this "
                        f"same exchange. Do not acknowledge and do not answer yet — the image "
                        f"is arriving with this result. Reply once, from what you actually see "
                        f"in it."
                    )

            elif name == "close_camera":
                self.ui.stop_camera_stream()
                result = "Camera closed."

            elif name == "system_status":
                r = await loop.run_in_executor(None, get_system_status)
                result = str(r)

            elif name == "get_time":
                result = datetime.now().strftime("It is %I:%M %p on %A, %B %d, %Y.")

            elif name == "manage_monitor":
                action = args.get("action", "").lower().strip()
                topic  = args.get("topic", "").strip()
                if action == "add" and topic:
                    result = await asyncio.to_thread(add_monitor, topic)
                elif action == "remove" and topic:
                    result = await asyncio.to_thread(remove_monitor, topic)
                elif action == "list":
                    topics = await asyncio.to_thread(list_monitors)
                    result = ("Monitoring: " + ", ".join(topics)) if topics else "No topics are being monitored."
                else:
                    result = "Specify action (add/remove/list) and a topic."

            elif self._action_registry.has(name):
                # file_processor: fall back to the currently-uploaded file when none is given
                if name == "file_processor" and not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file

                def _dispatch_sync(step_name: str, step_args: dict) -> str:
                    """Blocking (tool_name, args) -> str re-entry into this same
                    router, for actions that need to run other tools as steps
                    (e.g. sequence replay in actions/sequence_recall.py). Runs on
                    the caller's worker thread; hands the actual coroutine to the
                    event loop and blocks only this thread, not the loop."""
                    future = asyncio.run_coroutine_threadsafe(
                        self._dispatch_tool(step_name, dict(step_args or {})), loop
                    )
                    return future.result(timeout=120)

                _ctx = {"player": self.ui, "speak": self.speak,
                        "response": None, "session_memory": None,
                        "dispatch": _dispatch_sync}
                r = await loop.run_in_executor(None, lambda: self._action_registry.run(name, args, _ctx))
                result = r or "Done."
                # web_search: mirror results to the on-screen content panel
                if (name == "web_search" and r
                        and not r.startswith("No results")
                        and not r.startswith("Search failed")):
                    _mode  = args.get("mode", "search")
                    _query = args.get("query") or ", ".join(args.get("items", []))
                    _label = f"{_mode.upper()} — {_query[:38]}" if _query else _mode.upper()
                    self.ui.show_content(_label, r)

            else:
                _connector_target = self._tool_connector_registry.has_declaration(name)
                if self._plugin_registry.has(name):
                    r = await loop.run_in_executor(
                        None,
                        lambda: self._plugin_registry.run(name, args, player=self.ui, session_memory=None)
                    )
                    result = r or "Done."
                elif _connector_target:
                    # READ_ONLY runs and returns its result inline; REVERSIBLE/
                    # DESTRUCTIVE instead comes back as a "[CONFIRMATION_PENDING]"
                    # sentence for the model to relay — the registry itself
                    # decides which, via core.confirm — exactly like
                    # shutdown_jarvis above, just one layer further down.
                    connector_name, connector_action = _connector_target
                    r = await loop.run_in_executor(
                        None,
                        lambda: self._tool_connector_registry.execute(connector_name, connector_action, args),
                    )
                    result = r or "Done."
                else:
                    result = f"Unknown tool: {name}"

        except Exception as e:
            result  = f"Tool '{name}' failed: {e}"
            success = False
            traceback.print_exc()
            self.speak_error(name, e)

        # Feed the Predictive Assistant. Best-effort: a logging hiccup must
        # never surface as a dispatch failure, so it's swallowed rather than
        # propagated. save_memory/recall_memory return early above and are
        # deliberately not logged here — they aren't user-facing "actions" in
        # the sense the pattern detectors care about.
        try:
            await loop.run_in_executor(
                None,
                lambda: predictive_assistant.log_event(
                    action_type=name,
                    context=self.ui.current_file or "",
                    input_data=str(args)[:500],
                    output=str(result)[:500],
                    success=success,
                ),
            )
        except Exception:
            pass

        # Mirror the same dispatch into the causal-reasoning timeline (see
        # core/causal_reasoning.py) so tool calls and screen_monitor alerts
        # share one cause-and-effect graph instead of two disconnected logs.
        # Best-effort for the same reason as the block above: a reasoning
        # side-channel must never be able to fail a real tool dispatch.
        if success:
            try:
                await loop.run_in_executor(
                    None, lambda: causal_reasoning.record_event(f"tool:{name}")
                )
            except Exception:
                pass

        self._maybe_show_suggestion()

        # Record-mode macros (core/sequence_memory.py): a no-op unless the
        # user is actively recording one (start_recording/manage_sequence
        # with action="record_start"). manage_sequence's own steps are
        # excluded inside record_step itself, so a macro can't record itself.
        # "confirm" is set from the sentinel core/confirm.py's request()
        # returns — see manage_sequence's replay path re-gating on it.
        if success:
            sequence_memory.record_step(
                name, args, confirm=isinstance(result, str) and result.startswith("[CONFIRMATION_PENDING]")
            )

        return result

    def _log_context_turn(self, role: str, content: str) -> None:
        """Fire-and-forget persistence into context_manager's durable turn
        store (separate from self._session_log, which only lives for the
        current session/summary) — used from both the Live audio pipeline
        and Local Mode's text loop so semantic recall works across sessions
        no matter which engine produced the turn. Best-effort: a storage
        hiccup must never stall speech or transcription."""
        async def _do():
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: context_manager.log_turn(role, content)
                )
            except Exception as e:
                print(f"[ContextManager] ⚠️ log_turn failed: {e}")
        asyncio.ensure_future(_do())
