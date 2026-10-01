"""Local Mode (offline STT, local LLM, offline TTS) and automatic fallback to
it. Mixed into JarvisLive (main.py).
"""

import asyncio
import threading
import time
import json
import sounddevice as sd
import numpy as np
from memory.config_manager import get_input_device, get_plugin_config
from core import (
    gemini as _gemini, audio_devices, context_manager, sentiment_adapter,
    telemetry, fallback,
)
from live.audio_analysis import _pcm_level
from live.constants import CHANNELS, CHUNK_SIZE, SEND_SAMPLE_RATE
from live.prompting import _load_system_prompt


class LocalModeMixin:
    """Local Mode (offline STT, local LLM, offline TTS) and automatic fallback to it."""

    # ── Local Mode: offline STT → local LLM → offline TTS ─────────────────────
    # An additive second pipeline, selected via ⚙ → PLUGIN SETTINGS → ENGINE.
    # It reuses the same tool registry, system prompt assembly, and tool
    # dispatch (_dispatch_tool) as the Gemini Live path above — only the
    # transport (audio streaming vs. record/transcribe/chat/speak) differs.
    # Feature parity is intentionally NOT a goal: vision and Gemini-specific
    # behaviour (session resumption, proactive audio) are unavailable here,
    # and _dispatch_tool already says so rather than pretending to work.

    async def _local_wake_wait(self) -> None:
        """While asleep in Local Mode, actually listen for "Hey Jarvis"
        instead of just polling the _awake flag. Local Mode has no persistent
        mic stream the way the cloud path's _listen_audio does — normally
        each utterance opens its own short-lived InputStream via
        _record_utterance — so without this, a detector armed by
        _enter_standby()/_ui_wake_toggle would sit fed with silence forever
        and "Hey Jarvis" could never fire here. Opens one stream and waits
        (with a timeout so a wedged stream can't hang the loop) rather than
        looping open/close on every poll."""
        det = self._wake_detector
        if det is None or not det.ready:
            await asyncio.sleep(0.5)
            return
        woke = asyncio.Event()
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
            # Same settle window as the cloud path's mic callback (see
            # WAKE_SETTLE_SECONDS) — skip feeding the tail of the utterance
            # that just triggered sleep, so it can't immediately re-wake us.
            if time.monotonic() >= self._wake_feed_gate_open_at:
                det.feed(indata)
            if self._awake:
                loop.call_soon_threadsafe(woke.set)

        _mic_name = get_input_device()
        _mic_dev  = audio_devices.resolve(_mic_name, "input")
        try:
            stream = sd.InputStream(
                samplerate=SEND_SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                blocksize=CHUNK_SIZE, device=_mic_dev, callback=callback,
            )
        except Exception as e:
            self.ui.write_log(f"ERR: Could not open mic to listen for 'Hey Jarvis': {e}")
            await asyncio.sleep(2.0)
            return
        with stream:
            try:
                await asyncio.wait_for(woke.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                pass

    async def _record_utterance(self, max_secs: float = 15.0) -> np.ndarray:
        """Record from the mic until ~700 ms of silence following detected
        speech, or `max_secs` as a hard cap. Returns float32 mono @16kHz,
        normalised to [-1, 1] for faster-whisper / Vosk. Reuses _pcm_level()
        (the same loudness measure the cloud path's HUD waveform uses) as a
        cheap, dependency-free voice-activity gate — no separate VAD model."""
        loop = asyncio.get_event_loop()
        chunks: list[np.ndarray] = []
        state = {"speech_started": False, "silence_run": 0.0}
        done = asyncio.Event()

        def callback(indata, frames, time_info, status):
            level = _pcm_level(indata)
            chunks.append(indata.copy())
            block_secs = frames / SEND_SAMPLE_RATE
            if level > 0.03:
                state["speech_started"] = True
                state["silence_run"] = 0.0
            elif state["speech_started"]:
                state["silence_run"] += block_secs
            if state["speech_started"] and state["silence_run"] > 0.7:
                loop.call_soon_threadsafe(done.set)

        _mic_name = get_input_device()
        _mic_dev  = audio_devices.resolve(_mic_name, "input")
        stream = sd.InputStream(
            samplerate=SEND_SAMPLE_RATE, channels=CHANNELS, dtype="int16",
            blocksize=CHUNK_SIZE, device=_mic_dev, callback=callback,
        )
        with stream:
            try:
                await asyncio.wait_for(done.wait(), timeout=max_secs)
            except asyncio.TimeoutError:
                pass

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio_i16 = np.concatenate(chunks).flatten()
        return audio_i16.astype(np.float32) / 32768.0

    async def _stream_round(self, messages: list, tools: list, tts_queue: "queue.Queue") -> dict:
        """Runs one streaming LLM turn in a background thread: forwards text
        deltas through a SentenceChunker onto tts_queue as sentences complete,
        and collects tool calls to return once the stream ends. Returns the
        same {"content", "tool_calls"} shape llm_client.call_llm() does, so
        the tool-calling loop in _run_local_loop doesn't need to know
        streaming is happening underneath it."""
        from core import llm_client
        from core.sentence_chunker import SentenceChunker

        self._local_stream_cancel.clear()

        def _run() -> dict:
            chunker = SentenceChunker()
            full_content = ""
            tool_calls: list = []
            usage: dict = {}
            for event in llm_client.stream_llm(messages, tools, cancel_event=self._local_stream_cancel):
                if "delta" in event:
                    full_content += event["delta"]
                    for sentence in chunker.feed(event["delta"]):
                        tts_queue.put(sentence)
                elif "tool_call" in event:
                    tool_calls.append(event["tool_call"])
                elif "done" in event:
                    usage = event["done"] or {}
                    break
            if not self._local_stream_cancel.is_set():
                remainder = chunker.flush()
                if remainder:
                    tts_queue.put(remainder)
            return {"content": full_content.strip(), "tool_calls": tool_calls, "usage": usage}

        return await asyncio.to_thread(_run)

    def _tts_consumer(self, tts_queue: "queue.Queue", tts_player, speech_profile, turn=None) -> None:
        """Drains sentences _stream_round pushes and speaks them one at a
        time, on its own thread for the whole turn — so speech overlaps with
        the model still generating (and any tool dispatch in between rounds)
        instead of waiting for the full reply. interrupt() stops this
        mid-sentence by draining tts_queue, pushing the sentinel, and calling
        tts_player.stop(). `turn` (core.telemetry.Turn), if given, is marked
        "first_audio" right before the first sentence is actually spoken."""
        first = True
        while True:
            item = tts_queue.get()
            if item is None:
                break
            try:
                if first and turn is not None:
                    turn.mark_once("first_audio")
                first = False
                tts_player.speak(item, profile=speech_profile)
            except Exception as e:
                print(f"[Local] TTS error: {e}")

    # ── Automatic Local Mode (core/fallback.py) ───────────────────────────────

    async def _maybe_fall_back(self, models_resting: bool) -> bool:
        """Run Local Mode until Gemini is usable again. False (and the cloud
        loop simply keeps retrying) when it is switched off or not set up."""
        from core import llm_client
        cfg = get_plugin_config("local_engine")
        if not cfg.get("auto_fallback", True):
            return False
        ready, missing = await asyncio.to_thread(
            fallback.local_readiness, cfg, llm_client.ensure_ollama_running)
        if not ready:
            now = time.monotonic()
            if now - self._fallback_warned_at > 600:
                self._fallback_warned_at = now
                why = "out of quota" if models_resting else "not reachable"
                self.ui.write_log(
                    f"SYS: Gemini is {why} and Local Mode can't take over yet — missing: "
                    + "; ".join(missing) + ".")
            return False

        self.session = None
        self._in_fallback = True
        self._fallback_streak += 1
        why = "out of quota" if models_resting else "unreachable"
        self.ui.write_log(f"SYS: Gemini is {why} — switching to Local Mode until it's back.")
        local = asyncio.ensure_future(self._run_local_loop(fallback_mode=True))
        probe = asyncio.ensure_future(self._wait_for_cloud())
        try:
            done, _ = await asyncio.wait({local, probe}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (local, probe):
                if not t.done():
                    t.cancel()
            player, self._local_tts_player = self._local_tts_player, None
            if player is not None:
                player.stop()
            await asyncio.gather(local, probe, return_exceptions=True)
            self._in_fallback = False
        if probe in done:
            self._cloud_health.connected()
            self.ui.write_log("SYS: Gemini is reachable again — switching back to the cloud.")
            return True
        return False     # Local Mode could not start; keep retrying the cloud

    async def _wait_for_cloud(self) -> None:
        """Return once Gemini looks usable. Each consecutive fallback waits
        longer before the first check, so a long outage doesn't flap."""
        wait = fallback.PROBE_SECONDS * (2 ** min(self._fallback_streak - 1, 4))
        await asyncio.sleep(wait)
        while True:
            if (not _gemini.all_live_models_resting()
                    and await asyncio.to_thread(fallback.cloud_reachable)):
                return
            await asyncio.sleep(fallback.PROBE_SECONDS)

    async def _run_local_loop(self, fallback_mode: bool = False) -> None:
        import queue
        from core import llm_client
        from core.tool_schema import gemini_tools_to_openai
        from core import tts as tts_mod

        cfg      = get_plugin_config("local_engine")
        provider = llm_client.get_llm_provider()

        if not fallback_mode:
            self.ui.write_log(f"SYS: Starting Local Mode ({provider}) — no cloud calls will be made.")
            self.ui.set_state("THINKING")

        # Fail loudly and stop — never fall back to the cloud API the user
        # explicitly opted out of by choosing Local Mode.
        reachable = await asyncio.to_thread(llm_client.ensure_ollama_running)
        if not reachable:
            url, model = llm_client.get_llm_settings()
            self.ui.write_log(
                f"ERR: Local LLM backend unreachable at {url}. "
                f"Start it (or check the URL in Settings), or switch back to "
                f"Cloud mode in ⚙ → PLUGIN SETTINGS → ENGINE."
            )
            self.ui.set_state("SLEEPING")
            return

        try:
            static_prompt = _load_system_prompt()
            await asyncio.to_thread(llm_client.warmup_model, static_prompt)
        except Exception as e:
            print(f"[Local] Warmup skipped: {e}")

        stt_engine_name = str(cfg.get("local_stt_engine", "whisper")).lower()
        try:
            if stt_engine_name == "vosk":
                from core.stt import VoskSTT
                stt = await asyncio.to_thread(
                    VoskSTT, None, cfg.get("local_stt_language", "en-us"))
            else:
                from core.stt import WhisperSTT
                stt = await asyncio.to_thread(
                    WhisperSTT, cfg.get("local_stt_model", "base"), cfg.get("local_stt_language", "en"))
        except Exception as e:
            self.ui.write_log(f"ERR: Local speech-to-text failed to load: {e}")
            self.ui.set_state("SLEEPING")
            return

        try:
            tts_player = await asyncio.to_thread(tts_mod.create_tts_player, cfg)
        except Exception as e:
            self.ui.write_log(f"ERR: Local text-to-speech failed to load: {e}")
            self.ui.set_state("SLEEPING")
            return
        self._local_tts_player = tts_player

        if fallback_mode:
            # Taking over mid-conversation: stay as awake as we were.
            if self._wake_enabled and not self._awake:
                self._ensure_wake_detector()
                self.ui.set_state("SLEEPING")
            self.ui.write_log("SYS: JARVIS online (Local Mode, temporary).")
        elif self._wake_enabled:
            self._ensure_wake_detector()
            self._awake = False
            self.ui.set_state("SLEEPING")
            self.ui.write_log("SYS: JARVIS online (Local) — sleeping. Say 'Hey Jarvis' to wake me.")
        else:
            self._awake = True
            self.ui.write_log("SYS: JARVIS online (Local Mode).")

        messages: list[dict] = [
            {"role": "system", "content": self._assemble_system_prompt()},
            {"role": "system", "content": ""},  # reserved: refreshed with the ContextBundle each turn, not appended
        ]

        while True:
            if self._wake_enabled and not self._awake:
                await self._local_wake_wait()
                continue

            if not self.ui.muted:
                self._show_idle()
            try:
                audio = await self._record_utterance()
            except Exception as e:
                print(f"[Local] Mic error: {e}")
                await asyncio.sleep(1.0)
                continue

            if audio.size < int(SEND_SAMPLE_RATE * 0.3):   # too short — no real speech
                continue

            self.ui.set_state("THINKING")
            try:
                if stt_engine_name == "vosk":
                    text, _ = await asyncio.to_thread(
                        stt.process_chunk, (audio * 32768.0).astype(np.int16).tobytes())
                else:
                    text = await asyncio.to_thread(stt.transcribe, audio)
            except Exception as e:
                self.ui.write_log(f"ERR: Transcription failed: {e}")
                continue

            text = (text or "").strip()
            if not text:
                continue

            # Result contract: a new user turn closes any pending check.
            if self._false_success.pending:
                self._conclude_false_success(force=True)
            self.ui.write_log(f"You: {text}")
            self._session_log.append(f"User: {text}")
            self._log_context_turn("user", text)
            self._note_user_input(text)

            # Refresh the reserved context slot (messages[1]) right before
            # this turn is sent — session recency, project facts, and
            # semantically-relevant past turns, structured so the model can
            # tell them apart from the live conversation instead of getting
            # one undifferentiated dump. Best-effort: a context-build failure
            # falls back to the empty placeholder rather than blocking the turn.
            try:
                bundle = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: context_manager.build_context(text, session_log=self._session_log),
                )
                combined = bundle.to_prompt()
            except Exception as e:
                print(f"[ContextManager] ⚠️ build_context failed: {e}")
                combined = ""

            # Tone/verbosity/proactivity modifier for the prompt, plus a
            # SpeechProfile (rate/pitch/stability) for the TTS engine once the
            # reply comes back — one detect+log call covers both, via
            # evaluate() (see core/sentiment_adapter.py). recent_texts lets
            # detection notice "I already told you" style repeats. Returns
            # ("", None) outright when the user has disabled adaptation in
            # Settings, so a disabled toggle really means nothing touches the
            # prompt or the voice, not "always neutral".
            speech_profile = None
            try:
                style_text, speech_profile = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: sentiment_adapter.evaluate(text, recent_texts=self._session_log),
                )
                if style_text:
                    combined = f"{combined}\n\n{style_text}" if combined else style_text
            except Exception as e:
                print(f"[SentimentAdapter] ⚠️ evaluate failed: {e}")

            messages[1]["content"] = combined

            messages.append({"role": "user", "content": text})

            turn = telemetry.start_turn("local")

            tts_queue: "queue.Queue" = queue.Queue()
            self._local_tts_queue = tts_queue
            speaker = threading.Thread(
                target=self._tts_consumer, args=(tts_queue, tts_player, speech_profile, turn), daemon=True,
            )
            speaker.start()
            self.set_speaking(True)

            try:
                resp = await self._stream_round(messages, gemini_tools_to_openai(self._all_tool_declarations()), tts_queue)
                usage = resp.get("usage") or {}
                turn.tokens(tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"))
            except Exception as e:
                self.ui.write_log(f"ERR: Local LLM call failed: {e}")
                tts_queue.put(None)
                await asyncio.to_thread(speaker.join)
                self.set_speaking(False)
                self._local_tts_queue = None
                turn.finish()
                continue

            # Tool-calling loop — capped so a model stuck calling tools can
            # never spin forever.
            standby_entered = False
            for _ in range(5):
                tool_calls = resp.get("tool_calls") or []
                if not tool_calls:
                    break
                messages.append({
                    "role": "assistant",
                    "content": resp.get("content", ""),
                    "tool_calls": tool_calls,
                })
                for tc in tool_calls:
                    fn   = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}

                    if name == "shutdown_jarvis":
                        # Mute, don't exit — see _enter_standby(). No LLM call
                        # and no TTS after this: skip straight back to the top
                        # of the outer loop, where the _awake gate takes over.
                        self._enter_standby(reason=self._sleep_reason())
                        messages.append({
                            "role": "tool", "tool_call_id": tc.get("id", ""),
                            "name": name, "content": "muted until 'Hey Jarvis'",
                        })
                        standby_entered = True
                        break

                    print(f"[JARVIS] 🔧 {name}  {args}")
                    self.ui.set_state("THINKING")
                    with turn.tool_span(name):
                        tool_result = await self._dispatch_tool(name, args)
                    outcome = self._apply_result_contract(name, tool_result)
                    print(f"[JARVIS] 📤 {name} → {'ok' if outcome.ok else 'FAILED'}: {outcome.summary[:80]}")
                    messages.append({
                        "role": "tool", "tool_call_id": tc.get("id", ""),
                        "name": name, "content": json.dumps(outcome.as_response(), ensure_ascii=False),
                    })
                if standby_entered:
                    break
                try:
                    resp = await self._stream_round(messages, gemini_tools_to_openai(self._all_tool_declarations()), tts_queue)
                    usage = resp.get("usage") or {}
                    turn.tokens(tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"))
                except Exception as e:
                    self.ui.write_log(f"ERR: Local LLM call failed: {e}")
                    resp = {"content": "", "tool_calls": []}
                    break

            turn.mark("model_done")
            if self._local_stream_cancel.is_set():
                turn.set_interrupted()
            tts_queue.put(None)
            await asyncio.to_thread(speaker.join)
            self.set_speaking(False)
            self._local_tts_queue = None
            turn.finish()

            if standby_entered:
                continue

            reply = (resp.get("content") or "").strip()
            if reply:
                self._note_spoken_for_contract(reply)
                self.ui.write_log(f"{self._asst_name}: {reply}")
                self._session_log.append(f"{self._asst_name}: {reply}")
                self._log_context_turn("assistant", reply)
                messages.append({"role": "assistant", "content": reply})

            if not self.ui.muted:
                self._show_idle()
