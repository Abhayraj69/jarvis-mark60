"""Mic capture, sending to Gemini, receiving replies and playing them. Mixed
into JarvisLive (main.py).
"""

import asyncio
import contextlib
import time
import traceback
import sounddevice as sd
import numpy as np
from datetime import datetime
from google.genai import types
from memory.config_manager import get_input_device, get_output_device
from core import audio_devices, telemetry
from live.audio_analysis import (
    _CURSOR_SLACK, _FIRST_SOUND, _TAIL_MARGIN, _VIS_HOP, _pcm_level, _pcm_visemes,
)
from live.constants import CHANNELS, CHUNK_SIZE, RECEIVE_SAMPLE_RATE, SEND_SAMPLE_RATE
from live.prompting import _REPEAT_MIN, _clean_transcript, _is_repeat_chunk


class AudioIOMixin:
    """Mic capture, sending to Gemini, receiving replies and playing them."""

    def _enqueue_audio(self, item: dict) -> None:
        """Loop thread: queue one mic chunk for Gemini. When the sender has
        fallen behind and the queue is full, drop the OLDEST chunk (stale
        audio is worth less than what is being said now) and report the
        backlog once, instead of a traceback per chunk."""
        q = self.out_queue
        if q is None:
            return
        try:
            q.put_nowait(item)
            if self._audio_dropped:
                print(f"[JARVIS] ⚠️ Mic backlog cleared — {self._audio_dropped} chunk(s) "
                      f"(~{self._audio_dropped * CHUNK_SIZE / SEND_SAMPLE_RATE:.1f}s) were dropped.")
                self._audio_dropped = 0
        except asyncio.QueueFull:
            if not self._audio_dropped:
                print("[JARVIS] ⚠️ Mic backlog: audio is not reaching Gemini fast enough — "
                      "dropping the oldest.")
            self._audio_dropped += 1
            try:
                q.get_nowait()
                q.put_nowait(item)
            except Exception:
                pass

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            _t0 = time.monotonic()
            # Gemini 3.x Live rejects the old realtime_input.media_chunks field
            # (what `media=...` maps to) and closes the socket with a 1007. Send
            # mic / phone PCM through the new `audio` field instead. Queue items
            # are {"data": <bytes>, "mime_type": <str>} from _listen_audio and
            # the phone relay.
            await self.session.send_realtime_input(
                audio=types.Blob(
                    data=msg["data"],
                    mime_type=msg.get("mime_type", "audio/pcm"),
                )
            )
            self._last_realtime_send = time.monotonic()
            if self._last_realtime_send - _t0 > 1.0:
                # The loop was free (see the watchdog) but the socket was not:
                # the network or the server is what held the audio up.
                print(f"[JARVIS] ⚠️ Sending audio to Gemini took "
                      f"{self._last_realtime_send - _t0:.1f}s (slow connection).")

    async def _listen_audio(self):
        print("[JARVIS] 🎤 Mic started")
        loop = asyncio.get_event_loop()
        # Load the sleep-check Whisper now, off the loop, so the first
        # "bye Jarvis" is not kept waiting for a model load.
        if self._sleep_stt is None and not self._sleep_stt_failed:
            loop.run_in_executor(None, self._warm_sleep_check)

        def callback(indata, frames, time_info, status):
            # ── Wake-word gate ───────────────────────────────────────────────
            # While asleep, the mic audio NEVER goes to Gemini (nothing is
            # streamed, so JARVIS can't respond to speech not addressed to it and
            # nothing leaves the machine). Frames are instead handed to the local
            # detector, which runs its model in ITS OWN thread — the cost here is
            # only a queue push, so the audio path is never slowed. When wake word
            # is off (default) or we're awake, this is a single boolean check.
            if self._wake_enabled and not self._awake:
                det = self._wake_detector
                # WAKE_SETTLE_SECONDS: drop, don't feed, for a moment after
                # going to sleep — otherwise the detector's first frames are
                # the acoustic tail of the utterance that just put it to
                # sleep ("...jarvis" fading out), which can score as a fresh
                # "Hey Jarvis" and wake it right back up.
                # Not while our own voice is still coming out of the speakers:
                # a reply finishing after "bye Jarvis" woke it straight back up
                # (2026-10-01, 19:58).
                if (det is not None and time.monotonic() >= self._wake_feed_gate_open_at
                        and not self._is_speaking and not self._tail_active()):
                    det.feed(indata)
                return
            # Keep the user's last few seconds for checking a sleep request
            # (SLEEP_CHECK_SECONDS) — a copy and an append, nothing more.
            if not self._is_speaking:
                self._remember_mic(indata)
            with self._speaking_lock:
                jarvis_speaking = self._is_speaking

            # ── Barge-in ─────────────────────────────────────────────────────
            # While JARVIS talks the mic is not streamed, but it is still worth
            # listening to locally: if the user starts speaking, cut the answer
            # short the way a person would stop when interrupted.
            #
            # The whole difficulty is echo — on speakers the mic hears JARVIS.
            # So the test is not "is the mic loud" but "is the mic louder than
            # the echo of what we are playing right now", sustained long enough
            # that a cough or a keystroke cannot trigger it.
            if jarvis_speaking:
                # Nothing is streamed while JARVIS talks.
                #
                # Interrupting by voice used to live here: `EchoGuard` can pick a
                # user out from under our own echo, and `core/echo.py` still does
                # that for the tail below. Re-enabling is small — classify each
                # block here and call interrupt() after `required_blocks` of
                # agreement — but it depends on the listener's room, so it stays
                # out until it can be tried on real hardware.
                return

            # ── Echo tail ────────────────────────────────────────────────────
            # The speaking flag has dropped but the speakers have not finished.
            # Sending this to the model is how an assistant hears itself, decides
            # it was addressed, and answers its own last sentence. The microphone
            # stays OPEN — the guard only drops blocks that are our own voice, so
            # replying the instant it stops still works.
            if self._tail_active():
                try:
                    if not self._echo.is_user_speech(
                            indata, SEND_SAMPLE_RATE, _pcm_level(indata)):
                        return
                    self._tail_until = 0.0      # a real voice ends the tail early
                except Exception:
                    return
            elif self._echo._hist:
                self._echo.reset()

            # ── Push-to-talk ─────────────────────────────────────────────────
            # When it is on the microphone is closed by default and the chord
            # opens it, which is the whole point: nothing leaves the machine
            # unless you are holding the key.
            if self._ptt_enabled and not self._ptt_held:
                return
            
            if not self.ui.muted and not self._phone_active:
                loop.call_soon_threadsafe(
                    self._enqueue_audio,
                    {"data": indata.tobytes(), "mime_type": "audio/pcm"}
                )
                # Feed the live mic level to the HUD so the waveform reacts to
                # the user's actual voice while listening. Purely cosmetic — any
                # failure here must never disturb the mic.
                try:
                    self.ui.set_audio_level(_pcm_level(indata))
                except Exception:
                    pass

        try:
            def _open_mic(dev):
                return sd.InputStream(
                    samplerate=SEND_SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=CHUNK_SIZE,
                    device=dev,
                    callback=callback,
                )

            # Which microphone. resolve() returns None for "system default" and
            # for a saved device that is no longer present — so a headset
            # unplugged since the last run falls back to the built-in mic
            # instead of raising on startup and taking the session with it.
            _mic_name = get_input_device()
            _mic_dev  = audio_devices.resolve(_mic_name, "input")
            if _mic_dev is not None:
                print(f"[JARVIS] 🎤 Input device: {_mic_name}")
            try:
                _mic_stream = _open_mic(_mic_dev)
            except Exception as _e:
                # A device the picker listed but the driver will not open right
                # now — exclusive mode, a webcam already in use, a virtual mic
                # whose source went away. Chosen hardware failing must never
                # mean the assistant cannot hear at all.
                if _mic_dev is None:
                    raise
                print(f"[JARVIS] ⚠️  Mic '{_mic_name}' failed: {_e} — using default")
                self.ui.write_log(
                    f"SYS: Microphone '{_mic_name}' unavailable — using system default."
                )
                _mic_stream = _open_mic(None)

            with _mic_stream:
                print("[JARVIS] 🎤 Mic stream open")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[JARVIS] ❌ Mic: {e}")
            raise

    async def _flush_pending_vision(self) -> bool:
        """Send a captured frame immediately after its tool response.

        The frame is already in hand by the time `screen_process` returns — the
        capture happened inside the tool call. The old flow still made the model
        speak a turn first and only injected the image on that turn's
        turn_complete, which cost a whole extra round trip AND produced two
        spoken answers: one improvised without the picture, then the real one.
        Sending it here means the model has the tool result and the image before
        it generates anything, so the user gets one answer, sooner.
        """
        if not (self._pending_vision and self.session):
            return False

        import base64 as _b64
        img_b, mime_t, question, angle = self._pending_vision
        self._pending_vision = None
        b64 = _b64.b64encode(img_b).decode("ascii")
        print(f"[Vision] 📤 {len(img_b):,} bytes (angle={angle}) → main session")

        # Label the source. Without it the image arrives carrying nothing but
        # the user's own sentence, and a screenshot of this app — which has a
        # face in the middle of it — got read as a photo of the user. What the
        # label *means* is explained once, in the generated [SELF] block.
        src = ("[IMAGE SOURCE: WEBCAM]" if angle == "camera"
               else "[IMAGE SOURCE: SCREEN CAPTURE]")
        await self.session.send_client_content(
            turns={"role": "user", "parts": [
                {"inline_data": {"mime_type": mime_t, "data": b64}},
                {"text": f"{src}\n\n{question}"},
            ]},
            turn_complete=True,
        )

        if self._vision_cam_active:
            # Camera: stay busy until JARVIS has finished speaking the answer,
            # then close the preview.
            self._vision_cam_active    = False
            self._vision_close_pending = True
        else:
            self._vision_busy = False
        return True

    async def _receive_audio(self):
        print("[JARVIS] 👂 Recv started")
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    # ── Session resumption ───────────────────────────────────
                    # The server sends this periodically. `resumable` goes false
                    # while a turn is mid-flight — replaying a handle from that
                    # moment is what the flag exists to prevent — so only
                    # resumable handles are kept. This is three lines and it is
                    # the entire fix for "every reconnect forgets everything".
                    # The server announces the end of a session (~10 min limit)
                    # with GoAway a little before it closes it. Ignoring it let
                    # the server kill the socket (1008 policy violation) —
                    # mid-sentence, with a traceback — before the reconnect.
                    # Instead reconnect ourselves at the next quiet moment.
                    if getattr(response, "go_away", None) is not None and not self._go_away_pending:
                        self._go_away_pending = True
                        asyncio.create_task(self._reconnect_before_go_away(response.go_away))

                    _sru = getattr(response, "session_resumption_update", None)
                    if _sru is not None:
                        if getattr(_sru, "resumable", False) and getattr(_sru, "new_handle", None):
                            if self._resume_handle is None:
                                print("[JARVIS] 🔗 Session resumption armed")
                            self._resume_handle = _sru.new_handle

                    if response.data:
                        self._generating = True
                        if self._interrupted or (self._wake_enabled and not self._awake):
                            pass  # discard: interrupted, or asleep (a reply to "bye" stays unsaid)
                        else:
                            if self._turn_done_event and self._turn_done_event.is_set():
                                self._turn_done_event.clear()
                            if self._current_turn:
                                self._current_turn.mark_once("first_audio")
                            # Split into ~50 ms chunks so interrupt() stops audio within 50 ms
                            # (24000 Hz × 2 bytes/sample × 0.05 s = 2400 bytes per slice)
                            _audio_data = response.data
                            _now = time.monotonic()
                            if self._reply_audio is None:
                                self._reply_audio = [_now, _now, 0]
                            self._reply_audio[1] = _now
                            self._reply_audio[2] += len(_audio_data)
                            _SLICE = 2400
                            for _i in range(0, len(_audio_data), _SLICE):
                                self.audio_in_queue.put_nowait(_audio_data[_i : _i + _SLICE])

                    if response.server_content:
                        sc = response.server_content

                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_transcript(sc.output_transcription.text)
                            # A turn that involves a tool call passes through
                            # several turn_completes, and the API re-sends the
                            # tail of the transcript across them. Comparing only
                            # against the previous chunk missed that — once
                            # out_buf had been flushed and emptied, the repeat
                            # sailed straight back in, which logged the answer
                            # twice AND made the avatar mouth it twice.
                            if txt and not _is_repeat_chunk(txt, out_buf):
                                out_buf.append(txt)
                                # Hand the words to the mouth as they arrive, so
                                # the avatar can form the consonants the audio
                                # alone cannot show. Pure string work — it adds
                                # nothing measurable to the response path.
                                self._visemes.feed_text(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt:
                                if not in_buf:
                                    self._in_started_at = time.monotonic()
                                    self._fast_voice_ran_turn = False
                                    if self._current_turn is None:
                                        self._current_turn = telemetry.start_turn("gemini")
                                in_buf.append(txt)
                                self._note_user_input(" ".join(in_buf))
                                self._schedule_fast_voice(" ".join(in_buf))
                                self._schedule_local_goodbye(" ".join(in_buf))

                        if sc.turn_complete:
                            if self._turn_done_event:
                                self._turn_done_event.set()

                            if self._current_turn:
                                self._current_turn.mark("model_done")
                                usage = getattr(response, "usage_metadata", None)
                                if usage is not None:
                                    self._current_turn.tokens(
                                        tokens_in=getattr(usage, "prompt_token_count", None),
                                        tokens_out=getattr(usage, "response_token_count", None),
                                    )

                            self._generating = False
                            self._report_reply_audio()
                            # If this turn_complete ends an interrupted response, clear the
                            # flag and skip all further processing for that turn.
                            if self._interrupted:
                                self._interrupted = False
                                in_buf  = []
                                out_buf = []
                                self._visemes.reset()
                                if self._current_turn:
                                    self._current_turn.set_interrupted()
                                    self._current_turn.finish()
                                    self._current_turn = None
                                continue

                            full_in = " ".join(in_buf).strip()
                            # Result contract: a user turn that began AFTER a
                            # tool failed closes the book on it — whatever
                            # JARVIS said in between is all it is going to say.
                            if full_in and self._false_success.pending \
                                    and self._in_started_at > (self._false_success.since or 0.0):
                                self._conclude_false_success(force=True)
                            if full_in:
                                self._last_out_logged = ""   # new exchange
                                self.ui.write_log(f"You: {full_in}")
                                self._session_log.append(f"User: {full_in}")
                                self._log_context_turn("user", full_in)
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "user",
                                        "text": full_in,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            # Second line of defence: even if a repeat slips
                            # into a *fresh* buffer after a flush, never log the
                            # same answer (or a tail of it) twice in a row.
                            if full_out and len(full_out) >= _REPEAT_MIN and self._last_out_logged:
                                if full_out in self._last_out_logged:
                                    full_out = ""
                            if full_out:
                                self._last_out_logged = full_out
                                self._note_spoken_for_contract(full_out)
                            self._conclude_false_success()
                            if full_out:
                                self.ui.write_log(f"{self._asst_name}: {full_out}")
                                self._session_log.append(f"{self._asst_name}: {full_out}")
                                self._log_context_turn("assistant", full_out)
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "jarvis",
                                        "text": full_out,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            out_buf = []

                            if self._current_turn:
                                self._current_turn.finish()
                                self._current_turn = None

                            if self._vision_close_pending:
                                # This turn_complete IS the vision answer — close camera + release busy flag
                                self._vision_close_pending = False
                                self._vision_busy = False
                                async def _cam_close():
                                    await asyncio.sleep(2.0)
                                    self.ui.stop_camera_stream()
                                asyncio.create_task(_cam_close())

                    if response.tool_call:
                        self._fast_voice_seq += 1     # the model got there first
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"[JARVIS] 📞 {fc.name}")
                            if self._runs_in_background(fc.name):
                                # Slow tool: this loop goes straight back to
                                # listening; the result is sent when ready.
                                self._start_background_tool(fc)
                                continue
                            span = (self._current_turn.tool_span(fc.name)
                                    if self._current_turn else contextlib.nullcontext())
                            # Counted so the sleep watch never puts JARVIS to
                            # sleep while a long tool is still working.
                            self._tools_running += 1
                            try:
                                with span:
                                    fr = await self._execute_tool(fc)
                            finally:
                                self._tools_running -= 1
                                self._touch_activity()
                            if fr is not None:          # `think` answers itself later
                                fn_responses.append(fr)
                        if fn_responses:
                            await self.session.send_tool_response(
                                function_responses=fn_responses
                            )
                        await self._flush_pending_vision()
        except Exception as e:
            print(f"[JARVIS] ❌ Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        print("[JARVIS] 🔊 Play started")

        _spk_name = get_output_device()
        _spk_dev  = audio_devices.resolve(_spk_name, "output")
        if _spk_dev is not None:
            print(f"[JARVIS] 🔊 Output device: {_spk_name}")

        def _open_spk(dev):
            st = sd.RawOutputStream(
                samplerate=RECEIVE_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                device=dev,
            )
            st.start()
            return st

        try:
            stream = _open_spk(_spk_dev)
        except Exception as _e:
            # A chosen output that the host API accepts by name but refuses to
            # open (exclusive mode, wrong sample rate, device asleep) must not
            # cost the user their voice. Fall back to the default and say so.
            if _spk_dev is None:
                raise
            print(f"[JARVIS] ⚠️  Output device '{_spk_name}' failed: {_e} — using default")
            self.ui.write_log(f"SYS: Speaker '{_spk_name}' unavailable — using system default.")
            stream = _open_spk(None)

        # Ask the device how far behind the speakers actually are, rather than
        # assuming. This is what the echo tail is sized from, so a machine with a
        # large audio buffer gets a correspondingly longer guard — and one with a
        # tiny buffer is not penalised with a delay it does not need.
        try:
            lat = float(getattr(stream, "latency", 0.0) or 0.0)
            if 0.0 < lat < 1.0:
                self._out_latency = lat
            print(f"[JARVIS] 🔊 Output latency {self._out_latency*1000:.0f} ms "
                  f"→ echo tail {(self._out_latency + _TAIL_MARGIN)*1000:.0f} ms")
        except Exception:
            pass

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_in_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    if (
                        self._turn_done_event
                        and self._turn_done_event.is_set()
                        and self.audio_in_queue.empty()
                    ):
                        self.set_speaking(False)
                        self._turn_done_event.clear()
                    continue

                # Already speaking = mid-reply: a starved speaker here is a real
                # gap. The first batch after silence always reports one.
                mid_reply = self._is_speaking
                self.set_speaking(True)

                # Batch all immediately-available chunks into one write to reduce
                # thread-pool round-trips (was one asyncio.to_thread per 50ms slice).
                # Cap at ~200 ms so interrupt() still stops audio within ~200 ms.
                batch = bytearray(chunk)
                while len(batch) < 9600:   # 9600 bytes ≈ 200 ms at 24 kHz / 16-bit mono
                    try:
                        batch.extend(self.audio_in_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                # Drive the HUD waveform and the avatar's mouth from JARVIS's
                # own voice. The batch is up to 200 ms long, so we hand over a
                # *schedule* of 20 ms viseme frames instead of a single averaged
                # level and let the HUD play it out in step with the audio.
                try:
                    pcm = np.frombuffer(bytes(batch), dtype=np.int16)
                    hop = _VIS_HOP / RECEIVE_SAMPLE_RATE
                    frames = _pcm_visemes(pcm, sr=RECEIVE_SAMPLE_RATE)
                    # When does this batch become audible? The stream was
                    # started at launch and its callback has been pulling
                    # silence ever since, so the first bytes of a reply reach
                    # the speaker about one callback period later — NOT one
                    # buffer later. `stream.latency` reports the buffer's
                    # capacity, which is how much can be queued ahead, and on
                    # Windows that is commonly 300-500 ms. Anchoring on it put
                    # the entire schedule a buffer late; that is the half second
                    # of lag, and it grew with whatever the device reported.
                    #
                    # After the anchor nothing needs measuring: the device
                    # consumes at exactly realtime, so each batch sounds one
                    # batch-duration after the one before it. The cursor is
                    # re-anchored only when it leaves the range physically
                    # possible — behind `now` means the device drained and this
                    # batch starts a fresh stretch of speech, while further
                    # ahead than the buffer can hold means it has drifted.
                    now = time.time()
                    horizon = self._out_latency + _CURSOR_SLACK
                    if not (now <= self._play_cursor <= now + horizon):
                        self._play_cursor = now + _FIRST_SOUND
                    at = self._play_cursor
                    # Advance by the batch's own duration whether or not it
                    # yielded frames, so a block too short to analyse cannot
                    # shift everything after it out of step with the audio.
                    self._play_cursor += pcm.size / RECEIVE_SAMPLE_RATE
                    if frames:
                        frames = self._visemes.frames(frames, hop)
                        self.ui.push_visemes(frames, hop, at)
                        # Barge-in needs to know what we are playing, not just
                        # how loud: the guard subtracts this from the microphone.
                        self._out_level = max(f[0] for f in frames)
                        self._echo.note_output(pcm, RECEIVE_SAMPLE_RATE,
                                               self._out_level)
                    else:
                        lvl = _pcm_level(pcm)
                        self.ui.set_audio_level(lvl)
                        self._out_level = lvl
                        self._echo.note_output(pcm, RECEIVE_SAMPLE_RATE, lvl)
                except Exception:
                    pass

                try:
                    underflowed = await asyncio.to_thread(stream.write, bytes(batch))
                    if underflowed and mid_reply:
                        self._underruns += 1
                except (RuntimeError, asyncio.CancelledError):
                    break   # executor shutting down — exit cleanly
        except Exception as e:
            print(f"[JARVIS] ❌ Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()
