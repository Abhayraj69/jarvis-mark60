"""
Local wake-word detection for JARVIS ("Hey Jarvis").

Design goals:
  • ZERO cost when the feature is off — openwakeword is imported ONLY inside
    start()/install helpers, never at module load. If the user never enables
    wake word, none of this touches the app.
  • ZERO latency on the audio path — the microphone callback only ever does a
    cheap, non-blocking queue push (feed()); the actual model inference runs in
    this module's own background thread, so the real-time audio thread and the
    Gemini stream are never slowed.
  • Fully local & offline — audio fed here never leaves the machine; there is no
    network call except the one-time model download the user triggers from the UI.

openwakeword ships small ONNX models (a few MB each) and runs comfortably on a
CPU. The pretrained wake phrase used here is "Hey Jarvis".
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

# Pretrained openwakeword model that listens for "Hey Jarvis".
WAKE_MODEL = "hey_jarvis"
# Score in [0,1]; above this counts as a detection. Tunable per environment.
# 0.5 (openwakeword's generic default) made a normal-volume "Hey Jarvis" from
# across the desk miss often enough that it had to be repeated or shouted;
# real voices on a laptop mic peak well below the ~1.0 a clean recording gets.
DEFAULT_THRESHOLD = 0.35
# Scores at or above this but under the threshold are logged as near-misses,
# so the threshold can be tuned from the console instead of guessed.
NEAR_MISS_SCORE = 0.15
# The queue only has to absorb a stall, never to drop audio: _loop() empties
# it in one batch, so this is ~16 s of 1024-sample frames of headroom.
QUEUE_FRAMES = 250
# Mic frames arrive at 16 kHz int16; this is just the detector's input rate.
SAMPLE_RATE = 16000

# openwakeword is imported lazily from more than one thread (the UI polls
# is_ready() while the audio thread's start() loads the Model). Two threads
# importing it at once can hand one of them a half-initialised module — "has no
# attribute 'get_pretrained_model_paths' (most likely due to a circular
# import)" — and the wake word is then unavailable for the whole run. Every
# import of it goes through this lock.
_IMPORT_LOCK = threading.RLock()


def is_installed() -> bool:
    """True if the openwakeword package is importable (no model check)."""
    try:
        import importlib.util
        return importlib.util.find_spec("openwakeword") is not None
    except Exception:
        return False


def is_ready() -> bool:
    """True if openwakeword is installed AND its model files are present on disk.

    This is a cheap, DETERMINISTIC file-existence check. It deliberately does NOT
    construct a Model to probe readiness — doing that is slow and, worse, can clash
    with the detector's own Model when it's already running, which intermittently
    returned False and made the UI flicker to 'not downloaded'. Never raises.
    """
    if not is_installed():
        return False
    try:
        with _IMPORT_LOCK:
            import openwakeword
        models_dir = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
        if not models_dir.is_dir():
            return False
        has_wake = (any(models_dir.glob(f"{WAKE_MODEL}*.onnx"))
                    or any(models_dir.glob(f"{WAKE_MODEL}*.tflite")))
        has_mel = (any(models_dir.glob("melspectrogram*.onnx"))
                   or any(models_dir.glob("melspectrogram*.tflite")))
        has_emb = (any(models_dir.glob("embedding_model*.onnx"))
                   or any(models_dir.glob("embedding_model*.tflite")))
        return bool(has_wake and has_mel and has_emb)
    except Exception:
        return False


def install_and_download(logger: Callable[[str], None] = print,
                         notify: Callable[[str], None] | None = None) -> tuple[bool, str]:
    """
    One-click setup for the UI button: pip-install openwakeword if missing, then
    download the wake model. Returns (ok, message). Never raises — every failure
    is reported through the returned message and the logger.
    """
    _tell = notify or (lambda _msg: None)
    try:
        if not is_installed():
            logger("Wake word: installing openwakeword (one-time)…")
            _tell("Wake word: installing openwakeword (one-time)…")
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "openwakeword"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
                return False, f"pip install failed: {tail[0][:160]}"
        # Download the pretrained melspectrogram/embedding + wake models.
        logger("Wake word: downloading models…")
        _tell("Wake word: downloading models…")
        try:
            with _IMPORT_LOCK:
                import openwakeword.utils as _u
            try:
                _u.download_models([WAKE_MODEL])
            except TypeError:
                _u.download_models()   # older signature downloads the default set
        except Exception as e:
            return False, f"model download failed: {e}"

        if not is_ready():
            return False, "installed, but the wake model could not be loaded."
        logger("Wake word: ready.")
        return True, "Wake word installed and ready."
    except Exception as e:
        return False, f"setup error: {e}"


class WakeWordDetector:
    """
    Runs the wake model in a dedicated thread. The mic thread calls feed() with
    raw int16 frames; detections invoke on_detect() (called from this thread —
    the callback must marshal to whatever loop/UI it needs).
    """

    def __init__(self, on_detect: Callable[[], None],
                 threshold: float = DEFAULT_THRESHOLD,
                 logger: Callable[[str], None] = print,
                 notify: Callable[[str], None] | None = None):
        self._on_detect = on_detect
        self._threshold = threshold
        self._logger    = logger
        # See PluginRegistry: `logger` is the console and gets everything,
        # `notify` is the activity log and gets only what the user must act on.
        self._notify    = notify or (lambda _msg: None)
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_FRAMES)
        self._thread: threading.Thread | None = None
        self._running = False
        self._model = None
        self._ready = False

    def start(self) -> bool:
        """Load the model and spawn the inference thread. Returns True on success.
        Safe to call again — a no-op if already running. Never raises."""
        if self._running:
            return True
        try:
            with _IMPORT_LOCK:
                from openwakeword.model import Model
            self._model = Model(wakeword_models=[WAKE_MODEL], inference_framework="onnx")
        except Exception as e:
            self._logger(f"Wake word: could not load model — {e}")
            self._notify("Wake word unavailable — use the WAKE NOW button.")
            self._model = None
            return False
        self._running = True
        self._ready = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="WakeWordThread")
        self._thread.start()
        self._logger("Wake word: listening for 'Hey Jarvis'.")
        return True

    def reset(self) -> None:
        """Clear the model's internal audio/embedding buffers before a fresh
        listening session (see WakeWordDetector.start()'s docstring for why
        this matters here). Safe to call any time the detector isn't
        mid-predict() — true whenever nothing has called feed() since the
        last detection, which is exactly how the caller in main.py uses it
        (right as it re-arms the detector, before any new audio is fed)."""
        if self._model is not None:
            try:
                self._model.reset()
            except Exception as e:
                self._logger(f"Wake word: reset failed — {e}")

    def stop(self) -> None:
        self._running = False
        # unblock the thread if it's waiting on the queue
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self._model = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def feed(self, frame_int16) -> None:
        """Called from the mic callback (real-time thread). Must stay cheap and
        never block — the frame is copied and dropped only if the queue is
        completely full, which _loop()'s batching makes a many-second stall."""
        if not self._running:
            return
        try:
            # frame_int16 is a numpy int16 array (possibly 2-D mono) — flatten to 1-D
            data = frame_int16[:, 0].copy() if getattr(frame_int16, "ndim", 1) > 1 else frame_int16.copy()
            self._queue.put_nowait(data)
        except queue.Full:
            pass
        except Exception:
            pass

    def _loop(self) -> None:
        import time
        import numpy as np
        last_near_miss_log = 0.0
        last_backlog_log = 0.0
        while self._running:
            try:
                frame = self._queue.get()
                if frame is None or not self._running:
                    break
                # Take everything that queued up behind this frame and score it
                # in ONE predict() call. When the CPU is busy (HUD rendering,
                # Whisper in the widget, a Gemini turn) inference falls behind;
                # scoring one frame at a time then let the queue fill and throw
                # new audio away, and a "Hey Jarvis" with holes cut in it often
                # scores far below threshold — so it had to be said again.
                # predict() on a longer buffer runs every 80 ms step inside it
                # and returns the max, so batching loses nothing.
                frames = [frame]
                stop = False
                while True:
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        stop = True
                        break
                    frames.append(nxt)
                if stop or not self._running:
                    break
                now = time.monotonic()
                if len(frames) >= 8 and now - last_backlog_log > 5.0:
                    last_backlog_log = now
                    self._logger(f"Wake word: caught up on a {len(frames)}-frame backlog "
                                 f"(CPU busy) — no audio dropped.")
                audio = frames[0] if len(frames) == 1 else np.concatenate(frames)
                scores = self._model.predict(np.asarray(audio, dtype=np.int16))
                score = 0.0
                if isinstance(scores, dict):
                    # match the jarvis model regardless of exact key suffix
                    for k, v in scores.items():
                        if "jarvis" in k.lower():
                            score = max(score, float(v))
                    if score == 0.0 and scores:
                        score = max(float(v) for v in scores.values())
                if score >= self._threshold:
                    # drain any backlog so we don't double-fire on the same utterance
                    self._drain()
                    # Clear the model's rolling window here, on the thread that
                    # owns it: otherwise predict() on the next sub-80 ms frame
                    # returns the previous (above-threshold) score and fires again.
                    self._reset_model()
                    self._logger(f"Wake word: 'Hey Jarvis' detected (score {score:.2f}).")
                    try:
                        self._on_detect()
                    except Exception as e:
                        self._logger(f"Wake word: on_detect error — {e}")
                elif score >= NEAR_MISS_SCORE and now - last_near_miss_log > 1.0:
                    last_near_miss_log = now
                    self._logger(f"Wake word: near miss (score {score:.2f}, "
                                 f"threshold {self._threshold:.2f}).")
            except Exception as e:
                self._logger(f"Wake word: inference error — {e}")

    def _reset_model(self) -> None:
        try:
            if self._model is not None:
                self._model.reset()
        except Exception:
            pass

    def _drain(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass
