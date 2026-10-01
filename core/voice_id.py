"""
Voice ID: JARVIS only listens to its owner.

Before this, anything the microphone heard while JARVIS was awake went to
Gemini: a side conversation, a video playing, someone else in the room. It
answered them, and acted on them ("open YouTube" from a misheard sentence).

HOW IT WORKS
    Each utterance is checked locally before any of it is sent:

      1. Quiet audio passes straight through, so Gemini still hears the pauses
         that end a turn.
      2. When speech starts, up to VERIFY_SECONDS of it is held back, turned
         into a speaker embedding, and compared with the enrolled profile.
      3. Owner: the held audio is released and the rest of the utterance
         streams normally. Anyone else: the held audio and the rest of that
         utterance are replaced with silence of the same length, so timing is
         unchanged and nothing they said leaves the machine.

    The cost is about one second of delay at the start of each utterance;
    Gemini receives that second in a burst and catches up.

    "Hey Jarvis" is checked the same way, against the last couple of seconds
    of audio, so another voice cannot wake it either.

DESIGN
    • Off and free unless enabled — sherpa-onnx is imported only when a gate is
      built, the same way core/wake_word.py treats openwakeword.
    • Never in the audio callback — embeddings run on this module's own thread;
      the callback only pushes frames onto a queue.
    • Fails open — if the model will not load, JARVIS says so and keeps
      working without voice ID rather than going deaf.
"""
from __future__ import annotations

import collections
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000

_BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = _BASE_DIR / "memory" / "models"
MODEL_FILE = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
MODEL_PATH = MODEL_DIR / MODEL_FILE
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             f"speaker-recongition-models/{MODEL_FILE}")
PROFILE_PATH = _BASE_DIR / "memory" / "voice_profile.npy"

# Cosine similarity at or above this is the owner. CAM++ puts the same speaker
# around 0.6-0.8 on a laptop mic and different speakers below ~0.3; 0.45
# leaves room for a cold or a different microphone. Tunable in settings, and
# every decision is printed with its score so it can be tuned from the console.
DEFAULT_THRESHOLD = 0.45
# How much speech to hold back before deciding. Shorter is faster but the
# embedding gets noisier; one second is enough for CAM++.
VERIFY_SECONDS = 1.0
# This much quiet ends an utterance (and the decision that covered it).
HANGOVER_SECONDS = 0.8
# Below this RMS (0..1) nothing counts as speech, however quiet the room.
MIN_SPEECH_LEVEL = 0.012
# Speech must be this many times the room's noise floor.
SPEECH_OVER_FLOOR = 3.0
# How much audio to keep while asleep, for checking "Hey Jarvis".
WAKE_WINDOW_SECONDS = 1.6
# Enrollment: how long to record, and the window each embedding covers.
ENROLL_SECONDS = 12.0
ENROLL_WINDOW_SECONDS = 2.0


# ── Install / readiness ──────────────────────────────────────────────────────

def is_installed() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("sherpa_onnx") is not None
    except Exception:
        return False


def is_ready() -> bool:
    """Package, model and an enrolled profile are all present."""
    return is_installed() and MODEL_PATH.exists() and PROFILE_PATH.exists()


def install(logger: Callable[[str], None] = print) -> tuple[bool, str]:
    """pip-install sherpa-onnx and download the speaker model. Never raises."""
    try:
        if not is_installed():
            logger("Voice ID: installing sherpa-onnx…")
            r = subprocess.run([sys.executable, "-m", "pip", "install", "sherpa-onnx"],
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                return False, f"pip install sherpa-onnx failed: {r.stderr.strip()[-300:]}"
        if not MODEL_PATH.exists():
            logger("Voice ID: downloading the speaker model…")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            tmp = MODEL_PATH.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.replace(MODEL_PATH)
        return True, "Voice ID model installed."
    except Exception as e:
        return False, f"Voice ID install failed: {e}"


# ── Embeddings ───────────────────────────────────────────────────────────────

def _to_float(block) -> np.ndarray:
    a = np.asarray(block)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    return a.reshape(-1).astype(np.float32)


def level(block) -> float:
    a = _to_float(block)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Embedder:
    """CAM++ speaker embeddings through sherpa-onnx."""

    def __init__(self, model_path: Path = MODEL_PATH):
        import sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=1)
        if not cfg.validate():
            raise RuntimeError(f"invalid speaker model: {model_path}")
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        s = self._ex.create_stream()
        s.accept_waveform(sample_rate=SAMPLE_RATE, waveform=_to_float(audio))
        s.input_finished()
        return np.asarray(self._ex.compute(s), dtype=np.float32)


def build_profile(embed: Callable[[np.ndarray], np.ndarray], audio: np.ndarray) -> np.ndarray:
    """Average the embeddings of every voiced window in `audio`."""
    audio = _to_float(audio)
    win = int(ENROLL_WINDOW_SECONDS * SAMPLE_RATE)
    embs = []
    for start in range(0, max(len(audio) - win, 0) + 1, win // 2):
        chunk = audio[start:start + win]
        if len(chunk) == win and level(chunk) >= MIN_SPEECH_LEVEL:
            e = embed(chunk)
            embs.append(e / (np.linalg.norm(e) or 1.0))
    if len(embs) < 3:
        raise ValueError("not enough speech was recorded — speak continuously and try again")
    prof = np.mean(embs, axis=0)
    return prof / (np.linalg.norm(prof) or 1.0)


def enroll(record: Callable[[float], np.ndarray], logger: Callable[[str], None] = print,
           embed: Callable[[np.ndarray], np.ndarray] | None = None) -> tuple[bool, str]:
    """Record the owner and save their profile. `record(seconds)` returns mono
    16 kHz audio. Never raises."""
    try:
        embed = embed or Embedder()
        logger(f"Voice ID: recording for {ENROLL_SECONDS:.0f} seconds — talk normally "
               "(read something aloud) until it finishes.")
        audio = record(ENROLL_SECONDS)
        prof = build_profile(embed, audio)
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(PROFILE_PATH, prof)
        return True, "Voice ID enrolled. JARVIS will now only answer your voice."
    except Exception as e:
        return False, f"Voice ID enrollment failed: {e}"


def record_mic(seconds: float) -> np.ndarray:
    import sounddevice as sd
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    return audio.reshape(-1)


# ── The gate ─────────────────────────────────────────────────────────────────

class _State(Enum):
    IDLE = "idle"
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class VoiceGate:
    """Decides, utterance by utterance, whether mic audio is the owner's.

    process() is the pure core: frames in, frames to send out (real or
    silenced). start()/feed() run it on a worker thread for the live mic.
    """

    def __init__(self, embed: Callable[[np.ndarray], np.ndarray], profile: np.ndarray,
                 threshold: float = DEFAULT_THRESHOLD,
                 on_decision: Callable[[bool, float], None] | None = None):
        self._embed = embed
        self._profile = np.asarray(profile, dtype=np.float32)
        self.threshold = float(threshold)
        self._on_decision = on_decision
        self._state = _State.IDLE
        self._held: list[np.ndarray] = []
        self._quiet = 0.0
        self._floor = MIN_SPEECH_LEVEL / SPEECH_OVER_FLOOR
        self._recent: collections.deque = collections.deque()
        self._recent_len = 0
        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None

    # pure core ----------------------------------------------------------------

    def score(self, audio: np.ndarray) -> float:
        return cosine(self._embed(audio), self._profile)

    def is_owner(self, audio: np.ndarray) -> bool:
        s = self.score(audio)
        ok = s >= self.threshold
        if self._on_decision:
            try:
                self._on_decision(ok, s)
            except Exception:
                pass
        return ok

    def _is_speech(self, lvl: float) -> bool:
        return lvl >= max(MIN_SPEECH_LEVEL, self._floor * SPEECH_OVER_FLOOR)

    def process(self, block) -> list[bytes]:
        """One mic block in; zero or more blocks of int16 PCM bytes out."""
        block = np.asarray(block, dtype=np.int16).reshape(-1)
        dur = len(block) / SAMPLE_RATE
        lvl = level(block)
        speech = self._is_speech(lvl)
        if not speech:
            # Track the room so a noisy fan does not count as talking.
            self._floor = 0.95 * self._floor + 0.05 * lvl
        self._quiet = 0.0 if speech else self._quiet + dur

        if self._state is _State.IDLE:
            if not speech:
                return [block.tobytes()]
            self._state, self._held = _State.PENDING, [block]
            return []

        if self._state is _State.PENDING:
            self._held.append(block)
            held = np.concatenate(self._held)
            ended = self._quiet >= HANGOVER_SECONDS
            if len(held) / SAMPLE_RATE < VERIFY_SECONDS and not ended:
                return []
            owner = self.is_owner(held)
            self._held = []
            if ended:
                self._state = _State.IDLE
            else:
                self._state = _State.ACCEPTED if owner else _State.REJECTED
            return [held.tobytes()] if owner else [bytes(held.nbytes)]

        out = block.tobytes() if self._state is _State.ACCEPTED else bytes(block.nbytes)
        if self._quiet >= HANGOVER_SECONDS:
            self._state = _State.IDLE
        return [out]

    def reset(self) -> None:
        self._state, self._held, self._quiet = _State.IDLE, [], 0.0

    def set_profile(self, profile: np.ndarray) -> None:
        self._profile = np.asarray(profile, dtype=np.float32)

    # wake word ----------------------------------------------------------------

    def note_recent(self, block) -> None:
        """Keep the last WAKE_WINDOW_SECONDS of audio (called while asleep)."""
        b = np.asarray(block, dtype=np.int16).reshape(-1).copy()
        self._recent.append(b)
        self._recent_len += len(b)
        limit = int(WAKE_WINDOW_SECONDS * SAMPLE_RATE)
        while self._recent and self._recent_len - len(self._recent[0]) >= limit:
            self._recent_len -= len(self._recent.popleft())

    def recent_is_owner(self) -> bool:
        """Was the wake phrase that just fired said by the owner?"""
        if not self._recent:
            return False
        return self.is_owner(np.concatenate(list(self._recent)))

    # live thread --------------------------------------------------------------

    def start(self, sink: Callable[[bytes], None]) -> None:
        """Run process() on a worker thread, handing results to `sink`."""
        if self._thread is not None:
            return
        self._q = queue.Queue(maxsize=400)

        def _loop():
            while True:
                block = self._q.get()
                try:
                    for out in self.process(block):
                        sink(out)
                except Exception as e:
                    print(f"[VoiceID] ⚠️ {e} — passing audio through")
                    self.reset()
                    sink(np.asarray(block, dtype=np.int16).tobytes())

        self._thread = threading.Thread(target=_loop, daemon=True, name="voice-id")
        self._thread.start()

    def feed(self, block) -> None:
        """Called from the audio callback: a copy and a queue push, nothing more."""
        if self._q is None:
            return
        try:
            self._q.put_nowait(np.array(block, dtype=np.int16).reshape(-1))
        except queue.Full:
            pass


def load_profile() -> np.ndarray:
    return np.load(PROFILE_PATH)


def load_gate(threshold: float = DEFAULT_THRESHOLD,
              on_decision: Callable[[bool, float], None] | None = None) -> VoiceGate:
    """Build the live gate from the installed model and saved profile."""
    return VoiceGate(Embedder(), load_profile(), threshold, on_decision)


if __name__ == "__main__":
    # python -m core.voice_id enroll   — the same thing the settings button does
    if sys.argv[1:] == ["enroll"]:
        ok, msg = install()
        print(msg)
        if ok:
            time.sleep(0.5)
            print(enroll(record_mic)[1])
    else:
        print("usage: python -m core.voice_id enroll")
