"""
Automatic Local Mode when Gemini is unavailable.

Before this, a day's quota running out or Gemini having a bad hour meant
JARVIS went silent: the run loop retried forever, and Local Mode — which
already existed — was only reachable by switching ENGINE and restarting.

Now the run loop keeps a CloudHealth record. When every Live model is resting
(quota / 1011s) or connecting has failed FAIL_COUNT times in FAIL_WINDOW
seconds, and Local Mode is ready (local LLM reachable, local speech-to-text
installed, some voice to speak with), JARVIS switches to Local Mode on its
own. A probe checks the cloud every PROBE_SECONDS and switches back as soon as
it can — Local Mode is the spare tyre, not the destination.

Failures that fix themselves on the next connect (idle drops, a settings
reconnect, a rejected resume handle, dropped preview options) never count. A
bad API key does not either: that needs the user, not a different engine.
"""
from __future__ import annotations

import importlib.util
import shutil
import socket
import sys
import time
from typing import Callable

FAIL_COUNT = 3
FAIL_WINDOW = 120.0
PROBE_SECONDS = 60.0
CLOUD_HOST = "generativelanguage.googleapis.com"

# _classify_live_error kinds (main.py) that mean "the cloud is not working".
COUNTED_KINDS = {"network", "other"}


class CloudHealth:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._failures: list[float] = []

    def failed(self, kind: str) -> None:
        if kind in COUNTED_KINDS:
            self._failures.append(self._clock())

    def connected(self) -> None:
        self._failures.clear()

    def recent_failures(self) -> int:
        cutoff = self._clock() - FAIL_WINDOW
        self._failures = [t for t in self._failures if t >= cutoff]
        return len(self._failures)

    def should_fall_back(self, models_resting: bool) -> bool:
        return models_resting or self.recent_failures() >= FAIL_COUNT


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def tts_available(cfg: dict) -> bool:
    engine = str(cfg.get("tts_engine", "edgetts")).lower()
    wanted = {"kokoro": "kokoro", "elevenlabs": "requests", "edgetts": "edge_tts"}.get(engine, "edge_tts")
    return _have(wanted) or mac_say_available()


def mac_say_available() -> bool:
    return sys.platform == "darwin" and shutil.which("say") is not None


def local_readiness(cfg: dict, llm_reachable: Callable[[], bool]) -> tuple[bool, list[str]]:
    """(ready, what is missing). `llm_reachable` is llm_client.ensure_ollama_running
    in production — it may take a few seconds, so call this off the event loop."""
    missing = []
    stt = str(cfg.get("local_stt_engine", "whisper")).lower()
    if stt == "vosk":
        if not _have("vosk"):
            missing.append("speech-to-text (pip install vosk)")
    elif not _have("faster_whisper"):
        missing.append("speech-to-text (pip install faster-whisper)")
    if not tts_available(cfg):
        missing.append("a voice (pip install edge-tts, or kokoro)")
    try:
        if not llm_reachable():
            missing.append("a local LLM (install Ollama from ollama.com, then `ollama pull llama3.2`)")
    except Exception as e:
        missing.append(f"a local LLM ({e})")
    return (not missing), missing


def cloud_reachable(timeout: float = 3.0) -> bool:
    """Can we open a TCP connection to the Gemini API at all?"""
    try:
        with socket.create_connection((CLOUD_HOST, 443), timeout=timeout):
            return True
    except OSError:
        return False
