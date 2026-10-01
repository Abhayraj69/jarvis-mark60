"""Replay scripted Gemini Live conversations through the real JarvisLive.

A script is a list of steps. Each step is either a server message built with
the helpers below (heard, said, audio, turn_complete, tool_call, go_away) or a
plain callable, which runs between messages — that is how a test interrupts,
wakes, or lets time pass at an exact point in the conversation:

    j = make_jarvis()
    play(j, [
        heard("open chrome"),
        tool_call("open_app", {"app_name": "chrome"}),
        said("Opening Chrome."), audio(), turn_complete(),
    ])
    assert "JARVIS: Opening Chrome." in j.ui.logs

Nothing here touches the network, the microphone, the speakers, or the
files under memory/ — telemetry, the cross-session turn store, connectors and
the wake-word model are replaced with stand-ins.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import main


# ── Server messages ──────────────────────────────────────────────────────────

def _msg(**kw):
    base = dict(data=None, server_content=None, tool_call=None, go_away=None,
                session_resumption_update=None, usage_metadata=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _content(**kw):
    base = dict(output_transcription=None, input_transcription=None, turn_complete=False)
    base.update(kw)
    return _msg(server_content=SimpleNamespace(**base))


def heard(text: str):
    """The user's speech, as the server transcribes it."""
    return _content(input_transcription=SimpleNamespace(text=text))


def said(text: str):
    """JARVIS's own words (output transcription)."""
    return _content(output_transcription=SimpleNamespace(text=text))


def audio(n_bytes: int = 4800):
    """A chunk of JARVIS's reply audio (24 kHz int16; 4800 bytes = 100 ms)."""
    return _msg(data=bytes(n_bytes))


def turn_complete():
    return _content(turn_complete=True)


def tool_call(name: str, args: dict | None = None, call_id: str = "call-1"):
    fc = SimpleNamespace(name=name, args=dict(args or {}), id=call_id)
    return _msg(tool_call=SimpleNamespace(function_calls=[fc]))


def go_away(time_left: str = "10s"):
    return _msg(go_away=SimpleNamespace(time_left=time_left))


def resumable(handle: str = "handle-1"):
    return _msg(session_resumption_update=SimpleNamespace(resumable=True, new_handle=handle))


def elapse(seconds: float):
    """A step: move every 'last happened at' clock back, as if `seconds` passed."""
    def _step(j):
        for attr in ("_last_activity", "_last_user_speech", "_last_realtime_send",
                     "_session_started_at"):
            setattr(j, attr, getattr(j, attr) - seconds)
    _step.__name__ = f"elapse({seconds})"
    return _step


def pause(seconds: float):
    """A step: real time passes with nothing new from the server (lets
    timers such as the spoken-command settle delay fire)."""
    def _step(_j):
        return asyncio.sleep(seconds)
    _step.__name__ = f"pause({seconds})"
    return _step


# ── Stand-ins ────────────────────────────────────────────────────────────────

class FakeUI:
    """Records what the HUD was told. Anything not modelled is a no-op."""

    def __init__(self):
        self.muted = False
        self.states: list[str] = []
        self.logs: list[str] = []

    def set_state(self, state):
        self.states.append(state)

    def write_log(self, text):
        self.logs.append(text)

    @property
    def state(self):
        return self.states[-1] if self.states else None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


class FakeWakeDetector:
    def __init__(self, *a, **k):
        self.ready = True

    def start(self):
        return True

    def reset(self):
        pass

    def feed(self, _frames):
        pass


class FakeSession:
    """Plays a script through receive() and records everything sent back."""

    def __init__(self, jarvis, steps):
        self._jarvis = jarvis
        self._steps = list(steps)
        self.tool_responses: list = []
        self.client_content: list = []
        self.realtime: list = []

    async def receive(self):
        while self._steps:
            step = self._steps.pop(0)
            if callable(step):
                res = step(self._jarvis)
                if inspect.isawaitable(res):
                    await res
                continue
            yield step
        raise ScriptDone

    async def send_tool_response(self, function_responses):
        self.tool_responses.extend(function_responses)

    async def send_client_content(self, turns=None, turn_complete=True):
        self.client_content.append(turns)

    async def send_realtime_input(self, audio=None, **_):
        self.realtime.append(audio)


class ScriptDone(Exception):
    """Raised by FakeSession once every step has been played."""


# ── Building and driving JarvisLive ──────────────────────────────────────────

class Slow:
    """A tool result that takes `seconds` to arrive: make_jarvis(tools={"web_search": Slow(2, "...")})."""

    def __init__(self, seconds: float, result: str):
        self.seconds, self.result = seconds, result


def make_jarvis(*, wake_word: bool = True, awake: bool = True, tools: dict | None = None):
    """A real JarvisLive wired to stand-ins. `tools` maps a tool name to the
    result string _dispatch_tool should return for it."""
    registry = MagicMock()
    registry.discover.return_value = registry
    registry.get_tool_declarations.return_value = []

    with patch.object(main, "get_wake_word_enabled", return_value=wake_word), \
         patch.object(main, "get_push_to_talk_enabled", return_value=False), \
         patch.object(main, "get_plugin_config", return_value={}), \
         patch.object(main, "is_local_engine_enabled", return_value=False), \
         patch.object(main, "ToolRegistry", return_value=registry):
        j = main.JarvisLive(FakeUI())

    j._awake = awake
    j._has_connected = True          # scripts start mid-run, not at first launch
    j.audio_in_queue = asyncio.Queue()
    j.out_queue = asyncio.Queue()
    j._turn_done_event = asyncio.Event()
    j._session_started_at = time.monotonic()
    j._last_realtime_send = time.monotonic()

    j.dispatched: list[tuple[str, dict]] = []
    results = dict(tools or {})

    async def _dispatch(name, args):
        j.dispatched.append((name, args))
        res = results.get(name, f"{name} done.")
        if isinstance(res, Slow):
            await asyncio.sleep(res.seconds)
            res = res.result
        return res
    j._dispatch_tool = _dispatch
    return j


_STUBS = None


def _start_stubs():
    """Telemetry, the turn store and the wake model write to memory/ or load
    models; replace them for the whole test process."""
    global _STUBS
    if _STUBS is None:
        _STUBS = [
            patch.object(main, "telemetry", MagicMock()),
            patch.object(main, "context_manager", MagicMock()),
            patch.object(main, "WakeWordDetector", FakeWakeDetector),
        ]
        for p in _STUBS:
            p.start()


_start_stubs()


def play(j, steps) -> FakeSession:
    """Run the script through j._receive_audio and return the session, which
    holds everything JARVIS sent back."""
    session = FakeSession(j, steps)
    j.session = session

    async def _run():
        j._loop = asyncio.get_running_loop()
        # Queues are bound to the loop that first awaits them; rebuild per run.
        j.audio_in_queue = asyncio.Queue()
        j.out_queue = asyncio.Queue()
        j._turn_done_event = asyncio.Event()
        try:
            await j._receive_audio()
        except ScriptDone:
            pass
        await asyncio.sleep(0)       # let fire-and-forget tasks settle

    asyncio.run(_run())
    return session


def queued_audio_bytes(j) -> int:
    total = 0
    while not j.audio_in_queue.empty():
        total += len(j.audio_in_queue.get_nowait())
    return total


def tick_sleep_watch(j) -> None:
    """Run exactly one pass of j._run_sleep_watch (auto-sleep + keepalive)."""
    async def _run():
        real_sleep = asyncio.sleep
        calls = 0

        async def one_pass(_):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise ScriptDone
            await real_sleep(0)

        main.asyncio.sleep = one_pass
        try:
            await j._run_sleep_watch()
        except ScriptDone:
            pass
        finally:
            main.asyncio.sleep = real_sleep

    asyncio.run(_run())
