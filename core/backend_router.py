"""
core/backend_router.py — per-task-kind backend selection with health-based
failover.

WHY THIS EXISTS
    Three backends are wired into this app — Gemini (google-genai), Ollama or
    an OpenAI-compatible server (core/llm_client.py), and Claude
    (core/claude_bridge.py) — but which one answers a given request has been
    one global setting. That means a trivial voice command pays for whatever
    the heaviest configured engine costs, a sub-task like dev_agent's code
    generation runs through whichever engine happens to be selected for
    voice, and if Ollama is down, everything that depended on it just fails
    instead of quietly trying something else. This router picks a backend
    per TASK KIND from an explicit ordered policy, skips a backend for 60s
    after it fails (a circuit breaker, so a downed server isn't retried on
    every single call) and skips it entirely if it isn't configured at all.

    The Gemini Live voice session is NOT routed through this — it's a
    persistent bidirectional audio stream, not a one-shot completion, and it
    stays directly on Gemini. This module is for everything else: one-shot
    sub-task text generation such as actions/dev_agent.py and
    actions/code_helper.py use today via their own hand-rolled Claude/Gemini
    switch, and any future one-shot model call that wants a task-appropriate
    backend with a fallback instead of a single hardcoded choice.

USAGE
    from core.backend_router import TaskKind, complete

    result = complete(TaskKind.CODE_GEN, messages=[{"role": "user", "content": "..."}])
    # {"content": str, "tool_calls": list, "usage": dict, "backend": "claude"}

    # For callers using the .generate_content(prompt).text convention that
    # actions/dev_agent.py and actions/code_helper.py already share:
    model = get_text_model(TaskKind.CODE_GEN)
    model.generate_content("write a haiku").text

    # Vision: images are (bytes, mime) tuples; only the Gemini backends take
    # them, so VISION policies list gemini / gemini_lite only.
    complete(TaskKind.VISION, [{"role": "user", "content": "What is on screen?"}],
             images=[(png_bytes, "image/png")])

    # What will actually run, after the local-model gate (see resolve_order):
    resolve_order(TaskKind.CHAT)          # e.g. ["gemini", "claude", "ollama"]
    describe_routing()                    # multi-line summary for the settings panel

POLICY
    DEFAULT_POLICY maps each TaskKind to an ordered list of backend names.
    Pass a custom `policy` dict to complete() to override it per call (the
    ROUTING settings section builds one from user choices this way).
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Generator, Optional


class TaskKind(Enum):
    VOICE_TURN  = "voice_turn"
    INTENT      = "intent"
    CODE_GEN    = "code_gen"
    CODE_REVIEW = "code_review"
    SUMMARIZE   = "summarize"
    VISION      = "vision"
    CHAT        = "chat"


# Ordered preference per task kind. "ollama" is a *fallback* everywhere
# except INTENT: the routing eval (tests/eval/baseline.json) scored the
# shipped default local model, qwen3:1.7b, at 19% tool-routing accuracy, and
# a model that small in front of Gemini made study notes, file summaries and
# quiz generation measurably worse. resolve_order() below additionally
# demotes ollama out of first place at call time unless the configured local
# model is at least OLLAMA_MIN_PARAMS_B billion parameters — so a saved user
# policy that still says "ollama, gemini" keeps working, just in a sane order.
#
# "gemini_lite" is Gemini Flash-Lite: cheapest, fastest, weakest. It sits
# behind Flash and Claude as the text fallback for CHAT/SUMMARIZE (Flash
# returns 503 "high demand" in spikes, and Flash-Lite answering in 2s beats a
# 1.7B local model answering in 7s) and is the first choice only for callers
# that run on a timer (study_mode's background capture loop) via `policy`.
DEFAULT_POLICY: dict[TaskKind, list[str]] = {
    TaskKind.CODE_GEN:    ["claude", "gemini", "ollama"],
    TaskKind.CODE_REVIEW: ["claude", "gemini", "ollama"],
    TaskKind.INTENT:      ["ollama", "gemini", "claude"],
    TaskKind.SUMMARIZE:   ["gemini", "claude", "gemini_lite", "ollama"],
    TaskKind.VISION:      ["gemini", "gemini_lite"],
    TaskKind.VOICE_TURN:  ["gemini"],
    TaskKind.CHAT:        ["gemini", "claude", "gemini_lite", "ollama"],
}

BREAKER_COOLDOWN_S = 60.0
TRANSIENT_RETRY_DELAY_S = 1.0
_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded", "high demand")
GEMINI_MODEL       = "gemini-flash-latest"
GEMINI_LITE_MODEL  = "gemini-flash-lite-latest"

# Smallest local model allowed to sit in FIRST place of a policy order.
OLLAMA_MIN_PARAMS_B = 7.0
_PARAM_SIZE_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")

_lock           = threading.Lock()
_breaker_until: dict[str, float] = {}          # backend name -> monotonic time it's skipped until
_logged_trips:  set[str]         = set()       # backend names already logged for the CURRENT trip


def _base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _get_api_config() -> dict:
    try:
        return json.loads((_base_dir() / "config" / "api_keys.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _breaker_open(name: str) -> bool:
    with _lock:
        until = _breaker_until.get(name)
        if until is None:
            return False
        if time.monotonic() >= until:
            del _breaker_until[name]
            _logged_trips.discard(name)
            return False
        return True


def _is_transient(error: Exception) -> bool:
    text = str(error)
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _trip_breaker(kind: TaskKind, name: str, error: Exception) -> None:
    with _lock:
        _breaker_until[name] = time.monotonic() + BREAKER_COOLDOWN_S
        already_logged = name in _logged_trips
        _logged_trips.add(name)
    if not already_logged:
        # ASCII only: this can print on a cp125x console mid-tool-call.
        msg = str(error).encode("ascii", "replace").decode("ascii")[:200]
        print(f"[Router] {kind.value}: {name} failed, skipping it for {BREAKER_COOLDOWN_S:.0f}s: {msg}")


def reset_breakers() -> None:
    """Test/ops hook: clear all circuit-breaker state immediately."""
    with _lock:
        _breaker_until.clear()
        _logged_trips.clear()


def _is_configured(name: str) -> bool:
    if name == "ollama":
        return True   # no auth needed; reachability is checked at call time by the adapter itself
    if name == "claude":
        from core.claude_bridge import get_claude_config, get_claude_settings
        api_key, _, _ = get_claude_settings(get_claude_config())
        return bool(api_key)
    if name in ("gemini", "gemini_lite"):
        return bool(_get_api_config().get("gemini_api_key"))
    return False


# ── Local-model capability gate ───────────────────────────────────────────

def model_param_billions(model_name: str) -> float | None:
    """Parse the parameter count out of an Ollama-style model tag:
    'qwen3:1.7b' -> 1.7, 'llama3.1:8b-instruct-q4' -> 8.0,
    'mistral:7b' -> 7.0. None when the tag carries no size ('mistral',
    'gemma:latest') — treated as *unknown*, which the gate counts as small,
    since the default pulls for size-less tags are the small variants."""
    if not model_name:
        return None
    m = _PARAM_SIZE_RE.search(model_name)
    return float(m.group(1)) if m else None


def ollama_capability() -> tuple[bool, str, str]:
    """(capable_of_first_place, model_name, reason). Reads the configured
    local model name only — no network, so it is safe to call on every
    complete()."""
    try:
        from core import llm_client
        _url, model = llm_client.get_llm_settings()
    except Exception as e:
        return False, "", f"local engine settings unreadable ({e})"
    size = model_param_billions(model)
    if size is None:
        return False, model, f"'{model}' has no size in its tag; assumed small"
    if size < OLLAMA_MIN_PARAMS_B:
        return False, model, f"'{model}' is {size:g}B, below the {OLLAMA_MIN_PARAMS_B:g}B floor"
    return True, model, f"'{model}' is {size:g}B"


def resolve_order(kind: TaskKind, policy: dict[TaskKind, list[str]] | None = None) -> list[str]:
    """The backend order complete() will actually try for `kind`, after the
    local-model gate: if the local model is too small, "ollama" is moved to
    the END of the order whenever any other backend in it is configured —
    still a fallback, never the first answer. Everything else is preserved."""
    order = list((policy or DEFAULT_POLICY).get(kind, ["gemini", "claude", "ollama"]))
    if "ollama" in order:
        others = [n for n in order if n != "ollama"]
        # "Leads the order" means "would answer first", so an unconfigured
        # backend ahead of it doesn't count: the saved policy
        # "claude, ollama, gemini" with no Claude key was quietly sending
        # every code task to qwen3:1.7b.
        if others and any(_is_configured(n) for n in others):
            capable, _model, _reason = ollama_capability()
            if not capable:
                order = others + ["ollama"]
    return order


def describe_routing(policy: dict[TaskKind, list[str]] | None = None) -> str:
    """Human-readable 'what would actually run' summary for the settings
    panel: one line per task kind with the resolved order, plus which
    backends are unconfigured or currently in breaker cooldown."""
    lines = []
    for kind in TaskKind:
        raw = list((policy or DEFAULT_POLICY).get(kind, []))
        resolved = resolve_order(kind, policy)
        live = [n for n in resolved if _is_configured(n) and not _breaker_open(n)]
        first = live[0] if live else "none available"
        note = "  (ollama demoted: local model too small)" if resolved != raw else ""
        lines.append(f"{kind.value:12s} -> {first:11s} order: {', '.join(resolved)}{note}")
    capable, _model, reason = ollama_capability()
    lines.append("")
    lines.append("local model: " + reason
                 + ("" if capable else f" - needs >= {OLLAMA_MIN_PARAMS_B:g}B to lead an order"))
    unconf = [n for n in _ADAPTERS if not _is_configured(n)]
    if unconf:
        lines.append("not configured: " + ", ".join(unconf))
    cooling = [n for n in _ADAPTERS if _breaker_open(n)]
    if cooling:
        lines.append("in cooldown after a failure: " + ", ".join(cooling))
    return "\n".join(lines)


# ── Adapters ──────────────────────────────────────────────────────────────
# Each adapter takes (messages, tools, images, timeout) — the same flat
# {"role", "content"} message shape core/llm_client.py's call_llm() and
# core/claude_bridge.py's call_claude() already use — and returns
# {"content", "tool_calls", "usage", "backend"}. Kept in a plain dict (not
# hardcoded into complete()) so tests can substitute fakes without touching
# real network/config, per tests/test_backend_router.py.

def _ollama_reachable(url: str, timeout: float = 1.5) -> bool:
    """Plain reachability ping. Deliberately NOT llm_client.ensure_ollama_running():
    that helper auto-launches `ollama serve` and then waits for it, which is
    the right thing for Local Mode's voice loop but the wrong side effect for
    a one-shot fallback call inside a study-notes or summarize request."""
    try:
        import requests
        return requests.get(f"{url}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def _call_ollama(messages: list, tools: list | None, images: list | None, timeout: int) -> dict:
    if images:
        raise RuntimeError("ollama backend does not accept images")
    from core import llm_client
    url, _model = llm_client.get_llm_settings()
    if not _ollama_reachable(url):
        raise RuntimeError(f"Ollama unreachable at {url}")
    resp = llm_client.call_llm(messages, tools, timeout=timeout)
    return {"content": resp.get("content", ""), "tool_calls": resp.get("tool_calls") or [],
            "usage": {}, "backend": "ollama"}


def _call_claude(messages: list, tools: list | None, images: list | None, timeout: int) -> dict:
    if images:
        raise RuntimeError("claude adapter does not accept images")
    from core import claude_bridge

    system = None
    msgs = messages
    if msgs and msgs[0].get("role") == "system":
        system, msgs = msgs[0].get("content"), msgs[1:]

    anthropic_tools = None
    if tools:
        from core.tool_schema import gemini_tools_to_anthropic
        # Gemini-shaped declarations (the convention everywhere else in this
        # codebase) have a bare "parameters" key; Anthropic tools already in
        # {"name", "input_schema"} shape pass straight through unconverted.
        anthropic_tools = (gemini_tools_to_anthropic(tools) if tools[0].get("parameters") is not None
                           else tools)

    resp = claude_bridge.call_claude(msgs, anthropic_tools, system=system, timeout=timeout)
    return {"content": resp.get("content", ""), "tool_calls": resp.get("tool_calls") or [],
            "usage": {}, "backend": "claude"}


def _messages_to_prompt(messages: list) -> str:
    """Gemini's one-shot generate_content() takes a prompt, not a chat-message
    list — flatten system/user/assistant turns into one block. Only the
    non-streaming, non-Live path uses this (VOICE_TURN stays on Live)."""
    # A single user message (optionally after a system block) is the common
    # one-shot case — study notes, quiz generation, screen reads — and must
    # arrive verbatim: a "user: " prefix in front of a "return ONLY JSON"
    # instruction is noise the model sometimes echoes back.
    chat = [m for m in messages if m.get("role", "user") != "system"]
    if len(chat) <= 1:
        parts = [str(m.get("content", "")) for m in messages]
        return "\n\n".join(p for p in parts if p)
    parts = []
    for m in messages:
        role, content = m.get("role", "user"), m.get("content", "")
        if role == "system":
            parts.append(str(content))
        else:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _gemini_generate(model: str, backend: str, messages: list, images: list | None) -> dict:
    api_key = _get_api_config().get("gemini_api_key")
    if not api_key:
        raise RuntimeError("no gemini_api_key configured")
    from google import genai
    from google.genai import types as gtypes

    client  = genai.Client(api_key=api_key)
    # Image parts first, prompt last — the order the one-shot vision helpers
    # in actions/screen_processor.py always used.
    content: list = [gtypes.Part.from_bytes(data=img_bytes, mime_type=mime)
                     for img_bytes, mime in (images or [])]
    content.append(_messages_to_prompt(messages))
    resp = client.models.generate_content(model=model, contents=content)
    usage = {}
    meta = getattr(resp, "usage_metadata", None)
    if meta is not None:
        usage = {"tokens_in":  getattr(meta, "prompt_token_count", None),
                 "tokens_out": getattr(meta, "candidates_token_count", None)}
    return {"content": (getattr(resp, "text", None) or "").strip(), "tool_calls": [],
            "usage": usage, "backend": backend}


def _call_gemini(messages: list, tools: list | None, images: list | None, timeout: int) -> dict:
    return _gemini_generate(GEMINI_MODEL, "gemini", messages, images)


def _call_gemini_lite(messages: list, tools: list | None, images: list | None, timeout: int) -> dict:
    return _gemini_generate(GEMINI_LITE_MODEL, "gemini_lite", messages, images)


_ADAPTERS: dict[str, Callable[[list, Optional[list], Optional[list], int], dict]] = {
    "ollama":      _call_ollama,
    "claude":      _call_claude,
    "gemini":      _call_gemini,
    "gemini_lite": _call_gemini_lite,
}


# ── Streaming adapters ────────────────────────────────────────────────────
# Same (messages, images, timeout) contract, but a generator of events:
#   {"delta": str}                        text as it arrives
#   {"done": {"backend": str, "usage": {}}}  exactly once, at the end
# Used by core/think.py so speech can start on the first sentence instead of
# waiting for the whole answer. A backend without a native stream (Claude's
# bridge is non-streaming today) yields its full reply as one delta.

def _gemini_stream(model: str, backend: str, messages: list, images: list | None) -> Generator[dict, None, None]:
    api_key = _get_api_config().get("gemini_api_key")
    if not api_key:
        raise RuntimeError("no gemini_api_key configured")
    from google import genai
    from google.genai import types as gtypes

    client  = genai.Client(api_key=api_key)
    content: list = [gtypes.Part.from_bytes(data=img_bytes, mime_type=mime)
                     for img_bytes, mime in (images or [])]
    content.append(_messages_to_prompt(messages))
    usage: dict = {}
    for chunk in client.models.generate_content_stream(model=model, contents=content):
        text = getattr(chunk, "text", None)
        if text:
            yield {"delta": text}
        meta = getattr(chunk, "usage_metadata", None)
        if meta is not None:
            usage = {"tokens_in":  getattr(meta, "prompt_token_count", None),
                     "tokens_out": getattr(meta, "candidates_token_count", None)}
    yield {"done": {"backend": backend, "usage": usage}}


def _stream_gemini(messages, images, timeout):
    yield from _gemini_stream(GEMINI_MODEL, "gemini", messages, images)


def _stream_gemini_lite(messages, images, timeout):
    yield from _gemini_stream(GEMINI_LITE_MODEL, "gemini_lite", messages, images)


def _stream_ollama(messages, images, timeout):
    if images:
        raise RuntimeError("ollama backend does not accept images")
    from core import llm_client
    url, _model = llm_client.get_llm_settings()
    if not _ollama_reachable(url):
        raise RuntimeError(f"Ollama unreachable at {url}")
    usage: dict = {}
    for ev in llm_client.stream_llm(messages, None, timeout=timeout):
        if "delta" in ev:
            yield {"delta": ev["delta"]}
        elif "done" in ev:
            usage = ev["done"] or {}
    yield {"done": {"backend": "ollama", "usage": usage}}


def _stream_claude(messages, images, timeout):
    result = _call_claude(messages, None, images, timeout)
    if result.get("content"):
        yield {"delta": result["content"]}
    yield {"done": {"backend": "claude", "usage": result.get("usage") or {}}}


_STREAMERS: dict[str, Callable[[list, Optional[list], int], Generator[dict, None, None]]] = {
    "ollama":      _stream_ollama,
    "claude":      _stream_claude,
    "gemini":      _stream_gemini,
    "gemini_lite": _stream_gemini_lite,
}


def complete_stream(
    kind:     TaskKind,
    messages: list,
    images:   list | None = None,
    timeout:  int = 60,
    policy:   dict[TaskKind, list[str]] | None = None,
) -> Generator[dict, None, None]:
    """Streaming twin of complete(): same policy order, breaker and
    transient retry, but yields {"delta"} events and one final {"done"}.
    Failover only happens BEFORE the first delta — once text has been
    handed to the caller (and possibly spoken) a mid-stream failure is
    raised rather than silently restarted on another backend."""
    order = resolve_order(kind, policy)
    last_error: Exception | None = None
    attempted: list[str] = []

    for name in order:
        if _breaker_open(name) or not _is_configured(name):
            continue
        streamer = _STREAMERS.get(name)
        if streamer is None:
            continue
        attempted.append(name)
        for attempt in (0, 1):
            started = False
            try:
                for ev in streamer(messages, images, timeout):
                    if "delta" in ev and ev["delta"]:
                        started = True
                    yield ev
                return
            except Exception as e:
                if started:
                    raise
                if attempt == 0 and _is_transient(e):
                    time.sleep(TRANSIENT_RETRY_DELAY_S)
                    continue
                last_error = e
                _trip_breaker(kind, name, e)
                break

    if not attempted:
        raise RuntimeError(f"No backend available for {kind.value} (policy: {order}, all "
                            f"skipped — unconfigured or in cooldown)")
    raise RuntimeError(f"All backends failed for {kind.value} (tried {attempted}): {last_error}")


def complete(
    kind:     TaskKind,
    messages: list,
    tools:    list | None = None,
    images:   list | None = None,
    timeout:  int = 60,
    policy:   dict[TaskKind, list[str]] | None = None,
) -> dict:
    """Tries each backend in the policy order for `kind` until one succeeds.
    A backend is skipped if its circuit breaker is open or it isn't
    configured. Raises RuntimeError only if every backend in the order was
    skipped or failed."""
    order = resolve_order(kind, policy)
    last_error: Exception | None = None
    attempted: list[str] = []

    for name in order:
        if _breaker_open(name) or not _is_configured(name):
            continue
        adapter = _ADAPTERS.get(name)
        if adapter is None:
            continue
        attempted.append(name)
        try:
            return adapter(messages, tools, images, timeout)
        except Exception as e:
            # One quick retry on a transient capacity error before tripping
            # the 60s breaker: Gemini Flash returns 503 "high demand" in
            # short spikes, and a single 503 used to route the next minute of
            # study notes / quizzes to a weaker fallback.
            if _is_transient(e):
                time.sleep(TRANSIENT_RETRY_DELAY_S)
                try:
                    return adapter(messages, tools, images, timeout)
                except Exception as e2:
                    e = e2
            last_error = e
            _trip_breaker(kind, name, e)

    if not attempted:
        raise RuntimeError(f"No backend available for {kind.value} (policy: {order}, all "
                            f"skipped — unconfigured or in cooldown)")
    raise RuntimeError(f"All backends failed for {kind.value} (tried {attempted}): {last_error}")


@dataclass
class _TextResult:
    text: str


class _RoutedTextModel:
    """`.generate_content(prompt).text` adapter over complete(), for callers
    using the small google-genai-shaped wrapper convention (see
    core.claude_bridge.ClaudeTextModel) instead of the chat-messages shape."""

    def __init__(self, kind: TaskKind, timeout: int = 120,
                 policy: dict[TaskKind, list[str]] | None = None):
        self.kind    = kind
        self.timeout = timeout
        self.policy  = policy

    def generate_content(self, contents) -> _TextResult:
        prompt = contents if isinstance(contents, str) else str(contents)
        result = complete(self.kind, [{"role": "user", "content": prompt}],
                           timeout=self.timeout, policy=self.policy)
        return _TextResult(result["content"])


def get_text_model(kind: TaskKind, timeout: int = 120,
                    policy: dict[TaskKind, list[str]] | None = None) -> _RoutedTextModel:
    return _RoutedTextModel(kind, timeout, policy)


def load_policy_from_config(raw: dict[str, str]) -> dict[TaskKind, list[str]]:
    """Builds a policy dict from the ROUTING settings section's saved values
    — `raw` is {task_kind.value: "backend1, backend2, ..."} (see main.py's
    _routing_settings_section). Blank or missing entries fall back to
    DEFAULT_POLICY for that kind; unrecognised backend names are dropped
    rather than raising, since a saved value should never crash a real call."""
    policy: dict[TaskKind, list[str]] = {}
    for kind in TaskKind:
        raw_value = (raw or {}).get(kind.value, "").strip()
        if not raw_value:
            policy[kind] = DEFAULT_POLICY[kind]
            continue
        names = [n.strip() for n in raw_value.split(",") if n.strip() in _ADAPTERS]
        policy[kind] = names or DEFAULT_POLICY[kind]
    return policy
