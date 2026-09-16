"""
core/think.py — JARVIS's reasoning core.

WHY THIS EXISTS
    The Live session runs on a speech model tuned for latency, and until now
    it did all the thinking too: explanations, comparisons, planning, maths.
    It sounds instant and reasons shallowly. Iron Man's JARVIS does both,
    and the way to get both is to split the job: the Live model stays the
    ears, mouth and reflexes (open the app, turn it down, "yes sir"), and
    anything that needs thought is handed to a strong text model with the
    context the Live model doesn't carry — the durable turn history in
    memory/context_store.db, the full memory core, the last few turns, and
    (on request) a fresh screenshot. The answer streams back sentence by
    sentence so speech can start before the last sentence exists.

    Everything the Live path needs is already in the repo — the per-task
    router (core/backend_router.py), the context assembler
    (core/context_manager.py, previously used only by Local Mode), the memory
    block (memory/memory_manager.py) and the sentence chunker
    (core/sentence_chunker.py). This module only connects them; main.py
    owns the `think` tool declaration and the delivery into the session.

USAGE
    from core import think
    result = think.run("compare SM-2 and FSRS for my flashcards",
                       session_log=jarvis._session_log,
                       include_screen=False,
                       on_sentence=lambda s, i: ...)   # called as each sentence completes
    result.text / result.backend / result.sentences / result.elapsed_ms
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.backend_router import TaskKind

# The reasoning core is a *writer for a voice*, not a chatbot. Every rule here
# is about what a spoken answer needs: conclusion first, short, no markup.
REASONING_SYSTEM = (
    "You are JARVIS's reasoning core. The Live voice assistant will speak your "
    "answer aloud, so write for the ear:\n"
    "- Lead with the conclusion, then the one or two reasons that matter.\n"
    "- At most 120 words unless the request explicitly asks for detail.\n"
    "- No markdown, no headings, no bullet symbols; if steps must be performed, "
    "number them in plain sentences.\n"
    "- Answer in the language of the user's request.\n"
    "- Use the context blocks only when relevant; never recite them.\n"
    "- If something in the context contradicts the request, say so briefly.\n"
    "- Be precise with numbers and names; if you are not sure, say what you are sure of."
)

DEFAULT_SESSION_TURNS = 6
MAX_CONTEXT_CHARS = 3_500
MAX_QUERY_CHARS = 2_000


@dataclass
class ThinkResult:
    text: str
    backend: str
    sentences: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    usage: dict = field(default_factory=dict)
    included_screen: bool = False
    context_chars: int = 0


def _memory_block() -> str:
    try:
        from memory.memory_manager import load_memory, format_memory_for_prompt
        return format_memory_for_prompt(load_memory()) or ""
    except Exception:
        return ""


def _context_block(query: str, session_log: Optional[list[str]]) -> str:
    try:
        from core import context_manager
        bundle = context_manager.build_context(query, session_log=session_log,
                                               session_turns=DEFAULT_SESSION_TURNS,
                                               max_chars=MAX_CONTEXT_CHARS)
        return bundle.to_prompt() or ""
    except Exception:
        return ""


def _recent_turns(session_log: Optional[list[str]], n: int = DEFAULT_SESSION_TURNS) -> str:
    turns = list(session_log or [])[-n:]
    return "\n".join(turns)


def build_messages(query: str, session_log: Optional[list[str]] = None,
                   memory_block: Optional[str] = None, context_block: Optional[str] = None,
                   screen_attached: bool = False, now: Optional[str] = None) -> list[dict]:
    """Assemble the system + user messages. Pure: every block can be passed
    in, which is what the tests do; run() fills them from the live stores."""
    memory_block  = _memory_block() if memory_block is None else memory_block
    context_block = _context_block(query, session_log) if context_block is None else context_block

    system_parts = [REASONING_SYSTEM]
    if now:
        system_parts.append(f"[NOW] {now}")
    if memory_block.strip():
        system_parts.append(memory_block.strip())
    if context_block.strip():
        system_parts.append(context_block.strip())
    recent = _recent_turns(session_log)
    if recent and "[RECENT CONVERSATION]" not in context_block:
        system_parts.append("[RECENT TURNS]\n" + recent)
    if screen_attached:
        system_parts.append("[SCREEN] A screenshot of the user's screen is attached; "
                            "use it if the request refers to what they are looking at.")

    q = (query or "").strip()[:MAX_QUERY_CHARS]
    return [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "user", "content": q},
    ]


def run(query: str,
        session_log: Optional[list[str]] = None,
        include_screen: bool = False,
        on_sentence: Optional[Callable[[str, int], None]] = None,
        policy: Optional[dict] = None,
        timeout: int = 60,
        capture: Optional[Callable[[], tuple[bytes, str]]] = None,
        stream: Optional[Callable] = None) -> ThinkResult:
    """Reason about `query` on the best configured text backend and stream
    sentences to `on_sentence(sentence, index)` as they complete.

    include_screen=True attaches a fresh screenshot — which forces a Gemini
    backend, since it is the only one that takes images here. `capture` and
    `stream` are injection points for tests (default: the real screen grab
    and core.backend_router.complete_stream)."""
    from core.sentence_chunker import SentenceChunker

    t0 = time.monotonic()
    images = None
    attached = False
    if include_screen:
        try:
            cap = capture
            if cap is None:
                from actions.screen_processor import _capture_screen
                cap = _capture_screen
            img, mime = cap()
            images = [(img, mime)]
            attached = True
        except Exception as e:
            print(f"[Think] screen capture skipped: {e}")

    from datetime import datetime
    now = datetime.now().strftime("%A, %B %d, %Y %H:%M")
    messages = build_messages(query, session_log=session_log, screen_attached=attached, now=now)
    context_chars = len(messages[0]["content"])

    if images:
        # Images can only go to Gemini; don't let a CHAT policy that leads
        # with Claude/Ollama fail twice before getting there.
        wanted = [n for n in (policy or {}).get(TaskKind.CHAT, []) if n.startswith("gemini")]
        policy = {TaskKind.CHAT: wanted or ["gemini", "gemini_lite"]}

    streamer = stream
    if streamer is None:
        from core.backend_router import complete_stream
        streamer = complete_stream

    chunker = SentenceChunker()
    sentences: list[str] = []
    parts: list[str] = []
    backend = ""
    usage: dict = {}

    def _emit(sentence: str):
        sentences.append(sentence)
        if on_sentence is not None:
            try:
                on_sentence(sentence, len(sentences) - 1)
            except Exception as e:      # a delivery hiccup must not kill the reasoning
                print(f"[Think] on_sentence failed: {e}")

    for ev in streamer(TaskKind.CHAT, messages, images=images, timeout=timeout, policy=policy):
        if "delta" in ev and ev["delta"]:
            parts.append(ev["delta"])
            for sent in chunker.feed(ev["delta"]):
                _emit(sent)
        elif "done" in ev:
            backend = (ev["done"] or {}).get("backend", backend) or backend
            usage = (ev["done"] or {}).get("usage") or {}

    tail = chunker.flush()
    if tail:
        _emit(tail)

    text = "".join(parts).strip()
    return ThinkResult(text=text, backend=backend, sentences=sentences,
                       elapsed_ms=(time.monotonic() - t0) * 1000, usage=usage,
                       included_screen=attached, context_chars=context_chars)
