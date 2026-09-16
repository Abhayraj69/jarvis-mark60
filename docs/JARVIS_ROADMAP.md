# JARVIS Upgrade Roadmap — from "Mark LIII" to Iron Man JARVIS

Generated 2026-09-16 from a code + telemetry audit of this repo. Every claim
below is backed by a file/line or a number from `memory/telemetry.db`.

## Status

| Prompt | State | Result |
|---|---|---|
| 1.1 token diet | **done** (branch `feat/phase1-fast-brain`) | static per-turn payload 43,979 → 25,613 chars (tool JSON 31,987 → 21,985; prompt.txt 11,992 → 3,628); history now compressed at 12k → 6k tokens; guarded by `tests/test_tool_budget.py` |
| 1.3 per-task routing | **done** | screen reads / notes / quiz / screen-debug go through `backend_router` (Flash first, Flash-Lite fallback; Flash-Lite only for study_mode's timed loop); sub-7B Ollama models can no longer answer before a configured cloud backend; one retry on 503/429 before the breaker trips; ROUTING settings show the resolved order |
| 3.1 prompt rewrite | **done** | persona-first `core/prompt.txt` with 5 examples and one copy of each rule; identity/address lines in the same register; guarded by `tests/test_prompt_quality.py` |
| everything else | not started | — |

Next: run JARVIS for a day, then run Prompt 0.1 (bench) and compare tokens_in / total_ms against the baseline table below.

## How to use this document

* Each phase has **one or more prompts**. Paste a prompt into Claude Code as a
  fresh message, on a fresh branch (`git checkout -b feat/<phase>`).
* Do the phases **in order**. Phase 1 and Phase 3 are the cheapest and give the
  biggest felt improvement — do them first, in one evening.
* After every phase: `pytest -q`, run JARVIS for 10 turns, then paste the
  output of the *bench* prompt (Phase 0) back to Claude so the next phase is
  measured against the last one, not against a feeling.
* Keep one phase per session. Don't ask for "everything" in one prompt — that
  is how the repo got 17k lines of features that aren't wired to the voice
  path.

---

## The numbers that matter (baseline, 2026-09-16)

| Metric (121 Gemini Live turns) | Now | Target |
|---|---|---|
| Tokens sent per turn (p50 / p90) | 26,683 / 50,292 | < 8,000 / < 14,000 |
| Total turn time (p50 / p90) | 8.8 s / 24 s | < 3 s / < 8 s |
| Time to first audio (p50 / p90) | 0.6 s / 3.3 s | < 0.5 s / < 1.2 s |
| fast-intent hits | 0 | > 20% of action turns |
| Routing eval accuracy (fast_intent / ollama) | 32% / 19% | > 90% on the primary brain |
| Quiz answers persisted | 0 | 100% of graded answers |
| study_quiz latency | 8–15 s | < 4 s |

Where the problems live:

| Symptom | Cause | Where |
|---|---|---|
| Wrong tool, wrong language, "done" when it isn't | 12 KB contradictory prompt + 25 KB of tool declarations + uncompressed history | `core/prompt.txt`, `main.py:_build_config` |
| Shallow answers | One latency-tuned speech model does all reasoning; Claude backend disabled; qwen3:1.7b first in routing | `config/api_keys.json` plugin_config, `core/backend_router.py` |
| Forgets what you talked about | context_manager / sentiment / semantic recall only wired into Local Mode | `main.py:~3032` vs the Live receive loop |
| Slow multi-tool turns | Tool calls awaited sequentially inside the receive loop | `main.py:~2070` |
| Study mode dumb & forgetful | flash-lite for notes/quiz; no deck; grading + logging delegated to the voice model | `actions/screen_processor.py:93`, `actions/study_*.py` |

---

## Phase 0 — Measure before touching anything (30 min)

**Prompt 0.1 — bench command**

> Add a `python -m core.bench` script that prints, from memory/telemetry.db:
> p50/p90/max of tokens_in, total_ms and time_to_first_audio_ms grouped by
> backend; fast_intent hit rate; the 10 slowest tool spans; and the count of
> quiz_answers in memory/study_history.db. Then extend
> tests/eval/run_routing_eval.py so it can run the routing corpus against the
> Gemini backend (gemini-flash-latest, tools = the real declarations from
> ActionRegistry) and update tests/eval/baseline.json with a `gemini` entry.
> Do not change any runtime behaviour. Run both and paste the output in your
> final message.

---

## Phase 1 — Make it fast (the single biggest felt win)

Why: 27k tokens per turn and 25 tools is why it picks wrong tools, answers in
the wrong language, and takes 9 s. Halving tokens roughly halves model latency
and measurably improves tool selection on small models.

**Prompt 1.1 — token diet**

> Goal: cut the per-turn token count of the Gemini Live session from ~27k
> (p50, see memory/telemetry.db) to under 8k without losing capability.
> 1. In main.py `_build_config`, set `ContextWindowCompressionConfig(
>    trigger_tokens=12000, sliding_window=SlidingWindow(target_tokens=6000))`
>    so history is compressed continuously instead of only near the limit.
> 2. Measure the JSON size of every TOOL declaration (actions/*.py, plugins,
>    TOOL_DECLARATIONS in main.py) and rewrite descriptions to be at most 200
>    characters each, with parameter descriptions at most 80 characters. Keep
>    every enum value. The behavioural instructions that currently live in
>    tool descriptions ("say one sentence out loud", "ask one at a time")
>    must move into core/prompt.txt ONCE, not be duplicated in both places.
> 3. In core/prompt.txt remove every rule that duplicates a tool description
>    (the whole TOOL ROUTING block except genuinely cross-tool policy).
> 4. Add a test that fails if the total tool-declaration JSON exceeds 10,000
>    characters or prompt.txt exceeds 5,000 characters.
> Report before/after character counts for each.

**Prompt 1.2 — don't block the ear while the hands work**

> In main.py's receive loop (~line 2070, `if response.tool_call:`) tool calls
> are executed sequentially and the loop awaits each before reading the next
> server message. Refactor so that (a) all function_calls in one tool_call
> message run concurrently via asyncio.gather, (b) the tool execution and
> send_tool_response happen in a background task so the receive loop keeps
> draining audio/transcripts, and (c) a per-tool timeout (default 20 s,
> configurable in the TOOL dict as "timeout") returns a clear "still working,
> I'll tell you when it's done" result instead of hanging the turn. Keep
> confirmation-gated tools (restart/shutdown/wifi) and screen_process's
> vision-busy logic exactly as they are. Add tests with a fake session object.

**Prompt 1.3 — right model for each sub-task**

> actions/screen_processor.py hardcodes gemini-flash-lite-latest for
> _vision_query and _text_query, and web_search/file_processor/code_helper
> hardcode gemini-flash-latest. Route all of these through
> core/backend_router.complete()/get_text_model() with a TaskKind (VISION for
> screenshots, SUMMARIZE for notes, CODE_GEN/CODE_REVIEW for code, CHAT for
> quiz generation). Change DEFAULT_POLICY so `ollama` is never first unless
> the configured model name indicates 7B or larger; qwen3:1.7b scores 19% on
> the routing eval and must not be in front of Gemini. Keep flash-lite ONLY
> for the study_mode background loop (cost). Add a settings-panel note
> showing which backend each TaskKind resolves to right now. Tests for the
> policy resolution.

**Prompt 1.4 — fast-intent for voice**

> core/fast_intent.py only runs on typed text, so it fired 0 times in 121
> voice turns. In the Live path, input_audio_transcription arrives as text
> before the model's tool call for short commands. Add a race: when a
> complete input transcription matches a fast-intent pattern, execute it
> immediately and mark the turn; if the model then issues the same tool call
> within 3 s, answer it with the cached result instead of re-running. Never
> apply to questions or to confirmation-gated actions. Log hits to telemetry
> (fast_intent=1). Add 15 new patterns covering the most common actions from
> memory/telemetry.db tool_spans (computer_control focus/type, open_app,
> youtube play/pause; screen_process is excluded).

---

## Phase 2 — Give JARVIS a real brain (two-tier architecture)

Why: Iron Man's JARVIS *sounds* instant but *reasons* deeply. You get that by
splitting the job: the Live model is the ears and mouth and handles instant
actions; a strong model (Claude Sonnet 5 / Opus 5, or Gemini Pro) does the
thinking when a request needs it, and its answer is streamed back for the
Live model to speak. The plumbing (core/backend_router.py,
core/claude_bridge.py, core/context_manager.py) already exists — it's just
not connected to the voice path.

**Prompt 2.1 — the `think` tool**

> Add a core tool `think` (in main.py TOOL_DECLARATIONS, or a new
> actions/think.py) that the Live model calls for anything requiring
> reasoning, multi-step planning, explanation, comparison, maths, or an
> answer longer than two sentences. It:
> 1. Builds a ContextBundle via core/context_manager.build_context(query,
>    session_log=self._session_log) — this is currently only used in Local
>    Mode (main.py ~3032) and must now also serve the Gemini Live path.
> 2. Adds the memory core block, the last 6 turns, and (if
>    include_screen=true) a fresh screenshot.
> 3. Calls backend_router.complete(TaskKind.CHAT, ...) with a system prompt
>    that says: "You are JARVIS's reasoning core. Answer for a voice
>    assistant: lead with the conclusion, at most 120 words unless asked for
>    detail, no markdown, no lists unless enumerating steps to perform."
> 4. Streams the reply sentence-by-sentence (reuse core/sentence_chunker.py)
>    back into the Live session as text so speech starts before the full
>    answer exists.
> Update core/prompt.txt with a short rule: "Instant actions and small talk:
> answer yourself. Anything needing thought: call think and relay it in your
> own voice." Enable the claude_engine plugin config by default when an
> ANTHROPIC_API_KEY env var is present. Telemetry: record which backend
> served each think call.

**Prompt 2.2 — never say "done" when it isn't**

> Add a result contract. Every tool handler result is wrapped as
> {"ok": bool, "summary": str, "detail": str} by _execute_tool (string
> results are coerced: startswith("Tool '"+name+"' failed") or "Could not"/
> "Unknown" means ok=false). Put ONE rule in prompt.txt: "If a tool returns
> ok=false you must say what failed and what you'll try instead — never
> report success." Add a post-turn check in telemetry: if any tool in the
> turn had ok=false and the output transcript contains none of
> {"couldn't","failed","unable","didn't","problem","error"} (small list,
> extendable per language), log a `false_success` flag so we can count
> hallucinated successes. Tests.

**Prompt 2.3 — memory that is actually used**

> memory/context_store.db has ~960 turns of history and long_term.json has
> facts, but the Live session only gets the small memory core block. (a) On
> every user turn in the Live path, run context_manager.semantic_search on
> the input transcription in a background task; if the top hit scores above
> a threshold and is older than the current session, inject a single short
> text turn "[CONTEXT] Related past exchange (date): …" before the model
> answers — cap at 300 characters, at most one injection per 3 turns.
> (b) Add an end-of-session "reflection" step: ask the reasoning backend to
> extract new durable facts/preferences from the session log and save them
> via update_memory with a confidence field; show them in the Memory panel
> flagged "learned automatically" with one-click delete. (c) Replace the
> hashing-trick embed() in context_manager with real embeddings when
> available (google-genai text-embedding, fallback to current hashing) —
> keep the interface identical. Tests for injection cadence and fallback.

---

## Phase 3 — Personality: sound like JARVIS, not like a chatbot

Why: the entire persona today is one line ("Act like Jarvis from Iron Man")
buried among contradictory rules. Personality is mostly *what you refuse to
say*: no filler, no "Certainly!", no apologising twice, no explaining what
you're about to do. Dry, understated, anticipatory, loyal, and brief.

**Prompt 3.1 — rewrite core/prompt.txt from scratch**

> Rewrite core/prompt.txt from scratch (keep it under 4,500 characters). Structure:
> 1. PERSONA (about 600 chars): JARVIS from the Iron Man films. British-butler
>    register: calm, precise, dry understatement, wit used sparingly and
>    never at the user's expense, quietly proud of the user's work. Addresses
>    the user as "sir" (or the configured name) at most once per reply. Leads
>    with the answer. Never uses filler ("Certainly", "Of course", "Great
>    question", "I'd be happy to"). Never narrates what it is about to do
>    except the single acknowledgment sentence for slow tasks. Never
>    apologises more than once. When something fails, states it plainly and
>    proposes the next move. Offers exactly one relevant next step when
>    useful, phrased as a question ("Shall I…?").
> 2. FIVE short example exchanges (user -> JARVIS) showing: an instant action,
>    a failure, a slow task with acknowledgment, a witty-but-brief reply, and
>    a study-mode turn.
> 3. NON-NEGOTIABLE RULES only (language of last user message; confirmation
>    gate; ok=false contract; shutdown_jarvis; study tutor one-at-a-time).
>    Remove every duplicated, contradictory, shouting or misspelled rule from
>    the current file. Remove all tool-routing lines that the tool
>    descriptions already cover.
> Then update memory_manager.format_memory_for_prompt's identity lines to
> match the new register. Write tests/test_prompt_quality.py asserting
> length, no more than 3 ALL-CAPS "CRITICAL"/"ALWAYS"/"NEVER", and no
> duplicate sentences.

**Prompt 3.2 — voice and timing**

> Two small things that make speech feel like JARVIS: (a) barge-in — when
> the user starts talking while JARVIS is speaking, flush the audio queue
> within 100 ms (check _play_audio and the interrupted flag in telemetry;
> 13/121 turns were interrupted, measure how long the tail lasts today).
> (b) The instant-acknowledgment rule fires for every slow task even when the
> tool finishes in under 1.5 s, which sounds chatty. Add a tool-level hint
> "expected_ms" to TOOL dicts; only tools with expected_ms > 2000 get the
> acknowledgment sentence, and the rule in prompt.txt references that.

---

## Phase 4 — Study Mode 2.0 (make it a real tutor)

What's wrong today: notes and quiz use the weakest model (flash-lite); the
continuous mode appends screenshot-notes to a text file with only exact-match
dedupe; grading and progress logging are delegated to the voice model, which
forgot to log every answer of your last biology quiz (0 rows in
study_history.db); there is no deck, no scheduling, no way to say "teach me
this", and it can only read the screen, not the PDF you're actually reading.

Design principle: **the tutor logic lives in Python and a strong text model,
not in the voice model.** The voice model only relays.

**Prompt 4.1 — study sessions and a real deck**

> Build the data model for Study Mode 2.0 in memory/study_store.py (SQLite,
> same pattern as memory/study_history.py, with a migration script):
> tables `study_sessions` (id, topic, source_kind screen|file|url, source
> ref, started_at, ended_at, notes_path), `cards` (id, topic, front, back,
> kind flashcard|cloze|short_answer, source_session, created_at) and
> `reviews` (card_id, ts, grade 0-5, user_answer) plus SM-2 fields on cards
> (ease, interval_days, due_at, reps, lapses). Implement
> schedule(card, grade) per SM-2 and due_cards(topic=None, limit). Migrate
> existing study_history rows into reviews where possible. Full tests.

**Prompt 4.2 — notes that accumulate and cards that generate themselves**

> Rework actions/study_notes.py and study_mode.py: (a) notes are appended to
> ONE markdown file per topic (desktop/JarvisNotes/<topic>.md) with a
> per-section semantic dedupe (normalise + context_manager.embed cosine >
> 0.9 means skip) instead of a new file per capture; (b) every note capture
> also asks the SUMMARIZE backend for 3–8 flashcards and 2 cloze cards in
> strict JSON and stores them in the deck; (c) accept `source_file`
> (PDF/PPTX/DOCX via actions/file_processor) and `source_url` so studying
> doesn't require a screenshot; (d) use TaskKind.SUMMARIZE (strong model)
> for one-shot captures and keep flash-lite only for the timed background
> loop; (e) study mode's background loop skips the vision call entirely if
> the screen didn't change (compare a downscaled grayscale hash). Tests with
> mocked vision.

**Prompt 4.3 — quiz and grading done in Python, not by the voice model**

> Replace the current study_quiz/study_progress flow with a stateful
> `study_tutor` tool: actions start|answer|hint|skip|end. `start` builds a
> queue (due cards first, then new cards from the latest notes, then a
> generated question set from the notes if the deck is thin) and returns
> only the FIRST question. `answer` takes the user's spoken answer, grades
> it with a one-shot call to the reasoning backend using a rubric (returns
> JSON {correct: bool, grade 0-5, feedback <= 25 words}), writes the review
> row and SM-2 schedule itself (so logging can never be forgotten), and
> returns feedback + the next question. The voice model's only job is to
> read the question and pass the answer through. Support modes: flashcards,
> short-answer, cloze, "exam" (timed, no hints, score at end), and
> "teach-back" (user explains the concept; model critiques against the
> notes and lists what was missed). Update prompt.txt to a 3-line rule for
> this tool. Delete study_quiz once tests pass.

**Prompt 4.4 — explanations, plans and the morning briefing**

> Add to study_tutor: action='explain' (concept, depth beginner|exam|deep;
> uses the notes as ground truth and the reasoning backend; at most 150
> words spoken, full version appended to the topic's notes file),
> action='plan' (given an exam date and topics, produce a day-by-day
> revision plan weighted by weak topics from reviews; save as reminders via
> actions/reminder.py), and a Pomodoro: 'focus' starts a 25/5 timer, uses
> the existing proactive channel to say "break" and "back to it", and pauses
> study_mode captures during breaks. Hook the morning briefing so that if
> due_cards() > 0 it says "You have N cards due in <topic>, shall we?" Add
> a Study tab to the dashboard (dashboard/server.py) showing due counts,
> accuracy per topic, and streak. Tests.

---

## Phase 5 — Anticipation (what makes it feel like Iron Man's JARVIS)

**Prompt 5.1 — screen-aware context without being asked**

> JARVIS should know what I'm doing without me telling it. Add a lightweight
> "situational awareness" loop: every 45 s (adaptive via core/adaptive_poll)
> read the foreground window title and process name (pygetwindow/psutil,
> no screenshot, no API call) and keep a rolling `activity` context (app,
> title, minutes in it). Expose it in the memory core block as one line
> ("Currently: Chrome — 'Cellular respiration - slides' for 12 min") and
> feed it to predictive_assistant.log_event. Use it to (a) pick the right
> app for computer_control automatically, (b) set topic_hint for study tools,
> (c) let proactive check-ins say something specific ("You've been on that
> pull request for 40 minutes — want me to summarise the diff?"). Privacy: a
> toggle in settings, off by default; titles never leave the machine except
> as that one line.

**Prompt 5.2 — proactive suggestions in the voice path**

> core/predictive_assistant.py produces suggestions but the Live path only
> shows them on the HUD (_maybe_show_suggestion). Add a rate-limited spoken
> variant: at most one per 10 minutes, only when the user has been silent
> more than 60 s, only if the suggestion's confidence > 0.7, phrased by the
> reasoning backend in JARVIS's register. Log accept/dismiss to
> suggestion_feedback and lower weights on dismiss (the table exists but has
> 0 rows).

---

## Phase 6 — Structure and reliability (so the next 10 features don't slow it down)

**Prompt 6.1 — split main.py**

> main.py is 3,175 lines and ui.py 5,047. Split main.py into
> core/live_session.py (connect/receive/play loops), core/tool_dispatch.py
> (_execute_tool and registries), core/settings_panels.py (the plugin
> settings dicts), and core/briefing.py (morning briefing + proactive), with
> main.py as a thin entry point. No behaviour change; pytest must stay
> green; add smoke tests that import each module and build a config.

**Prompt 6.2 — golden conversation eval**

> Build tests/eval/conversations.jsonl: 40 multi-turn scripts (text) with
> expected tool sequence, expected language, and forbidden phrases (filler
> words, "done" after a failure). A runner executes them against the
> reasoning backend with the real tool declarations and prompt.txt, mocked
> handlers, and reports pass rate. Wire it as an opt-in pytest like
> test_routing_eval.py. Fail the run if pass rate drops below the saved
> baseline.

---

## Suggested order and what to expect

| Week | Do | You should notice |
|---|---|---|
| 1 | Phase 0, 1.1, 1.3, 3.1 | Turns 2–3x faster, right tool far more often, sounds like JARVIS |
| 2 | 1.2, 1.4, 2.1, 2.2 | No more "done" lies; complex questions get real answers; instant actions instant |
| 3 | 4.1–4.3 | Quiz actually remembers, grades fairly, schedules review |
| 4 | 2.3, 4.4, 3.2 | Remembers past conversations, revision plans, briefings with due cards |
| 5+ | 5.x, 6.x | Anticipates; codebase stays maintainable |

## Things to stop doing

* Adding new subsystems (causal graphs, MCP, macros) before the core turn is
  under 3 s. Every one adds tool declarations = tokens = latency = dumber
  tool choice.
* Putting instructions in three places (prompt.txt, TOOL description, code
  comments). One place: behaviour in prompt.txt, contract in TOOL.
* Delegating bookkeeping to the voice model ("call log_result silently").
  If it must happen, Python does it.
* Running qwen3:1.7b in front of anything. Either run a 7B+ local model or
  use it only for fast-intent-style classification with a strict schema.
