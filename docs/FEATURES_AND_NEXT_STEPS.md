# JARVIS (Mark-LV + Mark-LIII upgrades): features and next steps

Snapshot of branch `upgrade-mark-liii` — upstream Mark-LV plus the 22 custom
commits from Mark-LIII — as of 2026-10-01. The roadmap items referenced below
are in [`JARVIS_ROADMAP.md`](JARVIS_ROADMAP.md), where 1.1, 1.3, 2.1, 2.2 and
3.1 are done and everything else is not started.

## What's in the project today

### Voice and presence
- Real-time voice conversation through Gemini Live, in any language
- Local "Hey Jarvis" wake word; "bye Jarvis" puts it on standby; auto-sleep
  after 2 minutes without activity
- Animated 3D face in the HUD with lip-sync, 5 voices, themeable colours
- Push-to-talk, self-echo guard, barge-in (you can talk over it)
- "Arc Sentinel" desktop widget that pops up on "Hey Jarvis" (needs
  `pywebview` and `faster-whisper`, not installed yet)

### Thinking and memory
- `think` tool: hands real analysis to a second model
- Per-task routing across Gemini, Gemini Lite, Claude and Ollama, with
  automatic fallback when a backend fails
- Long-term memory (save / recall), cross-session conversation history,
  session summaries
- Result contract: every tool reports ok/failed, and a claim of success after
  a failure is recorded
- Per-turn timing and token telemetry, with a performance panel

### Computer control (24 tools)
- Apps, files, browser, mouse and keyboard, system settings, desktop
- Screen and webcam vision; background screen watching for a condition
  ("tell me when the download finishes")
- Web, news and price search; weather; flights; YouTube; Steam/Epic updates
- Code writing and fixing, plus a multi-file project agent
- WhatsApp/Telegram messages and OS reminders
- Macro record/replay, undo, on-screen confirmation for risky actions
  (shutdown, restart, Wi-Fi)

### Learning
- Study Mode: notes from the screen, quizzes, progress tracking with spaced
  review

### Reach and extensibility
- Phone dashboard (same Wi-Fi, or anywhere via Tailscale)
- MCP connectors: docker, filesystem, git
- Plugins and actions that hot-reload while JARVIS runs
- Suggestions based on usage patterns

### Known gap
The `plugins/` folder only contains `_template.py`, so the Gmail, Calendar and
smart-home features that `requirements.txt` mentions are **not installed** in
this build.

## What would make it outshine

| # | Improvement | Why it matters | Builds on |
|---|---|---|---|
| 1 | **Answer only its owner (voice ID)** | It has replied to side conversations and opened YouTube from misheard audio. A local speaker-verification check before the mic is streamed would make it ignore anyone who isn't you. | wake-word mic path |
| 2 | **Instant replies for common voice commands** | "Volume up", "open Chrome", "pause" take ~3 s through the cloud model; handled locally they feel instant. | `core/fast_intent.py` (typed text only today), roadmap 1.4 |
| 3 | **Keep listening while it works** | A slow tool blocks the conversation. Long tasks should run in the background and report back ("Downloading. I'll tell you when it's done"). | roadmap 1.2 |
| 4 | **Notice what you're doing** | "You've had that build error for 10 minutes. Want me to look?" is what makes it feel like the movie. The pieces exist but aren't wired into the voice. | screen capture, `predictive_assistant`, `proactive`, roadmap 5.1–5.2 |
| 5 | **Calendar, email and messages** | A morning briefing with real meetings and unread mail is worth more than another tool. | plugin system, rebuild Gmail/Calendar/smart-home plugins |
| 6 | **Works offline** | When Gemini is down or out of quota (1011 errors, 20-requests/day free tier), switch to Local Mode automatically instead of going silent. | Local Mode (Ollama + local STT/TTS) |
| 7 | **Safer computer control** | JARVIS types into other apps and presses Enter. Add an "acting on your behalf" indicator, an allow-list of apps it may type into, and confirmation before sending in someone else's window. | `core/confirm.py`, `computer_control` |
| 8 | **Hindi and Hinglish as first-class languages** | Built for the people actually using it: a Hindi voice and Hinglish handling in the prompt. | `core/prompt.txt`, voice settings |
| 9 | **Study Mode 2.0** | Real flashcard decks with spaced repetition, grading in code instead of by the voice model, study plans. Rare in a desktop assistant. | study_* actions, roadmap phase 4 |
| 10 | **Structure, so features stop breaking each other** | Most recent bugs came from features interfering: sleep vs reconnects, interrupts vs playback. Split `main.py` (4,146 lines) and `ui.py` (6,217 lines) and add recorded test conversations that run before each change. | roadmap 6.1–6.2 |

## Suggested order

1. **10, then 1** — stability, and it only answers you
2. **2 and 3** — it feels fast
3. **4 and 5** — it feels smart
4. **6–9** as time allows
