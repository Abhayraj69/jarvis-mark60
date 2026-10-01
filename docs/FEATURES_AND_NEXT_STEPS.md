# JARVIS (Mark-LV + Mark-LIII upgrades): features and next steps

Snapshot of branch `upgrade-mark-liii` (upstream Mark-LV plus the Mark-LIII
upgrades and fixes from live testing) as of 2026-10-01. The roadmap items
referenced below are in [`JARVIS_ROADMAP.md`](JARVIS_ROADMAP.md).

## What's in the project today

### Voice and presence
- Real-time voice conversation through Gemini Live. JARVIS always replies in
  English
- Local "Hey Jarvis" wake word. "Bye Jarvis" (or "sleep Jarvis", "end
  session") puts it to sleep. Each goodbye is double-checked with a local
  Whisper transcript, because the wake-word detector can't tell "hey" from
  "bye"
- Auto-sleep after 2 minutes without activity. After a reply it keeps
  listening for 45 s of quiet, then waits for the wake word
- While asleep nothing is streamed, late reply audio is dropped, and the HUD
  shows SLEEPING
- Animated 3D face in the HUD with lip-sync, 5 voices, themeable colours
- Push-to-talk, self-echo guard, barge-in (you can talk over it)
- Instant voice commands: a sentence that is exactly a device command
  ("volume up", "pause") runs locally after a 0.6 s pause, without waiting for
  the cloud
- "Arc Sentinel" desktop widget that pops up on "Hey Jarvis" (optional, needs
  `pywebview`)

### Staying up
- Live model ladder: a model that keeps failing (1011) rests, also across
  restarts, and the next one takes over
- Keep-alive silence stops the server's idle drop; a dropped session
  reconnects quietly
- Offline fallback: when every Gemini model is unavailable, JARVIS switches to
  Local Mode on its own (if a local model is installed) and switches back when
  Gemini recovers. macOS `say` is the voice of last resort
- Loop watchdog names any line that stalls the audio loop

### Thinking and memory
- `think` tool: hands real analysis to a second model
- Per-task routing across Gemini, Gemini Lite, Claude and Ollama, with
  automatic fallback when a backend fails
- Long-term memory (save / recall), cross-session conversation history,
  session summaries
- Result contract: every tool reports ok/failed, and a claim of success after
  a failure is recorded
- Correct time: the real clock is re-sent once a minute at quiet moments
- Per-turn timing and token telemetry, with a performance panel

### Computer control
- Apps, files, browser, mouse and keyboard, system settings, desktop
- Screen and webcam vision; background screen watching for a condition
  ("tell me when the download finishes")
- Web, news and price search; weather; flights; YouTube and video on the HUD;
  Steam/Epic updates
- Code writing and fixing, plus a multi-file project agent
- WhatsApp/Telegram messages and OS reminders
- Macro record/replay and undo
- Slow tools (search, flights, code, files, study notes) run in the
  background while the conversation continues, and report back when done
- Input guard: an ACTING FOR YOU indicator while JARVIS drives the keyboard.
  On-screen confirmation (for sending messages, Enter in chats, typing into
  unlisted apps, shutdown/restart/Wi-Fi) is **off by default**; turn it on in
  Plugin Settings → INPUT GUARD. Connector operations that destroy data always
  ask

### Awareness
- Tracks the focused app (stays on the computer) and mentions it in check-ins
- Break reminders after long non-stop use
- Morning briefing on the first wake of the day, with today's Calendar events
  and unread Mail (macOS Calendar and Mail apps, via the `calendar_events` and
  `email_inbox` plugins)
- Opt-in: offers help when the same error stays on screen in a terminal or
  editor

### Learning
- Study Mode: notes from the screen, quizzes, progress tracking with spaced
  review

### Reach and extensibility
- Phone dashboard (same Wi-Fi, or anywhere via Tailscale)
- MCP connectors: docker, filesystem, git
- Plugins and actions that hot-reload while JARVIS runs
- Suggestions based on usage patterns

Everything new can be switched off in Plugin Settings.

### Tried and removed
- **Voice ID** (answer only the owner): speaker verification was unreliable
  on 1–2 s clips and rejected the owner's own "Hey Jarvis"
- **Non-English filter and Hindi/Hinglish**: Gemini often transcribes English
  speech in Devanagari, so filtering by language discarded the user. JARVIS
  now replies in English only
- **Proactive audio**: it made JARVIS ignore the user until they said its name

### Known gaps
- **Offline mode isn't usable yet**: Ollama is not installed, so the
  automatic fallback has no local model to switch to
- **No smart-home control**: the Tuya/MQTT libraries are in `requirements.txt` but no
  smart-home plugin is in `plugins/`
- **Gmail and Google Calendar** plugins are not included; the briefing reads
  the macOS Mail and Calendar apps instead
- **macOS warning at launch** ("AVFFrameReceiver is implemented in both"):
  opencv-python and faster-whisper each bundle FFmpeg. Harmless, because
  nothing opens FFmpeg's capture device, and `tests/test_ffmpeg_overlap.py`
  keeps it that way

## Next steps

| # | Improvement | Why it matters | Builds on |
|---|---|---|---|
| 1 | **Split `main.py`** (~4,850 lines) and `ui.py` (~6,200 lines) | Most bugs in live testing came from features interfering: sleep vs reconnects, interrupts vs playback. Smaller modules with the existing tests make changes safe. | live harness tests, roadmap 6.1–6.2 |
| 2 | **Offline mode** | Install Ollama and a small model so the automatic fallback has somewhere to go when Gemini is down or out of quota. | `core/fallback.py`, Local Mode |
| 3 | **Smart home** | Lights and plugs by voice. Depends on which devices are in use (Tuya/Smart Life, Home Assistant, HomeKit). | plugin system |
| 4 | **Study Mode 2.0** | Real flashcard decks with spaced repetition, grading in code instead of by the voice model, study plans. | study_* actions, roadmap phase 4 |
| 5 | **Safer typing, on by default for strangers' windows** | Confirmation is off now for speed. A middle ground: ask only before sending in a messaging app. | `core/input_guard.py` |
