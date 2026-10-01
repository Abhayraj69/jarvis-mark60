import platform as _platform
import subprocess as _subprocess

# ── Nuclear: force CREATE_NO_WINDOW on EVERY subprocess call on Windows ───────
# This patches Popen itself, so no per-file flag is needed anywhere.
if _platform.system() == "Windows":
    _OrigPopen = _subprocess.Popen

    class _Popen(_OrigPopen):
        def __init__(self, args, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _subprocess.CREATE_NO_WINDOW
            kw.pop("startupinfo", None)   # drop any stale/shared STARTUPINFO
            super().__init__(args, **                       kw)

    _subprocess.Popen = _Popen


# ── Console must survive non-UTF-8 code pages ────────────────────────────────
# Every status line in this file carries an emoji, and on a legacy Windows
# console the active code page is the system one — cp1254 in Turkey, cp1251 in
# Russia, cp932 in Japan. Printing an emoji there raises UnicodeEncodeError, and
# because most of these prints sit inside the receive loop it takes the session
# down on startup. Reconfiguring to UTF-8 with a replacement fallback costs
# nothing and makes the app launch the same way in every locale.
import sys as _sys

# Under pythonw.exe there is no console at all and sys.stdout/sys.stderr are
# None. print() quietly no-ops on that, but library code that *inspects* the
# stream does not: uvicorn's log formatter calls sys.stdout.isatty(), which
# raises AttributeError, which logging.config turns into "Unable to configure
# formatter 'default'" and the dashboard never starts. Handing those libraries
# a real (discarding) stream keeps every such caller on its normal path.
import io as _io
import os as _os
for _name in ("stdout", "stderr"):
    if getattr(_sys, _name, None) is None:
        setattr(_sys, _name, _io.TextIOWrapper(
            open(_os.devnull, "wb"), encoding="utf-8", errors="replace", write_through=True
        ))

for _stream in ("stdout", "stderr"):
    try:
        _s = getattr(_sys, _stream, None)
        if _s is not None and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass          # pythonw / redirected pipes / anything exotic — never fatal

# ─────────────────────────────────────────────────────────────────────────────


import asyncio
import collections
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from google import genai
from ui import JarvisUI
from memory.memory_manager import set_trim_notifier
from actions.system_monitor import SystemMonitor
from actions.proactive import ProactiveEngine
from memory.config_manager import (
    get_brief_enabled, get_push_to_talk_enabled, get_wake_word_enabled,
    is_local_engine_enabled, get_plugin_config,
)
from core import (
    gemini as _gemini, undo as undo_stack, confirm as confirm_gate, audio_devices,
    result_contract, fallback, awareness,
)
from core.plugin_loader import discover_plugins
from core.action_loader import discover_actions
from core.echo import EchoGuard
from core.viseme import VisemeStream
from core.wake_word import WakeWordDetector, is_ready as wake_is_ready
from core.local_control import LocalControlServer
from tool_connectors.registry import ToolRegistry
from live.audio_io import AudioIOMixin
from live.background import BackgroundMixin
from live.briefing import BriefingMixin
from live.constants import RECEIVE_SAMPLE_RATE, SEND_SAMPLE_RATE, WAKE_SLEEP_TIMEOUT
from live.dashboard_bridge import DashboardBridgeMixin
from live.errors import (
    _ReconnectSignal, _classify_live_error, _is_reconnect_signal, _keep_context_of,
)
from live.fast_commands import FastCommandsMixin
from live.inline_tools import TOOL_DECLARATIONS
from live.local_mode import LocalModeMixin
from live.prompting import _get_api_key
from live.session_config import SessionConfigMixin
from live.settings_panels import SettingsPanelsMixin
from live.speech import SpeechMixin
from live.tools import ToolsMixin
from live.wake import WakeMixin

# The conversation's model. A NAME, not a decision: the ladder lives in
# core/gemini.py and this is only whichever rung is currently in use, kept here
# as a module attribute because plugins read it (chat_takeover asks main for it
# so that upgrading the assistant upgrades the plugin too).
#
# It is reassigned on every connect, so a model that runs out of quota is
# stepped over and the assistant keeps talking instead of failing to start.
LIVE_MODEL          = "models/gemini-3.1-flash-live-preview"


class JarvisLive(
    WakeMixin,
    SettingsPanelsMixin,
    DashboardBridgeMixin,
    FastCommandsMixin,
    SpeechMixin,
    SessionConfigMixin,
    ToolsMixin,
    AudioIOMixin,
    BriefingMixin,
    BackgroundMixin,
    LocalModeMixin,
):
    def __init__(self, ui: JarvisUI):
        self.ui             = ui
        self._asst_name     = "JARVI    S"   # updated each session from config
        self.session              = None
        self.audio_in_queue       = None
        self.out_queue            = None
        self._loop                     = None
        self._is_speaking         = False
        self._speaking_lock       = threading.Lock()
        self._mic_open_at         = 0.0     # mic stays gated until then (ECHO_TAIL_SECONDS)
        self._has_connected       = False   # False only until the first Live connect
        self._tools_running       = 0       # tool calls in flight (sleep watch waits for them)
        self._go_away_pending     = False   # a GoAway-triggered reconnect is scheduled
        self._last_realtime_send  = 0.0     # monotonic time audio last went to the session
        self._audio_dropped       = 0       # mic chunks dropped in the current backlog
        self._session_started_at  = 0.0     # monotonic time the current session connected
        self._last_activity       = time.monotonic()  # drives auto-sleep (see _touch_activity)
        self._last_user_text      = ""      # latest thing the user said/typed, for sleep logs
        self._phone_active        = False   # True while phone mic is streaming; pauses PC mic
        self._pending_vision       = None    # (img_bytes, mime_type, question, angle) to inject after tool response
        self._vision_cam_active    = False   # True if camera was opened for vision → auto-close after response
        self._vision_close_pending = False   # True after vision injected; next turn_complete closes camera
        self._vision_last_time     = 0.0     # monotonic time of last screen_process call (cooldown guard)
        # Result contract (core/result_contract.py): tools that returned
        # ok=false, awaiting JARVIS's next words to see whether it admits it.
        self._false_success        = result_contract.FalseSuccessTracker()
        self._in_started_at        = 0.0     # monotonic time the current user turn's transcript began
        self._think_tasks: set     = set()   # background `think` tasks (kept so they aren't GC'd)
        self._bg_tasks: set        = set()   # NON_BLOCKING tools still running (see _start_background_tool)
        self._vision_busy          = False   # True while a vision capture/inject cycle is in flight
        self._interrupted          = False   # True while draining audio after user interrupt
        self._generating           = False   # server is still producing the current reply
        # Transcript-driven mouth shapes for the avatar. Fed from the receive
        # loop as words arrive, drained by the playback loop against the audio.
        self._visemes              = VisemeStream()
        self._last_out_logged      = ""      # de-dupes a re-sent transcript tail
        # Push-to-talk
        self._ptt_enabled          = False
        self._ptt_held             = False
        self._ptt                  = None    # core.hotkey.PushToTalk
        self._out_level            = 0.0     # level of the audio being played right now
        self._echo                 = EchoGuard()
        # `stream.write()` returns when the buffer accepts the audio, not when the
        # speaker has finished with it, so sound is still in the room after the
        # speaking flag drops. Streaming the microphone during that gap is how an
        # assistant ends up answering itself. Measured from the device rather than
        # guessed; see _play_audio.
        self._out_latency          = 0.20    # seconds, replaced with the real value
        self._tail_until           = 0.0     # monotonic time the echo tail expires
        # Wall-clock time at which the audio written next will begin to sound.
        # The mouth is scheduled against this, never against "now": batches are
        # handed to the device far faster than they play, so "now" ran the lips
        # ahead of the words and cut every schedule short. 0 = nothing playing.
        self._play_cursor          = 0.0
        self.ui.on_push_to_talk   = self.set_push_to_talk
        self.ui.ptt_hold          = self._on_ptt
        self._local_stream_cancel  = threading.Event()  # set by interrupt() to stop a Local-Mode stream mid-flight
        self._local_tts_queue      = None    # queue.Queue[str|None] draining sentences to TTS during Local Mode
        self._local_tts_player     = None    # active TTSPlayer during Local Mode, so interrupt() can stop() it
        self._current_turn         = None    # core.telemetry.Turn for the in-flight Gemini Live exchange, if any
        self._active_suggestion    = None    # dict shown via ui.show_suggestion(), mirrored to the phone dashboard
        self.ui.on_text_command   = self._on_text_command
        self.ui.on_remote_clicked = self._make_remote_key
        self.ui.on_remote_tailscale_clicked = self._make_remote_key_tailscale
        self.ui.on_interrupt      = self.interrupt
        self.ui.on_voice_change   = self._on_voice_change     # voice picker → rebuild session
        self.ui.on_audio_device_change = self._on_audio_device_change
        self.ui.on_suggestion_decision = self._on_suggestion_decision
        self._suggestion_cooldown_until = 0.0  # monotonic time — throttles how often a hint can appear
        self._last_suggested_pattern    = None  # pattern_key of the hint currently on screen, or last shown
        self._reconnect_event: asyncio.Event | None = None
        self._reconnect_keep = True   # False → next rebuild drops the resumption handle

        # ── Session resumption ─────────────────────────────────────────
        # The server issues a resumption handle every few seconds and reissues
        # it as the conversation moves on. Before this, session_resumption was
        # switched ON in the config and the update was never read, so the handle
        # was thrown away and EVERY reconnect — a dropped packet, a voice change,
        # switching microphone — started an empty session. "Unlimited sessions"
        # leaked through exactly this hole.
        #
        # Deliberately in RAM only, never written to disk. Persisting it would
        # make a fresh launch continue yesterday's conversation, which sounds
        # appealing but breaks the session-summary flow: _save_session_summary
        # runs at shutdown and the morning briefing pops it the next day. A
        # conversation that never ends never produces a summary, and the
        # "yesterday we talked about…" line silently disappears.
        self._resume_handle: str | None = None
        self._turn_done_event: asyncio.Event | None = None
        self._dashboard     = None
        self._briefing_sent    = False          # morning briefing fires once per process
        self._briefing_day     = ""             # date of the last briefing (first-wake briefing)
        self._sys_monitor      = SystemMonitor()  # persistent cooldown state
        self._proactive        = ProactiveEngine()
        self._last_user_speech = time.monotonic()  # updated on every user utterance
        self._session_log: list[str] = []          # conversation turns for end-of-session summary

        self._enhanced_live = True  # proactive audio; auto-disabled if the server rejects it
        self._tuned_live    = True  # turn-taking / media / thinking knobs; same fallback

        # ── Engine mode: Cloud (Gemini Live) vs. Local (offline STT/LLM/TTS) ──
        # Decided once at startup, not re-read mid-run: the two pipelines are
        # structurally different (a streaming multimodal session vs. a
        # blocking record → transcribe → chat → speak loop), so switching
        # requires a restart — see ⚙ → PLUGIN SETTINGS → ENGINE in the UI.
        self._mode = "local" if is_local_engine_enabled() else "cloud"
        # Automatic Local Mode while Gemini is down (core/fallback.py).
        self._cloud_health   = fallback.CloudHealth()
        self._fallback_due: bool | None = None   # set by the run loop; value = models_resting
        self._in_fallback    = False
        self._fallback_streak = 0                # consecutive fallbacks: stretches the probe
        self._fallback_warned_at = -1e9
        self._awareness = awareness.Awareness()    # what the user is doing (core/awareness.py)

        _base_dir = Path(__file__).resolve().parent
        _inline_names = {t["name"] for t in TOOL_DECLARATIONS}

        # File-backed tools: every actions/*.py with a TOOL dict, discovered the
        # same way plugins are. Reserved names = the inline tools above, so an
        # action can never shadow one.
        self._action_registry = discover_actions(
            actions_dir=_base_dir / "actions",
            reserved_names=_inline_names,
            logger=lambda msg: print(f"[Actions] {msg}"),
        )

        # Plugins must not collide with either an inline tool or a discovered action.
        _core_names = _inline_names | self._action_registry.names()
        self._plugin_registry = discover_plugins(
            plugins_dir=_base_dir / "plugins",
            core_tool_names=_core_names,
            # Console gets the full boot transcript; the activity log gets only
            # what the user has to know about. Every plugin loading correctly is
            # the expected case and does not belong in their conversation.
            logger=lambda msg: print(f"[Plugins] {msg}"),
            notify=lambda msg: self.ui.write_log(f"SYS: {msg}"),
        )
        self.ui.get_plugins = self._plugin_registry.list_for_ui
        self.ui.get_plugin_settings = self._settings_schemas  # ⚙ settings tab
        self.ui.reload_all_skills = self._reload_all_skills   # Plugin Manager: RELOAD ALL
        self.ui.request_say = self.plugin_say   # plugins: mid-task speech channel

        # tool_connectors/: a second registry (git/Docker/filesystem today),
        # published alongside actions and plugins with the same reserved-name
        # discipline. Declarations are namespaced "{connector}__{action}"
        # (see ToolRegistry.get_tool_declarations), so a collision here would
        # mean an action or plugin was deliberately named e.g. "git__commit" —
        # reserved_names below catches that rather than silently shadowing it.
        _names_incl_plugins = _core_names | {d["name"] for d in self._plugin_registry.get_tool_declarations()}
        self._tool_connector_registry = ToolRegistry(
            logger=lambda msg: (print(f"[Connectors] {msg}"), self.ui.write_log(f"SYS: {msg}")),
        ).discover()
        self._connector_declarations = self._tool_connector_registry.get_tool_declarations(
            reserved_names=_names_incl_plugins
        )

        # ── Skill hot reload (core/skill_watcher.py) ───────────────────────────
        # Off by default, and — like ENGINE mode above — read once at startup:
        # the watcher is a background thread, and toggling it live would need
        # its own start/stop plumbing for one setting nobody changes mid-run.
        # Local Mode already rebuilds its tool list fresh every turn (see
        # _run_local_loop), so only Live mode needs the reconnect nudge below.
        self._skill_watcher = None
        if bool(get_plugin_config("hot_reload").get("enabled", False)):
            from core.skill_watcher import SkillWatcher

            def _on_skills_reloaded(reason: str):
                if self._mode == "cloud":
                    self.request_reconnect(keep_context=True, reason=reason)

            self._skill_watcher = SkillWatcher(
                plugins_dir=_base_dir / "plugins",
                actions_dir=_base_dir / "actions",
                plugin_registry=self._plugin_registry,
                action_registry=self._action_registry,
                on_change=_on_skills_reloaded,
                logger=lambda msg: (print(msg), self.ui.write_log(f"SYS: {msg}")),
            )
            self._skill_watcher.start()

        # ── Wake word ────────────────────────────────────────────────────────
        # _awake gates the mic (see _listen_audio) and the background speakers.
        # It is True whenever wake word is OFF, so default behaviour is unchanged.
        self._wake_enabled     = get_wake_word_enabled()
        self._awake            = not self._wake_enabled
        self._wake_detector: WakeWordDetector | None = None
        self._wake_sleep_timeout = WAKE_SLEEP_TIMEOUT

        # Restore the saved push-to-talk preference. Doing it here rather than
        # in __init__ means the hotkey thread only exists once there is a
        # session to talk to.
        if get_push_to_talk_enabled():
            try:
                self.set_push_to_talk(True)
            except Exception as e:
                print(f"[JARVIS] ⚠ Push-to-talk unavailable: {e}")
        # Set only by _enter_standby() ("bye jarvis" on a session that never
        # turned wake-word mode on) so wake() knows to put _wake_enabled back
        # exactly as the user had it once they say "Hey Jarvis" again — the
        # standby is a one-off detour, not a silent, permanent settings change.
        self._standby_forced_wake         = False
        self._standby_restore_wake_enabled = False
        # See WAKE_SETTLE_SECONDS / STANDBY_REENTRY_GUARD_SECONDS above.
        self._wake_feed_gate_open_at  = 0.0
        self._wake_heard_at = -1e9      # last "Hey Jarvis" heard while already awake
        # Last SLEEP_CHECK_SECONDS of the user's mic audio (int16 blocks), and a
        # lazily loaded local Whisper, to double-check a sleep request.
        self._recent_mic: collections.deque = collections.deque()
        self._recent_mic_len = 0
        self._sleep_stt = None
        self._sleep_stt_failed = False
        # Per-reply audio health: [first arrival, last arrival, bytes] and how
        # often the speaker ran dry — tells a slow network apart from a busy CPU.
        self._reply_audio = None
        self._underruns = 0
        self._clock_sent = ""          # minute ("HH:MM") of the last [CLOCK] note
        # Spoken fast commands: a sequence number that cancels a pending match
        # when more speech (or a model tool call) arrives, and the last one run.
        self._fast_voice_seq = 0
        self._fast_voice_ran_turn = False
        self._fast_voice_done: tuple | None = None   # (Intent, monotonic time)
        self._goodbye_seq = 0           # cancels a pending local goodbye when more speech arrives
        self._standby_reentry_guard_until = 0.0
        # UI control surface for the Wake Word settings section.
        self.ui.wake_is_ready    = wake_is_ready          # () -> bool
        self.ui.wake_get_state   = self._wake_state       # () -> dict
        self.ui.on_wake_toggle   = self._ui_wake_toggle   # (enable: bool) -> str
        self.ui.on_wake_manual   = self._ui_wake_manual   # () -> toggle awake/asleep
        self.ui.on_wake_install  = self._ui_wake_install  # () -> (ok, msg)


    def request_reconnect(self, keep_context: bool = True, reason: str = ""):
        """Thread-safe: ask the run loop to tear down and rebuild the Live
        session. Called from the Qt thread. No-op until the async loop and
        reconnect event exist.

        `keep_context=False` drops the resumption handle so the new session
        starts empty — only for changes the server cannot apply to a resumed
        session."""
        loop = getattr(self, "_loop", None)
        ev   = self._reconnect_event
        self._reconnect_keep   = keep_context
        self._reconnect_reason = reason
        if loop and ev is not None:
            loop.call_soon_threadsafe(ev.set)

    async def _reconnect_before_go_away(self, go_away) -> None:
        """Reconnect (keeping the conversation) once JARVIS is not mid-answer or
        mid-tool, but always before the server's deadline."""
        left = getattr(go_away, "time_left", None)   # e.g. "50s"
        try:
            left_s = float(str(left).rstrip("s")) if left else 10.0
        except ValueError:
            left_s = 10.0
        print(f"[JARVIS] ⏳ Session ending in {left_s:.0f}s — reconnecting at the next pause")
        deadline = time.monotonic() + max(0.0, left_s - 2.0)
        while time.monotonic() < deadline:
            with self._speaking_lock:
                busy = self._is_speaking
            if not busy and self._tools_running == 0 and not self._tail_active():
                break
            await asyncio.sleep(0.25)
        self.request_reconnect(keep_context=True, reason="session refresh")

    def _on_voice_change(self):
        """Voice picker applied.

        The voice is baked into the session at connect time, so a rebuild is
        required. It is rebuilt WITHOUT the resumption handle on purpose:
        resuming restores the server's own session state, and the safe reading
        is that it restores the voice with it — which would make the picker
        appear to do nothing. Losing context here is acceptable because changing
        voice is a deliberate, rare act; losing it on a dropped packet was not."""
        self.request_reconnect(keep_context=False, reason="new voice")

    def _on_audio_device_change(self):
        """Microphone or speaker changed. Both streams are opened inside the
        session TaskGroup, so they can only be re-opened by rebuilding it —
        but the conversation is kept, which is the whole reason resumption
        landed before this feature did."""
        self.request_reconnect(keep_context=True, reason="audio device")

    async def _watch_reconnect(self):
        """Session-scoped task: when a voluntary reconnect is requested, raise a
        signal that unwinds the TaskGroup so the run loop rebuilds the session."""
        assert self._reconnect_event is not None
        await self._reconnect_event.wait()
        self._reconnect_event.clear()
        keep   = self._reconnect_keep
        reason = getattr(self, "_reconnect_reason", "") or "settings"
        self.ui.write_log(
            f"SYS: Applying {reason} — reconnecting"
            + ("..." if keep else " (starting a fresh conversation)...")
        )
        raise _ReconnectSignal(keep_context=keep)


    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self):
        self._loop = asyncio.get_event_loop()
        # Prints the exact blocking line if anything freezes the loop (which
        # stops mic audio, replies and playback all at once).
        from core.loop_watchdog import LoopWatchdog
        LoopWatchdog(self._loop).start()
        self._reconnect_event = asyncio.Event()

        # ── Wire the shared core services to the interface ───────────────────
        # The confirmation gate is useless without a way to ask, and a memory
        # trim is invisible without a way to say so. Both are bound once here
        # rather than passed down through every action signature.
        def _show_confirm_and_broadcast(title, detail):
            self.ui.show_confirm(title, detail)
            self._broadcast_remote_state_threadsafe()

        def _hide_confirm_and_broadcast():
            self.ui.hide_confirm()
            self._broadcast_remote_state_threadsafe()

        confirm_gate.bind(
            show = _show_confirm_and_broadcast,
            hide = _hide_confirm_and_broadcast,
            log  = self.ui.write_log,
        )
        undo_stack.bind(self._broadcast_remote_state_threadsafe)
        set_trim_notifier(self.ui.write_log)

        # Tell the device picker the exact rates the streams open at, from the
        # constants that actually open them — so it can never list a device that
        # cannot be opened at them.
        audio_devices.configure(SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE)

        # Enumerate audio devices off-thread. The settings drawer must never pay
        # for host-API enumeration on the Qt thread.
        audio_devices.prefetch()

        # Start dashboard (optional — needs: pip install fastapi "uvicorn[standard]" cryptography)
        try:
            from dashboard.server import DashboardServer
            self._dashboard = DashboardServer()
            self._dashboard.set_connect_callback(self._on_phone_connected)
            self._dashboard.set_confirm_callback(self._dashboard_confirm)
            self._dashboard.set_undo_callback(self._dashboard_undo)
            self._dashboard.set_suggestion_callback(self._dashboard_suggestion)

            async def _run_dashboard():
                # asyncio.create_task() below only SCHEDULES this coroutine —
                # an exception raised once it's actually running (e.g.
                # uvicorn failing to bind the port because something else
                # already holds it) is not caught by this method's own
                # try/except, which only covers the synchronous setup above.
                # Uncaught, it would vanish into asyncio's default "Task
                # exception was never retrieved" handler — silent, because
                # JARVIS runs under pythonw.exe with no console to print it
                # to. The result: a phone user gets a QR code / remote link
                # pointing at a dashboard that was never actually listening,
                # with zero indication anything went wrong — "this site
                # can't be reached" and no clue why.
                try:
                    await self._dashboard.serve()
                except Exception as e:
                    self.ui.write_log(f"ERR: Dashboard failed to start — {e}")
                    self._dashboard_error = str(e)
                    self._dashboard = None

            asyncio.create_task(_run_dashboard())
            # Runs for the whole lifetime, not just inside an active session
            asyncio.create_task(self._process_dashboard_commands())
        except Exception as e:
            print(f"[Dashboard] Disabled: {e}")
            self._dashboard = None

        # Loopback-only control channel for local companion processes (the
        # Arc Sentinel widget's mic-mute / interrupt buttons) — stdlib only,
        # so it works whether or not the (optional) dashboard is installed.
        def _toggle_mute_and_settle():
            # toggle_mute() only queues a cross-thread Qt signal — it returns
            # before _toggle_mute() actually runs on the Qt thread. Without
            # this short wait, the POST /mute response (read right after)
            # would report the PRE-toggle state; the widget's next poll
            # would still self-correct, but this makes the click feel instant
            # instead of one 900ms poll cycle behind.
            self.ui.toggle_mute()
            time.sleep(0.05)

        self._local_control = LocalControlServer(
            get_state      = self._local_control_state,
            on_mute_toggle = _toggle_mute_and_settle,
            on_interrupt   = self.interrupt,
        )
        if not self._local_control.start():
            print("[LocalControl] Port already in use — probably another JARVIS instance.")

        if self._mode == "local":
            # Entirely different pipeline (blocking record → transcribe →
            # chat → speak loop, no Live session, no TaskGroup) — the cloud
            # while-loop below is untouched and simply never reached.
            await self._run_local_loop()
            return

        while True:
            try:
                due, self._fallback_due = self._fallback_due, None
                if due is not None and await self._maybe_fall_back(models_resting=due):
                    self._conn_backoff = 0
                print("[JARVIS] Connecting...")
                self.ui.set_state("THINKING")
                # Uptime is measured per attempt: a failure before this one
                # connects must not inherit the previous session's age.
                self._session_started_at = time.monotonic()
                # Pick the rung to open the conversation on. A model resting
                # off a quota limit is skipped; the name is published back to
                # LIVE_MODEL so plugins follow whatever is actually in use.
                global LIVE_MODEL
                LIVE_MODEL = _gemini.live_model()
                live_model = LIVE_MODEL
                print(f"[JARVIS] Live model: {live_model}")

                _resumed_with = self._resume_handle is not None
                config = self._build_config()

                # Fresh client on every reconnect — avoids stale HTTP session state
                # v1alpha carries proactive audio; if it gets rejected we fall
                # back to v1beta.
                client = genai.Client(
                    api_key=_get_api_key(),
                    http_options={"api_version": "v1alpha" if self._enhanced_live else "v1beta"}
                )

                async with (
                    client.aio.live.connect(model=live_model, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session          = session
                    self.audio_in_queue   = asyncio.Queue()
                    self.out_queue        = asyncio.Queue(maxsize=200)
                    self._turn_done_event = asyncio.Event()

                    # Reset transient state that must not carry over from a previous session
                    self._pending_vision       = None
                    self._vision_cam_active    = False
                    self._vision_close_pending = False
                    self._vision_busy          = False
                    self._vision_last_time     = 0.0
                    self._interrupted          = False
                    self._generating           = False

                    print("[JARVIS] Connected.")
                    self._session_started_at = time.monotonic()
                    self._last_realtime_send = self._session_started_at
                    if _resumed_with:
                        # Say it plainly: the difference between "it reconnected"
                        # and "it reconnected and still knows what we were doing"
                        # is the whole point, and it is invisible otherwise.
                        self.ui.write_log("SYS: Reconnected — conversation restored.")

                    self._on_session_connected()
                    self._cloud_health.connected()

                    if self._dashboard:
                        await self._dashboard.broadcast({"type": "status", "state": "active"})

                    self._reconnect_event.clear()  # ignore requests from before this session
                    tg.create_task(self._watch_reconnect())
                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                    tg.create_task(self._run_system_monitor())
                    tg.create_task(self._run_background_monitor())
                    tg.create_task(self._run_proactive_mode())
                    tg.create_task(self._run_awareness())
                    tg.create_task(self._run_sleep_watch())
                    if self._dashboard:
                        tg.create_task(self._relay_phone_audio())

                    # Morning briefing — fires once per process launch (if enabled).
                    # Skipped in wake-word mode: it comes up asleep, and a briefing
                    # would mean talking while "asleep".
                    if not self._briefing_sent and get_brief_enabled() and self._awake:
                        self._briefing_sent = True
                        self._briefing_day = datetime.now().strftime("%Y-%m-%d")
                        tg.create_task(self._send_startup_briefing())

            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except BaseException as e:
                # Catches both Exception and BaseExceptionGroup (Python 3.11+
                # TaskGroup raises BaseExceptionGroup when tasks are cancelled
                # externally, which `except Exception` would miss, letting the
                # exception escape the while-loop and causing asyncio.run() to
                # start shutdown — resulting in "executor after shutdown" errors).
                # Voluntary reconnect (voice change) — not an error. Rebuild the
                # session immediately with no backoff and no scary logs.
                if _is_reconnect_signal(e):
                    print("[JARVIS] Voluntary reconnect requested.")
                    if not _keep_context_of(e):
                        # A deliberate clean slate (voice change) — drop the
                        # handle so the next connect really does start empty.
                        self._resume_handle = None
                    self._conn_backoff = 0
                    continue

                # The session's TaskGroup wraps the real failure: str() of the
                # group is only "unhandled errors in a TaskGroup", which hid
                # quota (429) and internal (1011) errors from the model ladder
                # below, so it never switched models.
                def _exc_text(exc) -> str:
                    subs = getattr(exc, "exceptions", None)
                    if subs:
                        return " | ".join([str(exc)] + [_exc_text(x) for x in subs])
                    return str(exc)
                err_str = _exc_text(e)
                kind = _classify_live_error(
                    err_str, str(e),
                    resumed_with=_resumed_with,
                    uptime=time.monotonic() - self._session_started_at,
                    tuned=self._tuned_live, enhanced=self._enhanced_live,
                )

                if time.monotonic() - self._session_started_at >= 300:
                    self._fallback_streak = 0     # the cloud held up for a while

                if kind == "bad_handle":
                    print("[JARVIS] 🔗 Resumption handle rejected — starting a fresh session")
                    self.ui.write_log("SYS: Could not restore the conversation — starting fresh.")
                    self._resume_handle = None
                    self._conn_backoff = 0
                    continue

                if kind == "idle_drop":
                    print("[JARVIS] Live session dropped (1008) — reconnecting now.")
                    self._conn_backoff = 0
                    continue

                if kind == "network":
                    # Expected when Wi-Fi drops or the Mac sleeps: one line, not a traceback.
                    print(f"[JARVIS] Connection lost ({err_str.split(' | ')[-1][:120]}).")
                else:
                    print(f"[JARVIS] Error ({type(e).__name__}): {e}")
                    traceback.print_exc()

                # Out of quota, or this model is not available to this key —
                # step down the ladder and reconnect straight away. This is the
                # difference between "JARVIS is quieter today" and "JARVIS does
                # not start today": one model means one daily limit, and the
                # limit always arrives mid-conversation.
                if _gemini.note_live_failure(live_model, err_str):
                    nxt = _gemini.live_model()
                    self.ui.write_log(
                        f"SYS: Switching to {nxt.split('/')[-1]} — the previous "
                        f"model is out of quota or failing."
                        if nxt != live_model else
                        "SYS: Every live model is rate-limited — retrying.")
                    self._conn_backoff = 0 if nxt != live_model else 15
                    if _gemini.all_live_models_resting():
                        self._fallback_due = True
                    if nxt == live_model:
                        await asyncio.sleep(self._conn_backoff)
                    continue

                # Turn-taking / media / thinking knobs rejected by the server
                # (preview API drift) — drop them first, because they are the
                # newest fields and the cheapest to lose. Proactive audio is
                # tried again on the next pass if the error persists.
                if kind == "drop_tuning":
                    self._tuned_live = False
                    print("[JARVIS] Live tuning rejected — reconnecting without it.")
                    continue

                # Proactive audio rejected by the server (preview API drift) —
                # drop it and reconnect with the plain config.
                if kind == "drop_proactive":
                    self._enhanced_live = False
                    self.ui.write_log(
                        "SYS: Proactive audio unavailable — reconnecting without it."
                    )
                    continue

                # Invalid API key — stop hammering the API, prompt re-configuration
                if kind == "bad_key":
                    self.ui.write_log("ERR: API key invalid — please re-enter your key.")
                    self.ui.set_state("SLEEPING")
                    self.ui.prompt_reconfig()
                    while not self.ui._win._ready:
                        await asyncio.sleep(1)
                    print("[JARVIS] New API key saved — reconnecting...")
                    _conn_backoff = 3
                    continue

                # Network / timeout errors — log clearly and back off
                if kind == "network":
                    _conn_backoff = min(getattr(self, "_conn_backoff", 3) * 2, 60)
                    self._conn_backoff = _conn_backoff
                    self.ui.write_log(
                        f"NET: Connection failed — retrying in {_conn_backoff}s. "
                        "(a VPN may be required)"
                    )
                else:
                    self._conn_backoff = 3
                self._cloud_health.failed(kind)
                if self._cloud_health.should_fall_back(models_resting=False):
                    self._fallback_due = False
            finally:
                self.session = None
                # Only save if there was a real conversation (≥3 turns)
                if len(self._session_log) >= 3:
                    asyncio.create_task(self._save_session_summary())

            self.set_speaking(False)
            # Between sessions. If JARVIS was awake it still is — the next
            # session keeps it awake — so show it as busy, not asleep; flashing
            # SLEEPING on every routine reconnect looked like it dozed off.
            if self._awake and self._has_connected:
                self.ui.set_state("THINKING")
            else:
                self.ui.set_state("SLEEPING")

            if self._dashboard:
                await self._dashboard.broadcast({"type": "status", "state": "sleeping"})

            delay = getattr(self, "_conn_backoff", 3)
            print(f"[JARVIS] Reconnecting in {delay}s...")
            await asyncio.sleep(delay)


def main():
    ui = JarvisUI("face.png")

    def runner():
        ui.wait_for_api_key()
        jarvis = JarvisLive(ui)
        try:
            asyncio.run(jarvis.run())
        except KeyboardInterrupt:
            print("\n🔴 Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

if __name__ == "__main__":
    main()