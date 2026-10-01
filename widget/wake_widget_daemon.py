"""
wake_widget_daemon.py — voice-triggered Arc Sentinel widget.

A standalone companion process, separate from main.py on purpose: it opens
its OWN microphone stream and runs its OWN small always-on-top window, so it
can be started/stopped independently of whichever engine (Cloud or Local)
JARVIS itself is running, and a crash here can never take the assistant down.

Say "Hey Jarvis"  -> the widget appears (pretrained openwakeword model, the
                      exact same one core/wake_word.py uses for the main
                      app's own wake gate).
Say "bye jarvis"  -> the widget disappears. There is no pretrained model for
                      this phrase, so it's spotted by transcribing short
                      rolling audio windows with faster-whisper (already a
                      dependency of core/stt.py) and checking for the words
                      "bye" and "jarvis" together. This only runs WHILE the
                      widget is visible, to keep it cheap the rest of the time.

It can also be popped open/closed with a button instead of your voice — the
main JARVIS window's "LAUNCH ARC SENTINEL" button talks to this daemon's own
loopback control server (127.0.0.1:8766, see _start_control_server below)
and calls /show, /hide or /toggle directly, same as saying the phrases would.

Requires (not part of the main app's requirements.txt — install by hand):
    pip install pywebview openwakeword faster-whisper

Run it:
    pythonw widget/wake_widget_daemon.py     (no console window)
    python  widget/wake_widget_daemon.py     (console, for troubleshooting)

Or launch it from the JARVIS phone dashboard's new "WIDGET" button, which
starts this exact script as a detached background process on the machine
running JARVIS.
"""
from __future__ import annotations

import sys

# ── Console encoding ─────────────────────────────────────────────────────
# This daemon's own status lines carry emoji ("[Widget] ⚙ ..."-style
# glyphs). On a non-UTF-8 console (cp1252 on this machine, cp1254/cp1251/
# cp932 elsewhere) printing one raises UnicodeEncodeError and kills the
# process — the same failure main.py fixed for itself (see its own
# "Console encoding" comment); this standalone entry point needs the same
# fix since nothing else applies it before print() runs.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import collections
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
WIDGET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))   # so `core.*` imports work run from anywhere

SAMPLE_RATE   = 16000
CHUNK_SIZE    = 1024
BYE_WINDOW_S  = 3.0     # seconds of rolling audio checked for "bye jarvis"
BYE_POLL_S    = 1.4     # how often that window is transcribed while visible
PID_FILE      = WIDGET_DIR / ".wake_widget.pid"
DEFAULT_ACCENT = "#00d4ff"   # matches ui.py's DEFAULT_UI_COLOR (unthemed default)
CONTROL_PORT  = 8766    # loopback-only — separate from JARVIS's own 8765 (core/local_control.py)

IS_MAC        = sys.platform == "darwin"
PREFS_FILE    = WIDGET_DIR / ".widget_prefs.json"

# The widget's three sizes, switched from buttons on the widget itself (see
# arc_sentinel_widget.html) and remembered in PREFS_FILE between launches:
# (width, height, corner radius). The radius is only used on macOS, where the
# rounded shape comes from the native vibrancy view rather than CSS.
WIDGET_SIZES = {
    "full":    (236, 276, 18),
    "compact": (304, 64, 32),
    "mini":    (60, 60, 30),
}
DEFAULT_MODE = "full"

# On top of the layout, the whole widget can be scaled down for small screens
# (right-click / "⋯" menu, or ⌘− ⌘= ⌘0). The page zooms by the same factor the
# window is resized by, so the layout itself never reflows — it just renders
# smaller. Presets rather than free dragging, so the text stays crisp.
WIDGET_SCALES = [
    ("Extra Small", 0.6),
    ("Small",       0.7),
    ("Medium",      0.85),
    ("Default",     1.0),
]
DEFAULT_SCALE = 1.0
MODE_LABELS = {"full": "Full", "compact": "Compact", "mini": "Orb Only"}


def _nearest_scale(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_SCALE
    return min((sc for _, sc in WIDGET_SCALES), key=lambda sc: abs(sc - value))


def _load_prefs() -> tuple[str, float]:
    try:
        prefs = json.loads(PREFS_FILE.read_text())
    except Exception:
        prefs = {}
    mode = prefs.get("mode")
    return (mode if mode in WIDGET_SIZES else DEFAULT_MODE,
            _nearest_scale(prefs.get("scale", DEFAULT_SCALE)))


def _save_prefs(mode: str, scale: float) -> None:
    try:
        PREFS_FILE.write_text(json.dumps({"mode": mode, "scale": scale}))
    except Exception as e:
        print(f"[Widget] Could not save size preference: {e}")


def _scaled_size(mode: str, scale: float) -> tuple[int, int, int]:
    width, height, radius = WIDGET_SIZES[mode]
    return round(width * scale), round(height * scale), round(radius * scale)


def _apply_mac_chrome(window, radius: int) -> None:
    """Real macOS frosted glass: a dark HUD-material NSVisualEffectView behind
    the (transparent) page, clipped to the widget's rounded shape, with the
    native window shadow following that shape. Re-run after every size change
    so the corner radius and shadow match. Must run on the main thread."""
    import AppKit
    ns_window = window.native
    if ns_window is None:
        return
    host = ns_window.contentView()
    effect = getattr(window, "_sentinel_effect", None)
    if effect is None:
        effect = AppKit.NSVisualEffectView.alloc().initWithFrame_(host.bounds())
        effect.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        effect.setMaterial_(AppKit.NSVisualEffectMaterialHUDWindow)
        effect.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
        effect.setState_(AppKit.NSVisualEffectStateActive)
        effect.setAppearance_(AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameVibrantDark))
        effect.setWantsLayer_(True)
        host.addSubview_positioned_relativeTo_(effect, AppKit.NSWindowBelow, None)
        window._sentinel_effect = effect
    effect.setFrame_(host.bounds())
    effect.layer().setCornerRadius_(radius)
    effect.layer().setMasksToBounds_(True)
    ns_window.setHasShadow_(True)
    ns_window.invalidateShadow()


def _pin_to_all_spaces(window) -> None:
    """Keep the widget on every Space (desktop) and over full-screen apps.

    pywebview leaves the NSWindow with the default collection behaviour, which
    ties a window to the one Space it was opened on, so swiping to another
    desktop leaves the widget behind. It also runs the process as a regular
    Dock app (activation policy 0), and macOS never draws a regular app's
    windows inside another app's full-screen Space, whatever the behaviour
    flags say. So:
      * CanJoinAllSpaces       — present on every desktop, not just one
      * Stationary             — stays put during the Space-switch animation
                                 and Mission Control instead of sliding away
      * FullScreenAuxiliary    — allowed to sit on top of full-screen apps
      * IgnoresCycle           — not part of Cmd-` window cycling
      * Accessory activation   — no Dock icon or menu bar, like a menu-bar
                                 utility; required for the full-screen case
    Must run on the main thread."""
    import AppKit
    ns_window = window.native
    if ns_window is None:
        return
    ns_window.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorStationary
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        | AppKit.NSWindowCollectionBehaviorIgnoresCycle)
    # Above normal and floating windows; pywebview's on_top uses the same level.
    ns_window.setLevel_(AppKit.NSStatusWindowLevel)
    ns_window.setHidesOnDeactivate_(False)
    ns_window.setCanHide_(False)   # survives Cmd-H / "Hide Others"
    nonactivating = bool(ns_window.styleMask() & AppKit.NSWindowStyleMaskNonactivatingPanel)
    print(f"[Widget] Pinned to all Spaces (behaviour={ns_window.collectionBehavior()}, "
          f"level={ns_window.level()}, "
          f"policy={AppKit.NSApplication.sharedApplication().activationPolicy()}, "
          f"window={ns_window.className()}, nonactivating={nonactivating})")


def _install_mac_overlay_window() -> None:
    """Must run BEFORE webview.create_window()/start().

    pywebview builds every window as a plain NSWindow subclass (WindowHost)
    and runs the process as a regular Dock app. Collection-behaviour flags
    alone weren't enough for that combination: the widget stayed on the
    desktop it was opened on, and never appeared over full-screen apps.
    macOS overlays that follow you everywhere (Spotlight-style HUDs, floating
    clocks) are non-activating NSPanels owned by an accessory (LSUIElement)
    app, so:
      * swap pywebview's window class for an NSPanel that always carries
        NSWindowStyleMaskNonactivatingPanel — clicking it never activates
        the app, so macOS never re-homes it to "the app's" Space;
      * switch the process to the accessory activation policy now, before
        any window exists, instead of flipping it on a live window.
    """
    import AppKit
    import objc
    from webview.platforms import cocoa

    class SentinelPanel(AppKit.NSPanel):
        def initWithContentRect_styleMask_backing_defer_(self, rect, mask, backing, defer):
            self = objc.super(SentinelPanel, self).initWithContentRect_styleMask_backing_defer_(
                rect, mask | AppKit.NSWindowStyleMaskNonactivatingPanel, backing, defer)
            return self

        # Key (so its buttons and page get clicks) but never main, never
        # pulling the app forward — same as a Spotlight/HUD panel.
        def canBecomeKeyWindow(self):
            return True

        def canBecomeMainWindow(self):
            return False

    cocoa.BrowserView.WindowHost = SentinelPanel
    cocoa.BrowserView.app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)


def _mac_show(window) -> None:
    """Bring the widget up on whatever Space is current WITHOUT activating the
    app. pywebview's own show() calls activateIgnoringOtherApps_, which steals
    focus and makes macOS treat the window as belonging to that app's Space."""
    from PyObjCTools import AppHelper
    AppHelper.callAfter(lambda: window.native.orderFrontRegardless())


_SentinelMenuTarget = None   # ObjC classes can only be defined once per process


def _show_mac_menu(controller) -> None:
    """Native right-click menu: layout, size, hide. Main thread only.

    Choices are sent back through the page (setMode / setScale in
    arc_sentinel_widget.html) on a worker thread — evaluate_js waits for the
    page, and the page needs this main thread to answer, so calling it
    directly from the menu action would deadlock."""
    global _SentinelMenuTarget
    import AppKit
    if _SentinelMenuTarget is None:
        class SentinelMenuTarget(AppKit.NSObject):
            def pick_(self, sender):
                self.handler(str(sender.representedObject()))
        _SentinelMenuTarget = SentinelMenuTarget

    def handle(choice: str) -> None:
        kind, _, value = choice.partition(":")
        def run():
            try:
                if kind == "mode":
                    controller.window.evaluate_js(f"setMode({json.dumps(value)})")
                elif kind == "scale":
                    controller.window.evaluate_js(f"setScale({float(value)})")
                elif kind == "hide":
                    controller.on_bye_jarvis()
            except Exception as e:
                print(f"[Widget] Menu action failed: {e}")
        threading.Thread(target=run, daemon=True).start()

    target = _SentinelMenuTarget.alloc().init()
    target.handler = handle
    controller._menu_target = target   # NSMenuItem doesn't retain its target

    menu = AppKit.NSMenu.alloc().initWithTitle_("Sentinel")
    menu.setAutoenablesItems_(False)

    def header(title):
        it = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, None, "")
        it.setEnabled_(False)
        menu.addItem_(it)

    def item(title, value, checked=False):
        it = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "pick:", "")
        it.setTarget_(target)
        it.setRepresentedObject_(value)
        it.setState_(AppKit.NSControlStateValueOn if checked else AppKit.NSControlStateValueOff)
        menu.addItem_(it)

    header("Layout")
    for mode, label in MODE_LABELS.items():
        item(label, f"mode:{mode}", mode == controller.mode)
    menu.addItem_(AppKit.NSMenuItem.separatorItem())
    header("Size")
    for label, scale in WIDGET_SCALES:
        item(f"{label}  ({round(scale * 100)}%)", f"scale:{scale}", scale == controller.scale)
    menu.addItem_(AppKit.NSMenuItem.separatorItem())
    item("Hide Widget", "hide:")

    menu.popUpMenuPositioningItem_atLocation_inView_(None, AppKit.NSEvent.mouseLocation(), None)


def _mac_chrome_later(window, radius: int) -> None:
    from PyObjCTools import AppHelper
    def run():
        try:
            _apply_mac_chrome(window, radius)
        except Exception as e:
            print(f"[Widget] macOS glass effect unavailable: {e}")
    AppHelper.callAfter(run)


def _mac_pin_later(window) -> None:
    from PyObjCTools import AppHelper
    def run():
        try:
            _pin_to_all_spaces(window)
        except Exception as e:
            print(f"[Widget] Could not pin widget to all Spaces: {e}")
    AppHelper.callAfter(run)


def _get_accent_hex() -> str:
    """Read the user's chosen HUD accent colour (⚙ → CUSTOMISE ASSISTANT,
    stored as config/api_keys.json's "ui_color") so the widget matches
    whatever theme is live in the main window instead of a fixed blue. A
    plain read-only json parse — no config_manager caching needed since this
    runs once at daemon startup, not on a hot path."""
    try:
        import json
        cfg = json.loads((BASE_DIR / "config" / "api_keys.json").read_text(encoding="utf-8"))
        hex_ = (cfg.get("ui_color") or "").strip().lower()
        if hex_.startswith("#") and len(hex_) == 7:
            int(hex_[1:], 16)   # validates it's actually hex
            return hex_
    except Exception:
        pass
    return DEFAULT_ACCENT


def _check_singleton() -> bool:
    """Refuse to start a second daemon — two of these would fight over the
    microphone and pop two widgets. Returns True if it's safe to proceed."""
    try:
        if PID_FILE.exists():
            old_pid = int(PID_FILE.read_text().strip())
            if _pid_alive(old_pid):
                print(f"[Widget] Already running (pid {old_pid}). Not starting a second copy.")
                return False
    except Exception:
        pass
    try:
        PID_FILE.write_text(str(os.getpid()))
    except Exception:
        pass
    return True


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _check_deps() -> bool:
    missing = []
    try:
        import webview  # noqa: F401
    except ImportError:
        missing.append("pywebview")
    try:
        import openwakeword  # noqa: F401
    except ImportError:
        missing.append("openwakeword")
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        missing.append("faster-whisper")
    try:
        import sounddevice  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        missing.append("sounddevice numpy")
    if missing:
        print("[Widget] Missing dependencies:", ", ".join(missing))
        print(f"[Widget] Install with:  {sys.executable} -m pip install " + " ".join(missing))
        return False

    from core.wake_word import is_ready
    if not is_ready():
        print("[Widget] The 'Hey Jarvis' wake model isn't downloaded yet.")
        print("[Widget] Open JARVIS -> Settings -> WAKE WORD -> download once (shared with this widget).")
        return False
    return True


class WidgetController:
    """Owns the pywebview window and the two listening loops. Everything that
    touches audio runs off pywebview's own thread — window.show()/hide() are
    the only pywebview calls made from other threads, and both are
    documented as thread-safe."""

    def __init__(self):
        self.window   = None
        self.mode     = DEFAULT_MODE     # layout + scale: set from prefs in main()
        self.scale    = DEFAULT_SCALE
        self._menu_target = None
        self.visible  = False
        self._ring    = collections.deque(maxlen=int(SAMPLE_RATE * BYE_WINDOW_S / CHUNK_SIZE) + 2)
        self._lock    = threading.Lock()
        self._whisper = None   # lazy — only loaded once "hey jarvis" actually fires
        self._bye_stop_evt = threading.Event()

    # ── wake / bye transitions ────────────────────────────────────────────

    def on_hey_jarvis(self) -> None:
        if self.visible or self.window is None:
            return
        print("[Widget] 'Hey Jarvis' detected — showing widget.")
        self.visible = True
        self._bye_stop_evt.clear()
        with self._lock:
            # Drop whatever's left from the previous session — it may still
            # contain the "bye jarvis" that just hid the widget, which would
            # otherwise get transcribed again the moment listening resumes
            # and immediately re-hide the widget it was meant to bring back.
            self._ring.clear()
        try:
            if IS_MAC:
                _mac_show(self.window)
            else:
                self.window.show()
        except Exception as e:
            print(f"[Widget] show() failed: {e}")
        threading.Thread(target=self._bye_listener_loop, daemon=True).start()

    def on_bye_jarvis(self) -> None:
        if not self.visible or self.window is None:
            return
        print("[Widget] 'Bye Jarvis' detected — hiding widget.")
        self.visible = False
        self._bye_stop_evt.set()
        try:
            self.window.hide()
        except Exception as e:
            print(f"[Widget] hide() failed: {e}")

    # ── mic feed (called from the sounddevice callback thread) ───────────

    def feed_audio(self, frame) -> None:
        with self._lock:
            self._ring.append(frame.copy())

    def _snapshot_audio(self):
        import numpy as np
        with self._lock:
            if not self._ring:
                return None
            chunks = list(self._ring)
        return np.concatenate(chunks).flatten()

    def _bye_listener_loop(self) -> None:
        """Runs only while the widget is visible. Polls the rolling audio
        buffer every BYE_POLL_S seconds and transcribes it — cheap relative
        to running Whisper continuously, since it's gated to the one window
        where the phrase actually matters."""
        if self._whisper is None:
            from core.stt import WhisperSTT
            print("[Widget] Loading local speech model for 'bye jarvis' (one-time)…")
            try:
                self._whisper = WhisperSTT(model_name="tiny", language="en")
            except Exception as e:
                print(f"[Widget] Could not load Whisper — 'bye jarvis' won't work "
                      f"this run (use the ✕ button instead): {e}")
                return

        while not self._bye_stop_evt.wait(BYE_POLL_S):
            audio = self._snapshot_audio()
            if audio is None or audio.size < SAMPLE_RATE:
                continue
            try:
                float_audio = audio.astype("float32") / 32768.0
                text = self._whisper.transcribe(float_audio).lower()
            except Exception as e:
                print(f"[Widget] Transcription error: {e}")
                continue
            if "bye" in text and "jarvis" in text:
                self.on_bye_jarvis()
                return


def _js_api(controller: WidgetController):
    class Api:
        def say_bye(self):
            controller.on_bye_jarvis()

        def set_mode(self, mode):
            """Called when the user picks full / compact / mini on the widget.
            The page has already switched its layout; this resizes the real
            window to match and remembers the choice."""
            if mode not in WIDGET_SIZES:
                return
            controller.mode = mode
            _apply_size(controller)

        def set_scale(self, scale):
            """Called when the user picks a size preset. The page has already
            zoomed itself by `scale`; this resizes the real window to match."""
            controller.scale = _nearest_scale(scale)
            _apply_size(controller)
            return controller.scale

        def show_menu(self):
            """Right-click / "⋯". Returns False where there's no native menu,
            so the page falls back to stepping through the size presets."""
            if not IS_MAC:
                return False
            from PyObjCTools import AppHelper
            AppHelper.callAfter(_show_mac_menu, controller)
            return True
    return Api()


def _apply_size(controller) -> None:
    """Resize the window to its layout × scale — pinned at the bottom-right
    corner so a widget docked there stays put — and remember the choice."""
    try:
        from webview.window import FixPoint   # deferred: see the `import webview` note in main()
        width, height, radius = _scaled_size(controller.mode, controller.scale)
        controller.window.resize(width, height, fix_point=FixPoint.SOUTH | FixPoint.EAST)
        if IS_MAC:
            _mac_chrome_later(controller.window, radius)
        _save_prefs(controller.mode, controller.scale)
    except Exception as e:
        print(f"[Widget] Resize failed: {e}")


def _start_control_server(controller: WidgetController) -> None:
    """A tiny loopback-only HTTP server so a button click elsewhere on this
    machine (the main JARVIS window's "LAUNCH ARC SENTINEL" button) can pop
    this exact widget instance open or closed on demand, instead of only
    reacting to 'Hey Jarvis' / 'bye jarvis'. Binds 127.0.0.1 only, same
    reasoning as core/local_control.py: nothing off-machine can ever reach
    it, and anything already running locally already has full mic/filesystem
    access, so a token here would protect against nothing real."""

    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass

        def do_GET(self) -> None:
            if self.path == "/status":
                self._json({"visible": controller.visible})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self.path == "/show":
                controller.on_hey_jarvis()
            elif self.path == "/hide":
                controller.on_bye_jarvis()
            elif self.path == "/toggle":
                (controller.on_bye_jarvis if controller.visible else controller.on_hey_jarvis)()
            else:
                self._json({"error": "not found"}, 404)
                return
            self._json({"visible": controller.visible})

        def log_message(self, *_args) -> None:
            pass   # keep the console clean — same as core/local_control.py

    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", CONTROL_PORT), Handler)
    except OSError as e:
        print(f"[Widget] Control port {CONTROL_PORT} unavailable ({e}) — "
              "the main app's button trigger won't reach this instance "
              "(voice trigger still works).")
        return
    threading.Thread(target=httpd.serve_forever, daemon=True, name="WidgetControl").start()


def main() -> None:
    if not _check_deps():
        sys.exit(1)
    if not _check_singleton():
        sys.exit(0)

    import atexit
    atexit.register(lambda: PID_FILE.unlink(missing_ok=True))

    import webview
    import sounddevice as sd
    from core.wake_word import WakeWordDetector
    from core import audio_devices
    from memory.config_manager import get_input_device

    # Reuse the SAME device-selection JARVIS itself uses (core/audio_devices.py
    # — the module that measures which host API a device actually opens
    # cleanly at, rather than guessing) and the SAME saved device name from
    # ⚙ → AUDIO DEVICES, instead of letting sounddevice grab whatever it calls
    # "default". Two processes reasoning about "the mic" differently is a
    # likelier source of conflicts than two processes sharing one mic in
    # WASAPI shared mode ever is. Kicked off now, in the background, so the
    # cache is warm well before start_backend() needs it (wake-model loading
    # alone already takes several seconds).
    audio_devices.configure(SAMPLE_RATE, SAMPLE_RATE)
    audio_devices.prefetch()

    controller = WidgetController()

    def on_wake_detected():
        controller.on_hey_jarvis()

    detector = WakeWordDetector(on_detect=on_wake_detected, logger=lambda m: print(f"[Widget] {m}"))

    def mic_callback(indata, frames, time_info, status):
        detector.feed(indata)
        if controller.visible:
            controller.feed_audio(indata[:, 0] if indata.ndim > 1 else indata)

    # The stream MUST stay referenced for the life of the process. If it is
    # garbage-collected, its cffi callback trampoline is freed while PortAudio's
    # IO thread is still calling it, and the process segfaults (SIGSEGV in
    # ffi_closure_SYSV_inner on com.apple.audio.IOThread.client).
    mic_streams = []

    def _open_mic(dev):
        s = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=CHUNK_SIZE, device=dev, callback=mic_callback,
        )
        s.start()
        mic_streams.append(s)
        return s

    def start_backend():
        if not detector.start():
            print("[Widget] Wake-word model failed to load — the widget will never appear.")
            return

        mic_name = get_input_device()
        mic_dev  = audio_devices.resolve(mic_name, "input")
        try:
            _open_mic(mic_dev)
            print(f"[Widget] Listening for 'Hey Jarvis' on "
                  f"'{mic_name or 'system default'}'. Ctrl+C in this console to stop.")
        except Exception as e:
            # Same fallback main.py's own _listen_audio uses: a device that's
            # listed but momentarily refuses to open (exclusive mode, JARVIS
            # itself holding it, a driver hiccup) must not mean the widget can
            # never hear "Hey Jarvis" at all.
            if mic_dev is None:
                print(f"[Widget] Could not open the microphone: {e}")
                print("[Widget] If JARVIS's own session is holding it exclusively, "
                      "try again once it's idle, or pick a different input device "
                      "in ⚙ → AUDIO DEVICES (both this widget and JARVIS use that choice).")
                return
            print(f"[Widget] Mic '{mic_name}' unavailable ({e}) — trying system default…")
            try:
                _open_mic(None)
                print("[Widget] Listening for 'Hey Jarvis' on the system default microphone.")
            except Exception as e2:
                print(f"[Widget] Could not open any microphone: {e2}")

    # ── window: frameless, always-on-top, docked bottom-right, hidden until
    #    'Hey Jarvis' fires (or the main app's button calls /show). Opens at
    #    whichever size the user last picked. On macOS it's transparent, with
    #    native frosted glass added behind the page once it has loaded. ─────
    if IS_MAC:
        _install_mac_overlay_window()

    mode, scale = _load_prefs()
    controller.mode, controller.scale = mode, scale
    width, height, radius = _scaled_size(mode, scale)
    margin = 24
    x = y = None
    try:
        screen = webview.screens[0]
        x = screen.x + screen.width  - width  - margin
        y = screen.y + screen.height - height - margin
    except Exception:
        pass   # no screen info — fall back to pywebview's own default placement

    from urllib.parse import quote
    page_url = (f"{WIDGET_DIR / 'arc_sentinel_widget.html'}"
                f"?accent={quote(_get_accent_hex())}&mode={mode}&scale={scale}&mac={int(IS_MAC)}")

    # min_size must allow the smallest mode, or the native window would clamp it.
    mini_w, mini_h, _ = _scaled_size("mini", min(sc for _, sc in WIDGET_SCALES))
    window = webview.create_window(
        "Arc Sentinel",
        url=page_url,
        width=width, height=height, x=x, y=y,
        frameless=True, on_top=True, easy_drag=True, resizable=False,
        min_size=(mini_w, mini_h),
        transparent=IS_MAC, background_color="#000000" if IS_MAC else "#111318",
        js_api=_js_api(controller),
        hidden=True,
    )
    controller.window = window
    _start_control_server(controller)
    window.events.loaded += lambda: threading.Thread(target=start_backend, daemon=True).start()
    if IS_MAC:
        window.events.loaded += lambda: _mac_chrome_later(window, radius)
        window.events.loaded += lambda: _mac_pin_later(window)
    webview.start()


if __name__ == "__main__":
    main()
