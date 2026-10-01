"""
Input guard: limits on JARVIS typing into other people's windows.

computer_control types, pastes and presses keys into whatever window has
focus. Combined with a misheard sentence that is how text ends up sent in a
chat nobody meant to write in. Three rules, all decided here and enforced in
actions/computer_control.py and actions/send_message.py:

  1. Allow-list. Keyboard input goes straight through only to apps on the
     list (editors, browsers, notes, terminals…). Any other app gets an
     on-screen confirmation first — the core/confirm.py gate, which the model
     cannot answer for the user.
  2. Sending needs a yes. Enter / Ctrl+Enter / Cmd+Enter in a messaging app
     (WhatsApp, Slack, Mail…, including their web versions, matched on the
     window title) and every send_message call go behind the same gate. Typing
     a draft is fine; sending it is the irreversible part.
  3. You can see it. While JARVIS is driving the keyboard the HUD says
     ACTING FOR YOU and the OS shows a notification, because the HUD is
     usually behind the window being typed into.

All of it can be tuned or switched off in ⚙ → Plugin Settings → INPUT GUARD.
"""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass

KEYBOARD_ACTIONS = {"type", "smart_type", "paste", "press", "hotkey", "clear_field"}
SEND_KEYS = {"enter", "return"}

DEFAULT_ALLOWED_APPS = [
    # editors / notes / office
    "TextEdit", "Notes", "Code", "Visual Studio Code", "Cursor", "Xcode", "PyCharm",
    "Sublime Text", "Notepad", "Word", "Pages", "Obsidian", "Notion",
    # terminals
    "Terminal", "iTerm", "Warp", "PowerShell", "cmd",
    # browsers (their messaging sites are still caught by window title)
    "Chrome", "Safari", "Firefox", "Arc", "Edge", "Brave", "Opera",
    # media / jarvis itself
    "Spotify", "Music", "Python", "JARVIS",
]
DEFAULT_MESSAGING_APPS = [
    "WhatsApp", "Telegram", "Messages", "Slack", "Discord", "Signal", "Teams",
    "Mail", "Outlook", "Gmail", "Messenger", "Instagram", "Skype", "Zoom",
]


@dataclass
class Settings:
    enabled: bool = True
    allowed_apps: tuple = tuple(DEFAULT_ALLOWED_APPS)
    messaging_apps: tuple = tuple(DEFAULT_MESSAGING_APPS)
    confirm_messages: bool = True


def _split(v, default) -> tuple:
    if isinstance(v, str) and v.strip():
        return tuple(s.strip() for s in v.split(",") if s.strip())
    if isinstance(v, (list, tuple)) and v:
        return tuple(str(s).strip() for s in v if str(s).strip())
    return tuple(default)


def load_settings() -> Settings:
    try:
        from memory.config_manager import get_plugin_config
        cfg = get_plugin_config("input_guard")
    except Exception:
        cfg = {}
    return Settings(
        enabled=bool(cfg.get("enabled", True)),
        allowed_apps=_split(cfg.get("allowed_apps"), DEFAULT_ALLOWED_APPS),
        messaging_apps=_split(cfg.get("messaging_apps"), DEFAULT_MESSAGING_APPS),
        confirm_messages=bool(cfg.get("confirm_messages", True)),
    )


# ── Which window has focus ───────────────────────────────────────────────────

def frontmost() -> tuple[str, str]:
    """(app name, window title) of the focused window; ('', '') if unknown."""
    try:
        if sys.platform == "darwin":
            script = (
                'tell application "System Events"\n'
                '  set p to first application process whose frontmost is true\n'
                '  set t to ""\n'
                '  try\n    set t to name of front window of p\n  end try\n'
                '  return (name of p) & "\\n" & t\n'
                'end tell')
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=3)
            app, _, title = r.stdout.strip().partition("\n")
            return app.strip(), title.strip()
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes as wt
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buf, 512)
            pid = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            app = ""
            try:
                import psutil
                app = psutil.Process(pid.value).name().rsplit(".", 1)[0]
            except Exception:
                pass
            return app, buf.value
    except Exception:
        pass
    return "", ""


# ── Decisions ────────────────────────────────────────────────────────────────

def _matches(name: str, patterns) -> bool:
    n = name.lower()
    return bool(n) and any(p.lower() in n for p in patterns)


def is_send(action: str, params: dict) -> bool:
    if action == "press":
        return str(params.get("key", "enter")).lower().strip() in SEND_KEYS
    if action == "hotkey":
        keys = str(params.get("keys", "")).lower().replace(" ", "")
        return any(k in SEND_KEYS for k in keys.split("+"))
    if action == "screen_click":
        desc = str(params.get("description", "")).lower()
        return any(w in desc for w in ("send", "post", "submit", "reply"))
    return False


def decide(action: str, params: dict, app: str, title: str, s: Settings) -> str:
    """'allow', 'confirm_send' or 'confirm_app'."""
    if not s.enabled:
        return "allow"
    if action not in KEYBOARD_ACTIONS and action != "screen_click":
        return "allow"
    messaging = _matches(app, s.messaging_apps) or _matches(title, s.messaging_apps)
    if is_send(action, params) and messaging:
        return "confirm_send"
    if action == "screen_click":
        return "allow"
    if not app:
        # Could not tell what has focus (no accessibility permission, or an
        # unsupported OS). Refusing everything would break typing entirely.
        return "allow"
    if messaging or _matches(app, s.allowed_apps):
        return "allow"
    return "confirm_app"


# ── Indicator ────────────────────────────────────────────────────────────────

_last_notice = 0.0
NOTICE_EVERY_SECONDS = 15.0


def acting(player, what: str) -> None:
    """Show that JARVIS is driving the keyboard: HUD state and, at most every
    NOTICE_EVERY_SECONDS, an OS notification (the HUD is usually hidden
    behind the window being typed into)."""
    global _last_notice
    try:
        if player is not None and hasattr(player, "set_state"):
            player.set_state("ACTING")
    except Exception:
        pass
    now = time.monotonic()
    if now - _last_notice < NOTICE_EVERY_SECONDS:
        return
    _last_notice = now
    try:
        if sys.platform == "darwin":
            msg = what.replace('"', "'")[:120]
            subprocess.Popen(["osascript", "-e",
                              f'display notification "{msg}" with title "JARVIS is acting for you"'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
