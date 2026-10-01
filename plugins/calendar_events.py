"""
Read your calendar (macOS Calendar accounts: Google, iCloud, Exchange…).
Read-only. See core/mac_apps.py for how and what permission it needs.
"""
from core import mac_apps

PLUGIN = {
    "name": "check_calendar",
    "description": (
        "Read the user's calendar: today's, tomorrow's or this week's meetings and events. "
        "Use for 'what's on my calendar', 'am I free at 3', 'what's my day like'. Read-only."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "day": {"type": "STRING", "description": "today | tomorrow | week (default today)"},
        },
        "required": [],
    },
}

_RANGES = {"today": (0, 1, "today"), "tomorrow": (1, 1, "tomorrow"), "week": (0, 7, "this week")}


def run(parameters: dict, player=None, session_memory=None) -> str:
    day = str((parameters or {}).get("day") or "today").strip().lower()
    offset, days, label = _RANGES.get(day, _RANGES["today"])
    try:
        evs = mac_apps.events(offset, days)
    except mac_apps.Unavailable as e:
        return f"Could not read the calendar: {e}"
    except Exception as e:
        return f"Could not read the calendar: {e}"
    if days > 1 and evs:
        by_day = {}
        for ev in evs:
            by_day.setdefault(ev.start.strftime("%A"), []).append(ev)
        return " ".join(mac_apps.describe_events(v, k) for k, v in by_day.items())
    return mac_apps.describe_events(evs, label)
