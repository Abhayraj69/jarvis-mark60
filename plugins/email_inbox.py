"""
Check unread email in the Mac's Mail app (whatever accounts it has).
Read-only: never sends, moves, deletes or marks anything as read.
"""
from core import mac_apps

PLUGIN = {
    "name": "check_email",
    "description": (
        "Check the user's unread email: how many, and the latest senders and subjects. "
        "Use for 'any new email', 'did X reply'. Read-only — cannot send or delete."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "limit": {"type": "INTEGER", "description": "How many recent unread to list (default 5)"},
        },
        "required": [],
    },
}


def run(parameters: dict, player=None, session_memory=None) -> str:
    try:
        limit = max(1, min(int((parameters or {}).get("limit") or 5), 15))
    except (TypeError, ValueError):
        limit = 5
    try:
        total, mails = mac_apps.unread_mail(limit)
    except mac_apps.Unavailable as e:
        return f"Could not check email: {e}"
    except Exception as e:
        return f"Could not check email: {e}"
    return mac_apps.describe_mail(total, mails)
