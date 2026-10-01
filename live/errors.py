"""Reconnect signals and classification of Live session failures."""

from live.constants import _IDLE_DROP_MIN_UPTIME

class _ReconnectSignal(Exception):
    """Raised inside the session TaskGroup to force a clean, voluntary reconnect
    (e.g. the user picked a new voice — the voice is fixed at connect time, so
    the session must be rebuilt).

    Carries `keep_context`: True for an ordinary rebuild, where the stored
    resumption handle is replayed and the conversation continues; False when the
    new session must genuinely start clean (see the voice-change note in
    _on_voice_change)."""

    def __init__(self, keep_context: bool = True):
        super().__init__()
        self.keep_context = keep_context


def _is_reconnect_signal(exc: BaseException) -> bool:
    """True if `exc` is a _ReconnectSignal, or a(n) (Base)ExceptionGroup that
    wraps one — TaskGroup bundles child exceptions into a group."""
    if isinstance(exc, _ReconnectSignal):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_reconnect_signal(sub) for sub in exc.exceptions)
    return False


def _keep_context_of(exc: BaseException) -> bool:
    """Read `keep_context` off a reconnect signal, unwrapping the group the
    TaskGroup put it in. Defaults to True: an unexpected shape must not silently
    wipe the conversation."""
    if isinstance(exc, _ReconnectSignal):
        return getattr(exc, "keep_context", True)
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            if _is_reconnect_signal(sub):
                return _keep_context_of(sub)
    return True


def _is_idle_drop(err: str, uptime: float) -> bool:
    """True for the server's idle/abort close (1008) on a session that had been
    up long enough to have been working (see _IDLE_DROP_MIN_UPTIME)."""
    return ("1008" in err or "operation was aborted" in err.lower()) \
        and uptime >= _IDLE_DROP_MIN_UPTIME


def _classify_live_error(err: str, top: str, *, resumed_with: bool, uptime: float,
                         tuned: bool, enhanced: bool) -> str:
    """What the run loop should do about a Live session failure. `err` is the
    full text including every exception inside a TaskGroup; `top` is str() of
    the outer exception alone. The model ladder (quota / 1011) is decided
    separately, between "idle_drop" and "drop_tuning", because it is stateful.

    Returns one of: bad_handle, idle_drop, drop_tuning, drop_proactive,
    bad_key, network, other.
    """
    low = err.lower()
    # A resumption handle the server will not accept — expired, or belonging
    # to a session it has since dropped. Without this, the same dead handle
    # would be replayed on every retry and the assistant would never come back
    # at all. Drop it once and let the next attempt start clean.
    if resumed_with and (
        "resum" in top.lower() or "handle" in top.lower()
        or "INVALID_ARGUMENT" in top or "NOT_FOUND" in top
    ):
        return "bad_handle"
    # Idle drop (1008) on a session that had been working. Nothing is wrong
    # with the model or the network, and the resumption handle survives it.
    if _is_idle_drop(err, uptime):
        return "idle_drop"
    # Turn-taking / media / thinking knobs rejected (preview API drift) —
    # dropped first, because they are the newest fields and cheapest to lose.
    if tuned and (
        "INVALID_ARGUMENT" in err or "Unknown name" in err
        or "unexpected keyword" in err or "realtime_input" in low
        or "media_resolution" in low or "thinking" in low
    ):
        return "drop_tuning"
    if enhanced and (
        "INVALID_ARGUMENT" in err or "proactiv" in low
        or "Unknown name" in err or "unexpected keyword" in err
    ):
        return "drop_proactive"
    if "API key not valid" in err or "1007" in err:
        return "bad_key"
    if any(k in err for k in (
        "TimeoutError", "timed out", "getaddrinfo", "CancelledError",
        "ConnectionRefusedError", "OSError", "Cannot connect",
        # 1006: the socket died without a goodbye — Wi-Fi dropped, or the Mac slept.
        "1006", "abnormal closure", "no close frame",
    )):
        return "network"
    return "other"
