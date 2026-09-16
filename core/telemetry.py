"""
core/telemetry.py — per-turn latency and cost telemetry.

WHY THIS EXISTS
    The last two commits (fast-intent shortcuts, adaptive polling backoff)
    were both written to cut latency, but nothing in main.py ever measured
    latency to begin with — no perf_counter, no per-tool timing, no record of
    which backend served a turn or how many tokens it cost. The choice of
    Cloud vs. Local vs. a fast-intent shortcut was a feeling, not a number.
    This module is the number: one row per exchange (backend, time to first
    spoken/played audio, total turn time, each tool's name and duration,
    tokens in/out, whether fast-intent short-circuited, whether the user
    interrupted), queryable as percentiles and averages via summary().

USAGE
    turn = telemetry.start_turn("local")
    turn.mark("model_done")               # arbitrary named timestamps, ms from turn start
    with turn.tool_span("weather_report"):
        ...                               # timed automatically
    turn.tokens(tokens_in=120, tokens_out=40)
    turn.set_fast_intent()                # or turn.set_interrupted()
    turn.finish()                         # writes the row; never raises

STORAGE
    Follows the memory/workflow.db pattern (core/predictive_assistant.py):
    module-level DB_PATH under get_base_dir(), ensure_schema() run on every
    connect so a fresh DB just works, and a migration script wrapper in
    memory/ for anyone who wants an explicit up-front step. All writes run on
    a short-lived background thread from Turn.finish() — a telemetry failure
    (a locked DB, a full disk) must never raise into the session loop that's
    busy speaking to the user.
"""
from __future__ import annotations

import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
DB_PATH  = BASE_DIR / "memory" / "telemetry.db"

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at              TEXT    NOT NULL,
    backend                 TEXT    NOT NULL,
    fast_intent             INTEGER NOT NULL DEFAULT 0,
    interrupted             INTEGER NOT NULL DEFAULT 0,
    time_to_first_audio_ms  REAL,
    model_done_ms           REAL,
    total_ms                REAL,
    tokens_in               INTEGER,
    tokens_out              INTEGER
);
CREATE INDEX IF NOT EXISTS idx_turns_started_at ON turns(started_at);
CREATE INDEX IF NOT EXISTS idx_turns_backend    ON turns(backend);

CREATE TABLE IF NOT EXISTS tool_spans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id     INTEGER NOT NULL REFERENCES turns(id),
    tool_name   TEXT    NOT NULL,
    duration_ms REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_spans_turn_id ON tool_spans(turn_id);

-- A tool returned ok=false and nothing JARVIS said afterwards admitted it
-- (see core/result_contract.py FalseSuccessTracker). Its own table rather
-- than a turns column because the admission, when it comes, often lands in
-- a later turn than the tool call.
CREATE TABLE IF NOT EXISTS false_successes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    tools       TEXT    NOT NULL,
    spoken      TEXT    NOT NULL
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


class Turn:
    """One in-flight exchange. Created by start_turn(); every method is
    best-effort and safe to call from either the async session loop or a
    background thread. finish() is the only method that touches disk, and it
    does so off-thread so a slow/locked DB never adds latency to a turn."""

    def __init__(self, backend: str, db_path: Optional[Path] = None):
        self.backend      = backend
        self._db_path     = db_path
        self._t0          = time.monotonic()
        self._marks: dict[str, float] = {}
        self._tool_spans: list[tuple[str, float]] = []
        self._tokens_in:  Optional[int] = None
        self._tokens_out: Optional[int] = None
        self._fast_intent = False
        self._interrupted = False
        self._finished    = False

    def mark(self, name: str) -> None:
        """Record `name` at the elapsed ms since the turn started. Calling it
        again with the same name overwrites — only the first meaningful call
        site should invoke it (e.g. "first_audio" only once per turn)."""
        self._marks[name] = (time.monotonic() - self._t0) * 1000

    def mark_once(self, name: str) -> None:
        """Like mark(), but a no-op if `name` was already recorded — for
        marks (like first_audio) that only mean something the first time."""
        if name not in self._marks:
            self.mark(name)

    @contextmanager
    def tool_span(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._tool_spans.append((name, (time.monotonic() - t0) * 1000))

    def add_tool_span(self, name: str, duration_ms: float) -> None:
        """Record an already-measured span — for work that ran outside the
        receive loop (a background `think` call) whose label is only known
        once it finishes (e.g. "think:claude"). If the turn has already been
        written, the span is attached to the latest stored turn instead so it
        is never lost (see record_late_span)."""
        if self._finished:
            record_late_span(name, duration_ms, db_path=self._db_path)
            return
        self._tool_spans.append((name, float(duration_ms)))

    @property
    def finished(self) -> bool:
        return self._finished

    def tokens(self, tokens_in: Optional[int] = None, tokens_out: Optional[int] = None) -> None:
        if tokens_in is not None:
            self._tokens_in = tokens_in
        if tokens_out is not None:
            self._tokens_out = tokens_out

    def set_fast_intent(self, value: bool = True) -> None:
        self._fast_intent = value

    def set_interrupted(self, value: bool = True) -> None:
        self._interrupted = value

    def finish(self) -> None:
        """Snapshot elapsed time and hand the write off to a daemon thread.
        Safe to call more than once (later calls are ignored)."""
        if self._finished:
            return
        self._finished = True
        total_ms = (time.monotonic() - self._t0) * 1000
        threading.Thread(target=self._write, args=(total_ms,), daemon=True).start()

    def _write(self, total_ms: float) -> None:
        try:
            with _lock:
                conn = _connect(self._db_path)
                try:
                    cur = conn.execute(
                        "INSERT INTO turns (started_at, backend, fast_intent, interrupted, "
                        "time_to_first_audio_ms, model_done_ms, total_ms, tokens_in, tokens_out) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            datetime.now().isoformat(timespec="seconds"),
                            self.backend,
                            1 if self._fast_intent else 0,
                            1 if self._interrupted else 0,
                            self._marks.get("first_audio"),
                            self._marks.get("model_done"),
                            total_ms,
                            self._tokens_in,
                            self._tokens_out,
                        ),
                    )
                    turn_id = cur.lastrowid
                    conn.executemany(
                        "INSERT INTO tool_spans (turn_id, tool_name, duration_ms) VALUES (?, ?, ?)",
                        [(turn_id, name, dur) for name, dur in self._tool_spans],
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            print(f"[Telemetry] write failed (non-fatal): {e}")


def start_turn(backend: str, db_path: Optional[Path] = None) -> Turn:
    return Turn(backend, db_path)


def record_late_span(name: str, duration_ms: float, db_path: Optional[Path] = None) -> None:
    """Attach a span to the most recently written turn — for background work
    (a `think` call) that outlives the turn it started in. Off-thread,
    best-effort."""
    def _do():
        try:
            with _lock:
                conn = _connect(db_path)
                try:
                    row = conn.execute("SELECT id FROM turns ORDER BY id DESC LIMIT 1").fetchone()
                    if row is None:
                        return
                    conn.execute(
                        "INSERT INTO tool_spans (turn_id, tool_name, duration_ms) VALUES (?, ?, ?)",
                        (row["id"], name, float(duration_ms)),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            print(f"[Telemetry] late span write failed (non-fatal): {e}")
    threading.Thread(target=_do, daemon=True).start()


def record_false_success(tools: list[str], spoken: str, db_path: Optional[Path] = None) -> None:
    """Persist one hallucinated-success event. Off-thread and best-effort,
    like Turn.finish(): never raises into the session loop."""
    def _do():
        try:
            with _lock:
                conn = _connect(db_path)
                try:
                    conn.execute(
                        "INSERT INTO false_successes (ts, tools, spoken) VALUES (?, ?, ?)",
                        (datetime.now().isoformat(timespec="seconds"), ",".join(tools), spoken[:1000]),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            print(f"[Telemetry] false_success write failed (non-fatal): {e}")
    threading.Thread(target=_do, daemon=True).start()


def _percentile(sorted_vals: list[float], pct: float) -> Optional[float]:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summary(days: int = 7, db_path: Optional[Path] = None) -> dict:
    """Per-backend p50/p95 time-to-first-audio, fast-intent hit rate,
    interrupt rate, and token totals over the trailing `days` days, plus
    average duration per tool across all backends."""
    conn = _connect(db_path)
    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        rows = conn.execute(
            "SELECT * FROM turns WHERE started_at >= ? ORDER BY started_at", (cutoff,)
        ).fetchall()

        by_backend: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            by_backend.setdefault(r["backend"], []).append(r)

        backends: dict[str, dict] = {}
        for backend, rs in by_backend.items():
            ttfa = sorted(r["time_to_first_audio_ms"] for r in rs if r["time_to_first_audio_ms"] is not None)
            n = len(rs)
            backends[backend] = {
                "turns":                n,
                "p50_time_to_first_audio_ms": _percentile(ttfa, 50),
                "p95_time_to_first_audio_ms": _percentile(ttfa, 95),
                "tokens_in":             sum(r["tokens_in"]  or 0 for r in rs),
                "tokens_out":            sum(r["tokens_out"] or 0 for r in rs),
                "fast_intent_hit_rate":  (sum(1 for r in rs if r["fast_intent"]) / n) if n else 0.0,
                "interrupted_rate":      (sum(1 for r in rs if r["interrupted"]) / n) if n else 0.0,
            }

        tool_rows = conn.execute(
            "SELECT tool_name, AVG(duration_ms) AS avg_ms, COUNT(*) AS n "
            "FROM tool_spans JOIN turns ON turns.id = tool_spans.turn_id "
            "WHERE turns.started_at >= ? GROUP BY tool_name ORDER BY tool_name",
            (cutoff,),
        ).fetchall()
        tools = {r["tool_name"]: {"avg_ms": r["avg_ms"], "count": r["n"]} for r in tool_rows}

        fs_rows = conn.execute(
            "SELECT tools FROM false_successes WHERE ts >= ?", (cutoff,)
        ).fetchall()
        false_by_tool: dict[str, int] = {}
        for r in fs_rows:
            for t in (r["tools"] or "").split(","):
                if t:
                    false_by_tool[t] = false_by_tool.get(t, 0) + 1

        return {"days": days, "backends": backends, "tools": tools,
                "false_successes": {"count": len(fs_rows), "by_tool": false_by_tool}}
    finally:
        conn.close()
