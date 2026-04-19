"""
metrics.py  —  lightweight request telemetry for Orbit.

Why sqlite and not an in-memory dict?
-------------------------------------
Orbit runs with `--workers 4` (see Procfile). Each uvicorn worker is a
separate process with its own memory, so an in-memory counter would give
four different views of the world depending on which worker happened to
serve the admin page. sqlite in WAL mode is the cheapest shared-state
option that still lets every worker read+write safely.

What's recorded
---------------
Every non-static request that passes through the middleware:
  endpoint     — canonical path ("/suburb/{name}" not "/suburb/Bondi")
  status       — HTTP status code
  latency_ms   — measured in the middleware
  client_hash  — first 16 chars of SHA256(XFF || user-agent)
  cache_hit    — True when /match served from the in-process LRU
  error        — truncated error string (non-2xx only)
  ts           — unix seconds (float)

What it's NOT
-------------
Not an analytics pipeline. Not PII-safe beyond the client hash.
Not a replacement for real observability. Designed for a 300-person
live-demo dashboard and nothing more.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Storage path — lives in data/ alongside the parquet features so it rides
# along with any existing volume/backup setup. Overridable via env so tests
# can point at /tmp.
# ---------------------------------------------------------------------------
_DB_PATH = Path(os.environ.get(
    "ORBIT_METRICS_DB",
    str(Path(__file__).parent.parent / "data" / "metrics.db"),
))

# Per-process connection. sqlite3.Connection is NOT thread-safe by default,
# so we set check_same_thread=False and rely on sqlite's own locking +
# WAL mode to serialise writes. Reads never block writes under WAL.
_CONN: Optional[sqlite3.Connection] = None

# Trim the events table to this many rows on each write to keep the db small
# during long-running demos. At ~30 req/s for an hour that's ~100k rows; we
# cap at 50k which is plenty for "last 5 minutes of activity" queries.
_MAX_ROWS = 50_000


def init() -> None:
    """Open the connection, enable WAL, create the schema. Idempotent — safe
    to call once per worker process at startup."""
    global _CONN
    if _CONN is not None:
        return
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(_DB_PATH),
        check_same_thread=False,
        isolation_level=None,  # autocommit; keeps writes atomic per statement
        timeout=5.0,
    )
    # WAL + NORMAL sync = cheap writes, concurrent reads, survives a clean
    # restart. We don't care about a torn write on process kill — worst case
    # the admin page loses the last second of data.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            ts          REAL    NOT NULL,
            endpoint    TEXT    NOT NULL,
            status      INTEGER NOT NULL,
            latency_ms  REAL    NOT NULL,
            client_hash TEXT,
            cache_hit   INTEGER NOT NULL DEFAULT 0,
            error       TEXT
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_endpoint ON events(endpoint);")

    # Cross-worker key/value settings — used for runtime toggles like the AI
    # kill-switch. Lives in the same db so one sqlite connection covers both.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        """
    )
    _CONN = conn


# ---------------------------------------------------------------------------
# Settings — a tiny key/value layer for live runtime toggles (e.g. the admin
# AI kill-switch). Shared across all uvicorn workers via the same sqlite db.
#
# Reads are cached per-process for _SETTINGS_TTL seconds so hot paths don't
# pound sqlite on every request. A 2-second staleness window is fine: the
# only consumer is the admin toggle, where the click-to-effect latency being
# ≤ 2 seconds is imperceptible.
# ---------------------------------------------------------------------------
_SETTINGS_CACHE: Dict[str, tuple] = {}
_SETTINGS_TTL = 2.0


def get_setting(key: str, default: str = "") -> str:
    """Return a setting value (string). Uses a short in-process cache."""
    now = time.time()
    cached = _SETTINGS_CACHE.get(key)
    if cached and (now - cached[1]) < _SETTINGS_TTL:
        return cached[0]
    if _CONN is None:
        return default
    try:
        row = _CONN.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        val = row[0] if row else default
        _SETTINGS_CACHE[key] = (val, now)
        return val
    except Exception:
        return default


def set_setting(key: str, value: str) -> None:
    """Upsert a setting. Invalidates the local cache immediately so the calling
    worker sees the new value on its next read (other workers see it within
    _SETTINGS_TTL seconds)."""
    if _CONN is None:
        return
    try:
        _CONN.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE "
            "SET value = excluded.value, updated_at = excluded.updated_at",
            (key, value, time.time()),
        )
        _SETTINGS_CACHE[key] = (value, time.time())
    except Exception:
        pass


def client_hash(xff: Optional[str], ua: Optional[str]) -> str:
    """Build a stable, non-identifying client key. First 16 hex chars of
    SHA256 over (forwarded-for || user-agent). Good enough to count unique
    devices in a room without storing anything recoverable."""
    raw = f"{xff or ''}|{ua or ''}".encode("utf-8", "ignore")
    return hashlib.sha256(raw).hexdigest()[:16]


def record(
    endpoint: str,
    status: int,
    latency_ms: float,
    client: Optional[str] = None,
    cache_hit: bool = False,
    error: Optional[str] = None,
) -> None:
    """Record one event. Never raises — a telemetry bug must not break
    the actual request path."""
    if _CONN is None:
        return
    try:
        _CONN.execute(
            "INSERT INTO events (ts, endpoint, status, latency_ms, client_hash, cache_hit, error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                endpoint,
                int(status),
                float(latency_ms),
                client,
                1 if cache_hit else 0,
                (error[:500] if error else None),
            ),
        )
        # Opportunistic trim: only check occasionally to avoid a COUNT on every
        # write. A rough ts-based prune catches long demos without blocking.
        if int(time.time()) % 60 == 0:
            _trim()
    except Exception:
        # Metrics are best-effort. Swallow silently.
        pass


def _trim() -> None:
    """Keep the events table bounded. Deletes the oldest rows past _MAX_ROWS."""
    if _CONN is None:
        return
    try:
        _CONN.execute(
            "DELETE FROM events WHERE ts IN ("
            " SELECT ts FROM events ORDER BY ts ASC "
            " LIMIT MAX(0, (SELECT COUNT(*) FROM events) - ?)"
            ")",
            (_MAX_ROWS,),
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Aggregations consumed by the admin dashboard
# ---------------------------------------------------------------------------
def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * pct
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return float(values[f])
    return float(values[f] + (values[c] - values[f]) * (k - f))


def get_timeseries(window_minutes: int = 60, bucket_seconds: int = 60) -> List[Dict]:
    """Return per-bucket `{t, req, users}` for the last `window_minutes`.

    Buckets are aligned to `bucket_seconds` boundaries (so 60s buckets line
    up with wall-clock minutes). Empty buckets are zero-filled so the graph
    x-axis stays continuous — a missing minute shouldn't collapse the next
    minute into its slot.

    `users` is COUNT(DISTINCT client_hash) inside each bucket, so the same
    visitor hitting the site 20 times in one minute still counts as 1 for
    that minute. Across buckets they count separately, which is desired —
    "active users this minute" is the meaningful stat for a live room.
    """
    if _CONN is None:
        return []

    now = time.time()
    cutoff = now - window_minutes * 60
    n_buckets = (window_minutes * 60) // bucket_seconds  # e.g. 60 for 60m/60s

    try:
        c = _CONN.cursor()
        rows = c.execute(
            # CAST(ts/bucket AS INT)*bucket = bucket-start timestamp.
            # bucket_seconds is a Python int we injected, not user input, so
            # string-building the SQL is safe here.
            f"SELECT "
            f"  CAST(ts / {int(bucket_seconds)} AS INTEGER) * {int(bucket_seconds)} AS bt, "
            f"  COUNT(*), "
            f"  COUNT(DISTINCT client_hash) "
            f"FROM events "
            f"WHERE ts >= ? "
            f"GROUP BY bt "
            f"ORDER BY bt ASC",
            (cutoff,),
        ).fetchall()

        by_t = {int(bt): (int(rc), int(uu)) for bt, rc, uu in rows}

        # End at the bucket containing "now" so the rightmost bar is the
        # current partial minute.
        end_bucket = (int(now) // bucket_seconds) * bucket_seconds
        series: List[Dict] = []
        for i in range(n_buckets):
            t = end_bucket - (n_buckets - 1 - i) * bucket_seconds
            rc, uu = by_t.get(t, (0, 0))
            series.append({"t": int(t), "req": rc, "users": uu})
        return series
    except Exception:
        return []


def get_stats() -> Dict:
    """Return a single snapshot of admin-dashboard-relevant numbers.

    Shape is stable — the admin template reads specific keys so changes here
    should update orbit_admin_template() in api.py too.
    """
    if _CONN is None:
        return _empty_stats()

    now = time.time()
    since_5m = now - 5 * 60
    since_1h = now - 60 * 60

    try:
        c = _CONN.cursor()

        # Totals across the full retained window (up to _MAX_ROWS rows).
        row = c.execute(
            "SELECT COUNT(*), COUNT(DISTINCT client_hash), MIN(ts) FROM events"
        ).fetchone()
        total_requests, unique_users_all, first_seen = row or (0, 0, None)

        # Last 5 minutes: req/min + unique users (what a demo operator actually
        # cares about — is the room actively using it right now?).
        row = c.execute(
            "SELECT COUNT(*), COUNT(DISTINCT client_hash) FROM events WHERE ts >= ?",
            (since_5m,),
        ).fetchone()
        req_last_5m, unique_users_5m = row or (0, 0)
        req_per_min_last_5 = round((req_last_5m or 0) / 5.0, 1)

        # Last hour unique users — captures the room even after someone walks
        # away from the screen for a couple of minutes.
        row = c.execute(
            "SELECT COUNT(DISTINCT client_hash) FROM events WHERE ts >= ?",
            (since_1h,),
        ).fetchone()
        unique_users_1h = (row[0] if row else 0) or 0

        # Per-endpoint breakdown (last hour) — which surface is getting hit.
        endpoints = [
            {"endpoint": ep, "count": int(n)}
            for ep, n in c.execute(
                "SELECT endpoint, COUNT(*) FROM events "
                "WHERE ts >= ? GROUP BY endpoint ORDER BY COUNT(*) DESC",
                (since_1h,),
            ).fetchall()
        ]

        # Cache hit rate — only meaningful for /match, which is the one
        # endpoint that sets cache_hit=1.
        row = c.execute(
            "SELECT COUNT(*), SUM(cache_hit) FROM events "
            "WHERE endpoint = '/match' AND ts >= ?",
            (since_1h,),
        ).fetchone()
        match_total, match_hits = row or (0, 0)
        match_total = match_total or 0
        match_hits = match_hits or 0
        cache_hit_rate = (
            round(100.0 * match_hits / match_total, 1) if match_total else 0.0
        )

        # Latency over the last 5 minutes across all tracked endpoints.
        lats = [r[0] for r in c.execute(
            "SELECT latency_ms FROM events WHERE ts >= ?",
            (since_5m,),
        ).fetchall()]
        p50 = round(_percentile(lats, 0.50), 1)
        p95 = round(_percentile(lats, 0.95), 1)

        # Last few errors — raw string, status code, endpoint, age.
        errors = [
            {
                "ts": float(ts),
                "age_s": round(now - float(ts), 1),
                "endpoint": ep,
                "status": int(st),
                "error": (err or "")[:200],
            }
            for ts, ep, st, err in c.execute(
                "SELECT ts, endpoint, status, error FROM events "
                "WHERE status >= 400 ORDER BY ts DESC LIMIT 5"
            ).fetchall()
        ]

        timeseries = get_timeseries(window_minutes=60, bucket_seconds=60)

        return {
            "generated_at":    now,
            "window_start":    float(first_seen) if first_seen else None,
            "total_requests":  int(total_requests or 0),
            "unique_users_all": int(unique_users_all or 0),
            "unique_users_5m":  int(unique_users_5m or 0),
            "unique_users_1h":  int(unique_users_1h or 0),
            "req_per_min_last_5": req_per_min_last_5,
            "req_last_5m":     int(req_last_5m or 0),
            "endpoints":       endpoints,
            "cache_hit_rate":  cache_hit_rate,
            "match_requests_1h": int(match_total),
            "match_cache_hits_1h": int(match_hits),
            "p50_ms":          p50,
            "p95_ms":          p95,
            "errors":          errors,
            "timeseries":      timeseries,
        }
    except Exception as exc:
        return _empty_stats(error=str(exc))


def _empty_stats(error: Optional[str] = None) -> Dict:
    """Shape-stable zero payload for when the db isn't initialised or a read
    blew up — keeps the admin template from branching."""
    return {
        "generated_at":       time.time(),
        "window_start":       None,
        "total_requests":     0,
        "unique_users_all":   0,
        "unique_users_5m":    0,
        "unique_users_1h":    0,
        "req_per_min_last_5": 0.0,
        "req_last_5m":        0,
        "endpoints":          [],
        "cache_hit_rate":     0.0,
        "match_requests_1h":  0,
        "match_cache_hits_1h": 0,
        "p50_ms":             0.0,
        "p95_ms":             0.0,
        "errors":             [],
        "timeseries":         [],
        "init_error":         error,
    }
