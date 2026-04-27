"""
SQLite-история запросов к /recommend и /predict.

Таблица requests:
  id, ts, endpoint, job_type, request_json, response_json, duration_ms, api_key_hint
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .settings import settings


def _db_path() -> Path:
    p = settings.db_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def _conn():
    con = sqlite3.connect(_db_path(), check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           REAL NOT NULL,
                endpoint     TEXT NOT NULL,
                job_type     TEXT,
                request_json TEXT,
                response_json TEXT,
                duration_ms  REAL,
                api_key_hint TEXT
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON requests(ts)")


def log_request(
    endpoint: str,
    job_type: str | None,
    request_body: dict,
    response_body: dict,
    duration_ms: float,
    api_key_hint: str | None = None,
) -> None:
    with _conn() as con:
        con.execute(
            """INSERT INTO requests
               (ts, endpoint, job_type, request_json, response_json, duration_ms, api_key_hint)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                endpoint,
                job_type,
                json.dumps(request_body, ensure_ascii=False),
                json.dumps(response_body, ensure_ascii=False),
                round(duration_ms, 1),
                api_key_hint,
            ),
        )


def get_history(limit: int = 50, offset: int = 0) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """SELECT id, ts, endpoint, job_type, duration_ms, api_key_hint,
                      request_json, response_json
               FROM requests ORDER BY ts DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
    items: list[dict] = []
    for row in rows:
        item = dict(row)
        item["ts_ms"] = int(float(item["ts"]) * 1000)
        item["ts_iso"] = datetime.fromtimestamp(float(item["ts"]), timezone.utc).isoformat()
        for key in ("request_json", "response_json"):
            raw = item.get(key)
            try:
                item[key] = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                item[key] = raw
        items.append(item)
    return items


def get_stats() -> dict:
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        by_ep = con.execute(
            "SELECT endpoint, COUNT(*) as n FROM requests GROUP BY endpoint"
        ).fetchall()
        by_jt = con.execute(
            "SELECT job_type, COUNT(*) as n FROM requests WHERE job_type IS NOT NULL GROUP BY job_type"
        ).fetchall()
        avg_ms = con.execute("SELECT AVG(duration_ms) FROM requests").fetchone()[0]
    return {
        "total_requests": total,
        "by_endpoint": {r["endpoint"]: r["n"] for r in by_ep},
        "by_job_type": {r["job_type"]: r["n"] for r in by_jt},
        "avg_duration_ms": round(avg_ms, 1) if avg_ms else None,
    }
