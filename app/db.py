"""SQLite schema, connections and cross-process rebuild-job coordination."""

from __future__ import annotations

import os
import re
import socket
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from .config import SETTINGS
from .accents import VALID_POLICIES

ACTIVE_JOB_STATES = ("queued", "building", "catching_up", "activating")
FTS_NAME_RE = re.compile(r"^fts_gen_\d+$")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    page_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    rev INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS changes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id TEXT NOT NULL,
    page_ref INTEGER,
    op TEXT NOT NULL CHECK (op IN ('upsert', 'delete')),
    title TEXT,
    body TEXT,
    rev INTEGER NOT NULL,
    changed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_changes_seq ON changes(seq);

CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy TEXT NOT NULL CHECK (policy IN ('keep', 'fold')),
    state TEXT NOT NULL,
    fts_table TEXT NOT NULL DEFAULT '',
    base_seq INTEGER NOT NULL DEFAULT 0,
    applied_seq INTEGER NOT NULL DEFAULT 0,
    doc_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    activated_at TEXT,
    finished_at TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS rebuild_jobs (
    id TEXT PRIMARY KEY,
    generation_id INTEGER NOT NULL REFERENCES generations(id),
    from_policy TEXT NOT NULL,
    to_policy TEXT NOT NULL CHECK (to_policy IN ('keep', 'fold')),
    state TEXT NOT NULL,
    claim_token TEXT NOT NULL,
    owner_host TEXT,
    owner_pid INTEGER,
    owner_proc_start TEXT,
    created_at TEXT NOT NULL,
    created_epoch REAL NOT NULL,
    started_at TEXT,
    heartbeat_at REAL,
    finished_at TEXT,
    base_seq INTEGER NOT NULL DEFAULT 0,
    applied_seq INTEGER NOT NULL DEFAULT 0,
    processed_documents INTEGER NOT NULL DEFAULT 0,
    total_documents INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    message TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_rebuild_one_active
ON rebuild_jobs((1))
WHERE state IN ('queued', 'building', 'catching_up', 'activating');
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def fts_table_name(generation_id: int) -> str:
    if not isinstance(generation_id, int) or generation_id <= 0:
        raise ValueError("invalid generation id")
    return f"fts_gen_{generation_id}"


def quote_identifier(identifier: str) -> str:
    if not FTS_NAME_RE.fullmatch(identifier):
        raise ValueError(f"unsafe SQLite identifier: {identifier!r}")
    return f'"{identifier}"'


def ensure_db_parent() -> None:
    SETTINGS.db_path.parent.mkdir(parents=True, exist_ok=True)


def connect(*, query_only: bool = False) -> sqlite3.Connection:
    """Open a configured autocommit SQLite connection.

    Transactions are started explicitly.  WAL mode is enabled by ``init_db``
    and persists in the database file.
    """

    ensure_db_parent()
    timeout = SETTINGS.db_busy_timeout_seconds
    conn = sqlite3.connect(
        str(SETTINGS.db_path),
        timeout=timeout,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    if query_only:
        conn.execute("PRAGMA query_only = ON")
    return conn


def begin_immediate(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")


def _create_gen_fts(conn: sqlite3.Connection, table: str) -> None:
    ident = quote_identifier(table)
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {ident} USING fts5(
            page_id UNINDEXED,
            title,
            body,
            tokenize = 'unicode61 remove_diacritics 0'
        )
        """
    )


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_start_token(pid: int) -> str | None:
    """Linux process start tick, used to detect PID reuse after a restart."""

    try:
        # The comm field can contain spaces/parentheses; split after the last
        # ')'.  Field 22 (starttime) is index 19 in the remaining fields.
        with open(f"/proc/{pid}/stat", "rb") as proc_stat:
            payload = proc_stat.read()
        tail = payload.rsplit(b")", 1)[1].split()
        return tail[19].decode("ascii")
    except Exception:
        return None


def init_db() -> None:
    """Create/upgrade the schema and ensure an active index exists.

    This function is safe to call concurrently from both API workers.  The
    schema transaction is serialised with ``BEGIN IMMEDIATE``.
    """

    ensure_db_parent()
    conn = connect()
    try:
        # journal_mode cannot be changed inside a transaction.  It is
        # idempotent once the database is in WAL mode.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA wal_autocheckpoint = 1000")
        begin_immediate(conn)

        for statement in SCHEMA_SQL.split(";"):
            if statement.strip():
                conn.execute(statement)

        # Forward-compatible with databases created before the FTS rowid
        # optimisation.  New rows always populate page_ref.
        change_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(changes)").fetchall()
        }
        if "page_ref" not in change_columns:
            conn.execute("ALTER TABLE changes ADD COLUMN page_ref INTEGER")

        job_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(rebuild_jobs)").fetchall()
        }
        if "owner_proc_start" not in job_columns:
            conn.execute("ALTER TABLE rebuild_jobs ADD COLUMN owner_proc_start TEXT")

        active_setting = conn.execute(
            "SELECT value FROM settings WHERE key = 'active_generation_id'"
        ).fetchone()
        if active_setting is None:
            now = utc_now()
            cur = conn.execute(
                """
                INSERT INTO generations
                    (policy, state, fts_table, base_seq, applied_seq, doc_count, created_at, activated_at)
                VALUES ('keep', 'active', '', 0, 0, 0, ?, ?)
                """,
                (now, now),
            )
            gen_id = int(cur.lastrowid)
            table = fts_table_name(gen_id)
            _create_gen_fts(conn, table)
            conn.execute(
                "UPDATE generations SET fts_table = ? WHERE id = ?", (table, gen_id)
            )
            conn.execute(
                "INSERT INTO settings(key, value) VALUES('active_generation_id', ?)",
                (str(gen_id),),
            )
            conn.execute(
                "INSERT INTO settings(key, value) VALUES('active_policy', 'keep')"
            )
        else:
            gen_id = int(active_setting["value"])
            gen = conn.execute(
                "SELECT * FROM generations WHERE id = ?", (gen_id,)
            ).fetchone()
            if gen is None:
                raise RuntimeError(f"active generation {gen_id} is missing")
            table = gen["fts_table"] or fts_table_name(gen_id)
            _create_gen_fts(conn, table)
            if not gen["fts_table"]:
                conn.execute(
                    "UPDATE generations SET fts_table = ? WHERE id = ?", (table, gen_id)
                )

        conn.execute("INSERT OR REPLACE INTO settings(key, value) VALUES('schema_version', '1')")
        conn.execute("PRAGMA user_version = 1")

        # If a previous process disappeared, its job becomes immediately
        # retriggerable on startup.  The live index was never touched.
        _mark_orphaned_owner_jobs(conn)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _mark_orphaned_owner_jobs(conn: sqlite3.Connection) -> None:
    host = socket.gethostname()
    rows = conn.execute(
        "SELECT id, generation_id, owner_host, owner_pid, owner_proc_start FROM rebuild_jobs "
        "WHERE state IN ('queued', 'building', 'catching_up', 'activating')"
    ).fetchall()
    now = utc_now()
    for row in rows:
        owner_is_gone = row["owner_host"] != host
        if row["owner_host"] == host and row["owner_pid"]:
            pid = int(row["owner_pid"])
            owner_is_gone = not pid_exists(pid)
            if not owner_is_gone and row["owner_proc_start"]:
                owner_is_gone = process_start_token(pid) != row["owner_proc_start"]
        if owner_is_gone:
            _mark_job_failed(
                conn,
                job_id=row["id"],
                generation_id=int(row["generation_id"]),
                state="interrupted",
                error="rebuild process stopped before completion",
                finished_at=now,
            )


def _mark_job_failed(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    generation_id: int,
    state: str,
    error: str,
    finished_at: str,
) -> None:
    conn.execute(
        """
        UPDATE rebuild_jobs
           SET state = ?, finished_at = ?, error = ?, heartbeat_at = ?,
               message = 'old index retained; rebuild can be retriggered'
         WHERE id = ? AND state IN ('queued', 'building', 'catching_up', 'activating')
        """,
        (state, finished_at, error, time.time(), job_id),
    )
    conn.execute(
        """
        UPDATE generations
           SET state = ?, finished_at = ?, error = ?
         WHERE id = ? AND state NOT IN ('active', 'retired')
        """,
        (state, finished_at, error, generation_id),
    )


def active_index(conn: sqlite3.Connection) -> sqlite3.Row:
    gen_row = conn.execute(
        "SELECT value FROM settings WHERE key = 'active_generation_id'"
    ).fetchone()
    policy_row = conn.execute(
        "SELECT value FROM settings WHERE key = 'active_policy'"
    ).fetchone()
    if gen_row is None or policy_row is None:
        raise RuntimeError("database has no active index")
    gen = conn.execute(
        "SELECT * FROM generations WHERE id = ?", (int(gen_row["value"]),)
    ).fetchone()
    if gen is None or gen["state"] != "active":
        raise RuntimeError("active index generation is inconsistent")
    return gen


def _active_job(conn: sqlite3.Connection) -> sqlite3.Row | None:
    placeholders = ", ".join("?" for _ in ACTIVE_JOB_STATES)
    return conn.execute(
        f"SELECT * FROM rebuild_jobs WHERE state IN ({placeholders}) "
        "ORDER BY created_epoch DESC LIMIT 1",
        ACTIVE_JOB_STATES,
    ).fetchone()


def _job_is_stale(job: sqlite3.Row, now_epoch: float) -> tuple[bool, str]:
    host = socket.gethostname()
    if job["owner_host"] == host and job["owner_pid"]:
        pid = int(job["owner_pid"])
        if not pid_exists(pid):
            return True, "rebuild owner process is no longer running"
        if job["owner_proc_start"] and process_start_token(pid) != job["owner_proc_start"]:
            return True, "rebuild owner PID was reused by a restarted process"
    heartbeat = job["heartbeat_at"] or job["created_epoch"]
    if now_epoch - float(heartbeat) > SETTINGS.rebuild_lease_seconds:
        return True, "rebuild heartbeat lease expired"
    return False, ""


def claim_rebuild(target_policy: str | None) -> dict[str, Any]:
    """Atomically claim the single rebuild slot or return the running job."""

    if target_policy is not None and target_policy not in VALID_POLICIES:
        raise ValueError(f"unsupported accent policy: {target_policy!r}")

    conn = connect()
    try:
        begin_immediate(conn)
        active = active_index(conn)
        current_policy = active["policy"]
        if target_policy is None:
            target_policy = "fold" if current_policy == "keep" else "keep"

        running = _active_job(conn)
        now_epoch = time.time()
        if running is not None:
            stale, reason = _job_is_stale(running, now_epoch)
            if stale:
                _mark_job_failed(
                    conn,
                    job_id=running["id"],
                    generation_id=int(running["generation_id"]),
                    state="failed",
                    error=reason,
                    finished_at=utc_now(),
                )
            else:
                conn.commit()
                return {
                    "job": job_to_dict(running),
                    "created": False,
                    "reused_existing_rebuild": True,
                    "requested_policy": target_policy,
                    "active_policy": current_policy,
                    "worker_token": None,
                }

        now = utc_now()
        cur = conn.execute(
            """
            INSERT INTO generations
                (policy, state, fts_table, base_seq, applied_seq, doc_count, created_at)
            VALUES (?, 'building', '', 0, 0, 0, ?)
            """,
            (target_policy, now),
        )
        generation_id = int(cur.lastrowid)
        table = fts_table_name(generation_id)
        _create_gen_fts(conn, table)
        conn.execute(
            "UPDATE generations SET fts_table = ? WHERE id = ?", (table, generation_id)
        )

        job_id = str(uuid.uuid4())
        claim_token = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO rebuild_jobs
                (id, generation_id, from_policy, to_policy, state, claim_token,
                 owner_host, owner_pid, owner_proc_start, created_at, created_epoch,
                 heartbeat_at, message)
            VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                generation_id,
                current_policy,
                target_policy,
                claim_token,
                socket.gethostname(),
                os.getpid(),
                process_start_token(os.getpid()),
                now,
                now_epoch,
                now_epoch,
                "queued for background rebuild",
            ),
        )
        conn.commit()
        job = conn.execute(
            "SELECT * FROM rebuild_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return {
            "job": job_to_dict(job),
            "created": True,
            "reused_existing_rebuild": False,
            "requested_policy": target_policy,
            "active_policy": current_policy,
            "worker_token": claim_token,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def job_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    total = int(row["total_documents"])
    processed = int(row["processed_documents"])
    percent = round((processed / total) * 100, 2) if total else 0.0
    return {
        "job_id": row["id"],
        "generation_id": int(row["generation_id"]),
        "from_policy": row["from_policy"],
        "to_policy": row["to_policy"],
        "accent_policy": row["to_policy"],
        "state": row["state"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "heartbeat_at": row["heartbeat_at"],
        "owner_host": row["owner_host"],
        "owner_pid": row["owner_pid"],
        "base_seq": int(row["base_seq"]),
        "applied_seq": int(row["applied_seq"]),
        "processed_documents": processed,
        "total_documents": total,
        "progress_percent": percent,
        "message": row["message"],
        "error": row["error"],
        "active": row["state"] in ACTIVE_JOB_STATES,
    }


def get_job(job_id: str) -> dict[str, Any] | None:
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM rebuild_jobs WHERE id = ?", (job_id,)).fetchone()
        return job_to_dict(row)
    finally:
        conn.close()


def get_status() -> dict[str, Any]:
    conn = connect()
    try:
        conn.execute("BEGIN")
        active = active_index(conn)
        running = _active_job(conn)
        latest = conn.execute(
            "SELECT * FROM rebuild_jobs ORDER BY created_epoch DESC LIMIT 1"
        ).fetchone()
        generations = conn.execute(
            "SELECT * FROM generations ORDER BY id DESC LIMIT 20"
        ).fetchall()
        conn.execute("COMMIT")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    generation_rows = [
        {
            "generation_id": int(g["id"]),
            "accent_policy": g["policy"],
            "state": g["state"],
            "base_seq": int(g["base_seq"]),
            "applied_seq": int(g["applied_seq"]),
            "document_count": int(g["doc_count"]),
            "created_at": g["created_at"],
            "activated_at": g["activated_at"],
            "finished_at": g["finished_at"],
            "error": g["error"],
        }
        for g in generations
    ]
    return {
        "active_policy": active["policy"],
        "active_generation_id": int(active["id"]),
        "active_index": {
            "generation_id": int(active["id"]),
            "accent_policy": active["policy"],
            "document_count": int(active["doc_count"]),
            "activated_at": active["activated_at"],
        },
        "rebuild_in_progress": running is not None,
        "current_rebuild_job": job_to_dict(running),
        "latest_rebuild_job": job_to_dict(latest),
        "generations": generation_rows,
        "server_time": utc_now(),
    }


def mark_job_progress(
    job_id: str,
    claim_token: str,
    *,
    state: str | None = None,
    started: bool = False,
    processed_documents: int | None = None,
    total_documents: int | None = None,
    base_seq: int | None = None,
    applied_seq: int | None = None,
    message: str | None = None,
) -> bool:
    """Update heartbeat/progress if this worker still owns an active job."""

    conn = connect()
    try:
        begin_immediate(conn)
        assignments = ["heartbeat_at = ?"]
        params: list[Any] = [time.time()]
        if state is not None:
            assignments.append("state = ?")
            params.append(state)
        if started:
            assignments.append("started_at = COALESCE(started_at, ?)")
            params.append(utc_now())
        if processed_documents is not None:
            assignments.append("processed_documents = ?")
            params.append(int(processed_documents))
        if total_documents is not None:
            assignments.append("total_documents = ?")
            params.append(int(total_documents))
        if base_seq is not None:
            assignments.append("base_seq = ?")
            params.append(int(base_seq))
        if applied_seq is not None:
            assignments.append("applied_seq = ?")
            params.append(int(applied_seq))
        if message is not None:
            assignments.append("message = ?")
            params.append(message)

        placeholders = ", ".join("?" for _ in ACTIVE_JOB_STATES)
        params.extend([job_id, claim_token, *ACTIVE_JOB_STATES])
        cur = conn.execute(
            f"""
            UPDATE rebuild_jobs SET {', '.join(assignments)}
             WHERE id = ? AND claim_token = ?
               AND state IN ({placeholders})
            """,
            params,
        )
        owned = cur.rowcount > 0
        if owned and (applied_seq is not None or base_seq is not None):
            job = conn.execute(
                "SELECT generation_id, state FROM rebuild_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if job is not None:
                if base_seq is not None and applied_seq is not None:
                    conn.execute(
                        "UPDATE generations SET base_seq = ?, applied_seq = ?, state = ? WHERE id = ?",
                        (int(base_seq), int(applied_seq), job["state"], int(job["generation_id"])),
                    )
                elif base_seq is not None:
                    conn.execute(
                        "UPDATE generations SET base_seq = ?, state = ? WHERE id = ?",
                        (int(base_seq), job["state"], int(job["generation_id"])),
                    )
                elif applied_seq is not None:
                    conn.execute(
                        "UPDATE generations SET applied_seq = ?, state = ? WHERE id = ?",
                        (int(applied_seq), job["state"], int(job["generation_id"])),
                    )
        conn.commit()
        return owned
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def job_is_owned(job_id: str, claim_token: str) -> bool:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT state, claim_token FROM rebuild_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return (
            row is not None
            and row["claim_token"] == claim_token
            and row["state"] in ACTIVE_JOB_STATES
        )
    finally:
        conn.close()


def fail_job(
    job_id: str,
    claim_token: str,
    *,
    state: str,
    error: str,
    generation_id: int,
) -> None:
    conn = connect()
    try:
        begin_immediate(conn)
        row = conn.execute(
            "SELECT generation_id, state, claim_token FROM rebuild_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if (
            row is not None
            and row["claim_token"] == claim_token
            and row["state"] in ACTIVE_JOB_STATES
        ):
            _mark_job_failed(
                conn,
                job_id=job_id,
                generation_id=generation_id,
                state=state,
                error=error,
                finished_at=utc_now(),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def interrupt_local_jobs(job_tokens: Iterable[tuple[str, str]]) -> None:
    """Best-effort graceful-shutdown marker; never touches the active index."""

    for job_id, token in job_tokens:
        conn = None
        try:
            conn = connect()
            begin_immediate(conn)
            row = conn.execute(
                "SELECT generation_id, state, claim_token, owner_pid FROM rebuild_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if (
                row is not None
                and row["claim_token"] == token
                and int(row["owner_pid"] or 0) == os.getpid()
                and row["state"] in ACTIVE_JOB_STATES
            ):
                _mark_job_failed(
                    conn,
                    job_id=job_id,
                    generation_id=int(row["generation_id"]),
                    state="interrupted",
                    error="service stopped before rebuild completed",
                    finished_at=utc_now(),
                )
                conn.commit()
            else:
                conn.rollback()
        except Exception:
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
        finally:
            if conn is not None:
                conn.close()
