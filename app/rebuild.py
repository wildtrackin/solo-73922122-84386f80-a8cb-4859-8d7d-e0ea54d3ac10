"""Background, cross-process-safe generation rebuild.

Only the process that successfully claims the job in :mod:`app.db` starts this
worker.  The worker writes solely to its private shadow FTS table.  It reads a
pinned snapshot of the canonical pages, then catches up every later change-log
entry.  The active-generation pointer is changed in one short write
transaction only after the catch-up is complete.
"""

from __future__ import annotations

import logging
import threading
import time

import sqlite3

from .accents import normalize_for_policy
from .config import SETTINGS
from .db import (
    ACTIVE_JOB_STATES,
    begin_immediate,
    connect,
    fail_job,
    interrupt_local_jobs,
    job_is_owned,
    mark_job_progress,
    quote_identifier,
    utc_now,
)

logger = logging.getLogger("archive.rebuild")

_local_jobs: dict[str, tuple[threading.Event, str, str]] = {}
_registry_lock = threading.Lock()


class RebuildCancelled(Exception):
    pass


class RebuildFenced(Exception):
    pass


def start_rebuild_thread(
    *, job_id: str, generation_id: int, policy: str, claim_token: str
) -> threading.Thread:
    stop_event = threading.Event()
    with _registry_lock:
        _local_jobs[job_id] = (stop_event, claim_token, policy)

    thread = threading.Thread(
        target=_run_worker,
        kwargs={
            "job_id": job_id,
            "generation_id": generation_id,
            "policy": policy,
            "claim_token": claim_token,
            "stop_event": stop_event,
        },
        name=f"rebuild-{job_id[:8]}",
        daemon=True,
    )
    thread.start()
    return thread


def cancel_local_rebuilds() -> None:
    """Ask this process's workers to stop, then mark their jobs interrupted."""

    with _registry_lock:
        entries = list(_local_jobs.items())
        for _, (event, _, _) in entries:
            event.set()
    if entries:
        interrupt_local_jobs([(job_id, token) for job_id, (_, token, _) in entries])


def _run_worker(
    *,
    job_id: str,
    generation_id: int,
    policy: str,
    claim_token: str,
    stop_event: threading.Event,
) -> None:
    table = f"fts_gen_{generation_id}"
    started = False
    try:
        started = mark_job_progress(
            job_id,
            claim_token,
            state="building",
            started=True,
            message="starting background rebuild",
        )
        if not started:
            raise RebuildCancelled("job is no longer owned by this process")

        cutoff_seq, total_documents = _build_baseline(
            job_id=job_id,
            claim_token=claim_token,
            table=table,
            policy=policy,
            stop_event=stop_event,
        )

        mark_job_progress(
            job_id,
            claim_token,
            state="catching_up",
            base_seq=cutoff_seq,
            applied_seq=cutoff_seq,
            processed_documents=total_documents,
            total_documents=total_documents,
            message=f"baseline complete; catching up changes after sequence {cutoff_seq}",
        )

        applied_seq = _catch_up(
            job_id=job_id,
            claim_token=claim_token,
            table=table,
            policy=policy,
            applied_seq=cutoff_seq,
            stop_event=stop_event,
        )

        _activate(
            job_id=job_id,
            claim_token=claim_token,
            generation_id=generation_id,
            policy=policy,
            table=table,
            applied_seq=applied_seq,
            stop_event=stop_event,
        )
        logger.info("rebuild %s activated generation %s (%s)", job_id, generation_id, policy)
    except RebuildCancelled as exc:
        logger.info("rebuild %s cancelled/interrupted: %s", job_id, exc)
        try:
            fail_job(
                job_id,
                claim_token,
                state="interrupted",
                error=str(exc) or "rebuild interrupted",
                generation_id=generation_id,
            )
        except Exception:
            logger.exception("could not mark interrupted rebuild %s", job_id)
    except RebuildFenced as exc:
        logger.info("rebuild %s fenced: %s", job_id, exc)
        try:
            fail_job(
                job_id,
                claim_token,
                state="failed",
                error=str(exc),
                generation_id=generation_id,
            )
        except Exception:
            logger.exception("could not mark fenced rebuild %s", job_id)
    except Exception as exc:
        logger.exception("rebuild %s failed", job_id)
        try:
            fail_job(
                job_id,
                claim_token,
                state="failed",
                error=f"{type(exc).__name__}: {exc}",
                generation_id=generation_id,
            )
        except Exception:
            logger.exception("could not record failure for rebuild %s", job_id)
    finally:
        with _registry_lock:
            _local_jobs.pop(job_id, None)


def _check_stop(job_id: str, claim_token: str, stop_event: threading.Event) -> None:
    if stop_event.is_set():
        raise RebuildCancelled("process shutdown requested")
    if not job_is_owned(job_id, claim_token):
        raise RebuildCancelled("rebuild job was superseded or interrupted")
    if SETTINGS.rebuild_batch_delay_ms:
        if stop_event.wait(SETTINGS.rebuild_batch_delay_ms / 1000.0):
            raise RebuildCancelled("process shutdown requested")


def _identity(title: str, body: str, policy: str) -> tuple[str, str]:
    return normalize_for_policy(title, policy), normalize_for_policy(body, policy)


def _insert_document(
    conn: sqlite3.Connection,
    table: str,
    page_ref: int,
    page_id: str,
    title: str,
    body: str,
    policy: str,
) -> None:
    indexed_title, indexed_body = _identity(title, body, policy)
    conn.execute(
        f"INSERT INTO {quote_identifier(table)}(rowid, page_id, title, body) "
        "VALUES (?, ?, ?, ?)",
        (page_ref, page_id, indexed_title, indexed_body),
    )


def _replace_document(
    conn: sqlite3.Connection,
    table: str,
    page_ref: int | None,
    page_id: str,
    title: str,
    body: str,
    policy: str,
) -> None:
    indexed_title, indexed_body = _identity(title, body, policy)
    ident = quote_identifier(table)
    if page_ref is None:
        conn.execute(f"DELETE FROM {ident} WHERE page_id = ?", (page_id,))
        conn.execute(
            f"INSERT INTO {ident}(page_id, title, body) VALUES (?, ?, ?)",
            (page_id, indexed_title, indexed_body),
        )
    else:
        conn.execute(
            f"INSERT OR REPLACE INTO {ident}(rowid, page_id, title, body) "
            "VALUES (?, ?, ?, ?)",
            (page_ref, page_id, indexed_title, indexed_body),
        )


def _delete_document(
    conn: sqlite3.Connection, table: str, page_ref: int | None, page_id: str
) -> None:
    ident = quote_identifier(table)
    if page_ref is None:
        conn.execute(f"DELETE FROM {ident} WHERE page_id = ?", (page_id,))
    else:
        conn.execute(f"DELETE FROM {ident} WHERE rowid = ?", (page_ref,))


def _build_baseline(
    *,
    job_id: str,
    claim_token: str,
    table: str,
    policy: str,
    stop_event: threading.Event,
) -> tuple[int, int]:
    """Copy the latest committed pages as of a pinned WAL read snapshot."""

    source = connect(query_only=True)
    target = connect()
    processed = 0
    try:
        source.execute("BEGIN")
        # This first read pins the read snapshot for the whole baseline scan.
        source.execute("SELECT value FROM settings WHERE key = 'active_generation_id'").fetchone()
        cutoff_seq = int(
            source.execute("SELECT COALESCE(MAX(seq), 0) FROM changes").fetchone()[0]
        )
        total_documents = int(source.execute("SELECT COUNT(*) FROM pages").fetchone()[0])
        mark_job_progress(
            job_id,
            claim_token,
            base_seq=cutoff_seq,
            total_documents=total_documents,
            applied_seq=cutoff_seq,
            message=f"scanning {total_documents} committed pages at snapshot sequence {cutoff_seq}",
        )

        cursor = source.execute(
            "SELECT rowid AS page_ref, page_id, title, body FROM pages ORDER BY page_id"
        )
        while True:
            _check_stop(job_id, claim_token, stop_event)
            batch = cursor.fetchmany(SETTINGS.rebuild_batch_size)
            if not batch:
                break
            begin_immediate(target)
            try:
                for row in batch:
                    _insert_document(
                        target,
                        table,
                        int(row["page_ref"]),
                        row["page_id"],
                        row["title"],
                        row["body"],
                        policy,
                    )
                target.commit()
            except Exception:
                target.rollback()
                raise
            processed += len(batch)
            mark_job_progress(
                job_id,
                claim_token,
                processed_documents=processed,
                total_documents=total_documents,
                message=f"scanning baseline: {processed}/{total_documents} pages",
            )

        source.execute("COMMIT")
        return cutoff_seq, total_documents
    finally:
        try:
            if source.in_transaction:
                source.execute("ROLLBACK")
        except Exception:
            pass
        source.close()
        target.close()


def _apply_change_rows(
    conn: sqlite3.Connection,
    table: str,
    policy: str,
    rows: list[sqlite3.Row],
) -> int:
    latest_seq = 0
    for row in rows:
        page_id = row["page_id"]
        page_ref = int(row["page_ref"]) if row["page_ref"] is not None else None
        if row["op"] == "upsert":
            _replace_document(
                conn,
                table,
                page_ref,
                page_id,
                row["title"] or "",
                row["body"] or "",
                policy,
            )
        else:
            _delete_document(conn, table, page_ref, page_id)
        latest_seq = int(row["seq"])
    return latest_seq


def _catch_up(
    *,
    job_id: str,
    claim_token: str,
    table: str,
    policy: str,
    applied_seq: int,
    stop_event: threading.Event,
) -> int:
    """Apply post-snapshot change-log rows without holding a long write lock."""

    conn = connect()
    try:
        while True:
            _check_stop(job_id, claim_token, stop_event)
            begin_immediate(conn)
            try:
                rows = conn.execute(
                    """
                    SELECT seq, page_id, page_ref, op, title, body
                      FROM changes
                     WHERE seq > ?
                     ORDER BY seq
                     LIMIT ?
                    """,
                    (applied_seq, SETTINGS.rebuild_batch_size),
                ).fetchall()
                if not rows:
                    conn.commit()
                    return applied_seq
                applied_seq = _apply_change_rows(conn, table, policy, rows)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

            current_max = 0
            max_conn = connect()
            try:
                current_max = int(
                    max_conn.execute("SELECT COALESCE(MAX(seq), 0) FROM changes").fetchone()[0]
                )
            finally:
                max_conn.close()
            mark_job_progress(
                job_id,
                claim_token,
                applied_seq=applied_seq,
                message=f"catching up changes through sequence {applied_seq}/{current_max}",
            )
    finally:
        conn.close()


def _activate(
    *,
    job_id: str,
    claim_token: str,
    generation_id: int,
    policy: str,
    table: str,
    applied_seq: int,
    stop_event: threading.Event,
) -> None:
    """Final catch-up, verification and atomic pointer swap."""

    if stop_event.is_set():
        raise RebuildCancelled("process shutdown requested")

    conn = connect()
    try:
        begin_immediate(conn)
        try:
            job = conn.execute(
                "SELECT * FROM rebuild_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if (
                job is None
                or job["claim_token"] != claim_token
                or job["state"] not in ACTIVE_JOB_STATES
            ):
                raise RebuildFenced("rebuild job is no longer active")
            generation = conn.execute(
                "SELECT * FROM generations WHERE id = ?", (generation_id,)
            ).fetchone()
            if (
                generation is None
                or generation["policy"] != policy
                or generation["state"] not in ("building", "catching_up", "activating")
            ):
                raise RebuildFenced("target generation is missing or has the wrong policy")

            conn.execute(
                "UPDATE rebuild_jobs SET state = 'activating', heartbeat_at = ?, "
                "message = 'final catch-up and atomic index switch' WHERE id = ?",
                (time.time(), job_id),
            )

            # The write lock is held only for this final catch-up and the
            # pointer swap.  If the incremental catch-up already drained the
            # log this loop usually has nothing to do.
            while True:
                rows = conn.execute(
                    """
                    SELECT seq, page_id, page_ref, op, title, body
                      FROM changes
                     WHERE seq > ?
                     ORDER BY seq
                     LIMIT ?
                    """,
                    (applied_seq, SETTINGS.rebuild_batch_size),
                ).fetchall()
                if not rows:
                    break
                applied_seq = _apply_change_rows(conn, table, policy, rows)

            max_seq = int(conn.execute("SELECT COALESCE(MAX(seq), 0) FROM changes").fetchone()[0])
            if applied_seq < max_seq:
                raise RuntimeError(
                    f"change log did not drain: applied={applied_seq}, max={max_seq}"
                )

            ident = quote_identifier(table)
            fts_count = int(conn.execute(f"SELECT COUNT(*) FROM {ident}").fetchone()[0])
            page_count = int(conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0])
            if fts_count != page_count:
                raise RuntimeError(
                    f"shadow index document count mismatch: fts={fts_count}, pages={page_count}"
                )

            now = utc_now()
            conn.execute(
                """
                UPDATE generations
                   SET state = 'retired', finished_at = ?
                 WHERE state = 'active' AND id <> ?
                """,
                (now, generation_id),
            )
            conn.execute(
                """
                UPDATE generations
                   SET state = 'active', applied_seq = ?, doc_count = ?,
                       activated_at = ?, finished_at = ?, error = NULL
                 WHERE id = ?
                """,
                (applied_seq, page_count, now, now, generation_id),
            )
            conn.execute(
                "INSERT OR REPLACE INTO settings(key, value) VALUES('active_generation_id', ?)",
                (str(generation_id),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO settings(key, value) VALUES('active_policy', ?)",
                (policy,),
            )
            conn.execute(
                """
                UPDATE rebuild_jobs
                   SET state = 'succeeded', heartbeat_at = ?, finished_at = ?,
                       applied_seq = ?, total_documents = ?, processed_documents = ?,
                       error = NULL,
                       message = 'new index active; switch completed atomically'
                 WHERE id = ?
                """,
                (time.time(), now, applied_seq, page_count, page_count, job_id),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()
