"""Transactional page writes and snapshot-consistent reading."""

from __future__ import annotations

import sqlite3
from typing import Any

from .accents import build_match_expression, normalize_for_policy
from .db import (
    active_index,
    begin_immediate,
    connect,
    quote_identifier,
    utc_now,
)
from .similar import (
    MAX_WRITE_ATTEMPTS,
    SimilarityStateChanged,
    apply_delete_relations,
    apply_upsert_relations,
    assert_snapshot_current,
    bigrams_for,
    ensure_similarities_current,
    plan_links,
    read_signatures_snapshot,
)


def _fts_ident(generation: sqlite3.Row) -> str:
    return quote_identifier(str(generation["fts_table"]))


def upsert_page(page_id: str, title: str, body: str) -> dict[str, Any]:
    """Insert/replace a page and refresh its duplicate-scan relations.

    The similarity plan is computed from a read-only snapshot *before* the
    write transaction starts, so the database write lock is never held while
    scores are computed.  The write transaction re-checks the change-log
    high-water mark first; if a concurrent writer committed in between, the
    transaction is rolled back and retried against a fresh snapshot, so
    simultaneously added similar pages cannot miss their link.  Page body,
    change log, live FTS index and similarity relations commit in the same
    transaction — or, on any failure, not at all.
    """

    new_bigrams = bigrams_for(body)
    ensure_similarities_current()
    for _attempt in range(MAX_WRITE_ATTEMPTS):
        snapshot_seq, signatures = read_signatures_snapshot()
        links = plan_links(page_id, new_bigrams, signatures)
        try:
            return _commit_upsert(page_id, title, body, new_bigrams, links, snapshot_seq)
        except SimilarityStateChanged:
            continue  # a concurrent write moved the log; replan and retry
    raise sqlite3.OperationalError(
        "page upsert could not be reconciled with concurrent writes"
    )


def _commit_upsert(
    page_id: str,
    title: str,
    body: str,
    new_bigrams: frozenset,
    links: list[dict[str, Any]],
    planned_seq: int,
) -> dict[str, Any]:
    conn = connect()
    try:
        begin_immediate(conn)
        # Under the write lock, before any write: the plan is only valid for
        # exactly this committed state.
        assert_snapshot_current(conn, planned_seq)
        generation = active_index(conn)
        policy = str(generation["policy"])
        now = utc_now()

        existing = conn.execute(
            "SELECT rowid AS page_ref FROM pages WHERE page_id = ?", (page_id,)
        ).fetchone()
        page_ref = int(existing["page_ref"]) if existing is not None else None

        cur = conn.execute(
            """
            INSERT INTO changes(page_id, page_ref, op, title, body, rev, changed_at)
            VALUES (?, ?, 'upsert', ?, ?, 0, ?)
            """,
            (page_id, page_ref, title, body, now),
        )
        seq = int(cur.lastrowid)

        conn.execute(
            """
            INSERT INTO pages(page_id, title, body, rev, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(page_id) DO UPDATE SET
                title = excluded.title,
                body = excluded.body,
                rev = excluded.rev,
                updated_at = excluded.updated_at
            """,
            (page_id, title, body, seq, now),
        )
        if page_ref is None:
            page_ref = int(
                conn.execute(
                    "SELECT rowid AS page_ref FROM pages WHERE page_id = ?", (page_id,)
                ).fetchone()["page_ref"]
            )
            conn.execute(
                "UPDATE generations SET doc_count = doc_count + 1, applied_seq = ? WHERE id = ?",
                (seq, int(generation["id"])),
            )
        else:
            conn.execute(
                "UPDATE generations SET applied_seq = ? WHERE id = ?",
                (seq, int(generation["id"])),
            )
        conn.execute(
            "UPDATE changes SET rev = ?, page_ref = ? WHERE seq = ?",
            (seq, page_ref, seq),
        )

        ident = _fts_ident(generation)
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {ident}(rowid, page_id, title, body)
            VALUES (?, ?, ?, ?)
            """,
            (
                page_ref,
                page_id,
                normalize_for_policy(title, policy),
                normalize_for_policy(body, policy),
            ),
        )
        apply_upsert_relations(
            conn, page_id=page_id, rev=seq, bigrams=new_bigrams, links=links
        )
        conn.commit()
        return {
            "page_id": page_id,
            "revision": seq,
            "accent_policy": policy,
            "generation_id": int(generation["id"]),
            "indexed": True,
            "upserted": True,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_page(page_id: str) -> dict[str, Any]:
    # Keep the "relations are current at the change-log high-water mark"
    # invariant: this write advances the log, so any pending backfill for
    # pre-existing pages must complete first (a cheap no-op when current).
    ensure_similarities_current()
    conn = connect()
    try:
        begin_immediate(conn)
        generation = active_index(conn)
        policy = str(generation["policy"])
        now = utc_now()

        existing = conn.execute(
            "SELECT rowid AS page_ref FROM pages WHERE page_id = ?", (page_id,)
        ).fetchone()
        page_ref = int(existing["page_ref"]) if existing is not None else None

        cur = conn.execute(
            """
            INSERT INTO changes(page_id, page_ref, op, title, body, rev, changed_at)
            VALUES (?, ?, 'delete', NULL, NULL, 0, ?)
            """,
            (page_id, page_ref, now),
        )
        seq = int(cur.lastrowid)
        conn.execute(
            "UPDATE changes SET rev = ?, page_ref = ? WHERE seq = ?",
            (seq, page_ref, seq),
        )

        conn.execute("DELETE FROM pages WHERE page_id = ?", (page_id,))
        if page_ref is not None:
            conn.execute(
                "UPDATE generations SET doc_count = doc_count - 1, applied_seq = ? WHERE id = ?",
                (seq, int(generation["id"])),
            )
        else:
            conn.execute(
                "UPDATE generations SET applied_seq = ? WHERE id = ?",
                (seq, int(generation["id"])),
            )

        ident = _fts_ident(generation)
        if page_ref is not None:
            conn.execute(f"DELETE FROM {ident} WHERE rowid = ?", (page_ref,))
        else:
            # Idempotent delete of an absent page.  This fallback also cleans
            # up any pre-existing orphan from an older schema version.
            conn.execute(f"DELETE FROM {ident} WHERE page_id = ?", (page_id,))

        apply_delete_relations(conn, page_id=page_id, seq=seq)
        conn.commit()
        return {
            "page_id": page_id,
            "revision": seq,
            "accent_policy": policy,
            "generation_id": int(generation["id"]),
            "deleted": True,
            "existed": existing is not None,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def search(raw_query: str, *, limit: int, offset: int) -> dict[str, Any]:
    conn = connect()
    try:
        # A single read transaction pins both the active-generation pointer and
        # the page rows.  Hits and returned bodies therefore come from the same
        # committed database state, even if a rebuild swap commits concurrently.
        conn.execute("BEGIN")
        generation = active_index(conn)
        policy = str(generation["policy"])
        expression, terms = build_match_expression(raw_query, policy)
        ident = _fts_ident(generation)

        total = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                  FROM {ident} AS f
                  JOIN pages AS p ON p.page_id = f.page_id
                 WHERE {ident} MATCH ?
                """,
                (expression,),
            ).fetchone()[0]
        )
        rows = conn.execute(
            f"""
            SELECT p.page_id, p.title, p.body, p.rev, p.updated_at,
                   bm25({ident}) AS score
              FROM {ident} AS f
              JOIN pages AS p ON p.page_id = f.page_id
             WHERE {ident} MATCH ?
             ORDER BY score ASC, p.page_id ASC
             LIMIT ? OFFSET ?
            """,
            (expression, limit, offset),
        ).fetchall()
        conn.execute("COMMIT")
        return {
            "query": raw_query,
            "query_terms": terms,
            "accent_policy": policy,
            "generation_id": int(generation["id"]),
            "total": total,
            "limit": limit,
            "offset": offset,
            "hits": [
                {
                    "page_id": row["page_id"],
                    "title": row["title"],
                    "body": row["body"],
                    "revision": int(row["rev"]),
                    "updated_at": row["updated_at"],
                    "score": float(row["score"]),
                }
                for row in rows
            ],
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_page(page_id: str) -> dict[str, Any] | None:
    conn = connect()
    try:
        conn.execute("BEGIN")
        generation = active_index(conn)
        row = conn.execute(
            "SELECT page_id, title, body, rev, updated_at FROM pages WHERE page_id = ?",
            (page_id,),
        ).fetchone()
        conn.execute("COMMIT")
        if row is None:
            return None
        return {
            "page_id": row["page_id"],
            "title": row["title"],
            "body": row["body"],
            "revision": int(row["rev"]),
            "updated_at": row["updated_at"],
            "accent_policy": str(generation["policy"]),
            "generation_id": int(generation["id"]),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
