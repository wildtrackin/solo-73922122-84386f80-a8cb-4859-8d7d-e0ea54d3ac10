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


def _fts_ident(generation: sqlite3.Row) -> str:
    return quote_identifier(str(generation["fts_table"]))


def upsert_page(page_id: str, title: str, body: str) -> dict[str, Any]:
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
