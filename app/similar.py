"""Duplicate-scan detection for the OCR archive.

A duplicate scan shows up as two pages whose OCR bodies are almost the same
sequence of words.  The grouping rule is **fixed** and deliberately
independent of the search accent policy (`keep`/`fold` rebuilds never touch
these relations):

* the body is normalised with NFC and then casefolded;
* tokens are maximal runs of Unicode alphanumeric characters;
* the page signature is the de-duplicated set of adjacent token bigrams;
* two pages with non-empty bigram sets are *linked* when their Jaccard
  similarity ``|A ∩ B| / |A ∪ B|`` is at least ``0.8``.

Relations are stored in SQLite (``page_signatures`` and ``similar_pairs``)
and are maintained inside the same transaction that commits the page body,
the change-log entry and the live FTS update — if any step fails, nothing
changes.  The expensive part (reading the current signatures and computing
Jaccard scores) happens *before* the write transaction starts, so the
database write lock is never held while similarity is computed.  The write
transaction re-checks the change-log high-water mark before writing; if
another writer committed in the meantime the whole write is rolled back and
retried against a fresh snapshot, so simultaneously added similar pages can
never miss their link.

Connected groups are derived from the stored pairs at query time, so
correcting or deleting a bridge page re-splits the groups on the next read.
Pages that existed before this feature are picked up by a one-time backfill
(:func:`ensure_similarities_current`) that also computes outside the write
lock and swaps its result in only if the change log has not moved.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from typing import Any

from .db import begin_immediate, connect

SIMILARITY_THRESHOLD = 0.8

# Bounded optimistic-concurrency retries.  A retry only happens when another
# writer committed in the (short) gap between planning and the write lock, so
# every retry means global progress; 30 is far beyond any observed contention.
MAX_WRITE_ATTEMPTS = 30

# Maximal runs of Unicode alphanumerics (letters/digits); underscores and
# combining marks act as separators.  Same token rule as the search index.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

RULE_DESCRIPTION: dict[str, Any] = {
    "normalization": "NFC, then casefold",
    "tokenization": "maximal runs of Unicode alphanumeric characters",
    "signature": "de-duplicated set of adjacent token bigrams",
    "link_condition": (
        "both bigram sets non-empty and |intersection| / |union| >= "
        f"{SIMILARITY_THRESHOLD}"
    ),
    "accent_policy_independent": True,
    "groups": "connected components of the link graph; only groups of at least two pages are returned",
}


class SimilarityStateChanged(Exception):
    """Another writer committed between similarity planning and the write."""


def tokens_for(body: str) -> list[str]:
    """NFC + casefold, then maximal runs of Unicode alphanumerics."""

    normalised = unicodedata.normalize("NFC", body).casefold()
    return _TOKEN_RE.findall(normalised)


def bigrams_for(body: str) -> frozenset[tuple[str, str]]:
    """De-duplicated set of adjacent two-token pairs of the normalised body."""

    tokens = tokens_for(body)
    return frozenset(zip(tokens, tokens[1:]))


def jaccard(
    first: frozenset[tuple[str, str]], second: frozenset[tuple[str, str]]
) -> float:
    if not first or not second:
        return 0.0
    intersection = len(first & second)
    union = len(first) + len(second) - intersection
    return intersection / union


def _encode_bigrams(bigrams: frozenset[tuple[str, str]]) -> str:
    # Tokens never contain spaces, so "w1 w2" is an unambiguous encoding.
    return json.dumps(
        sorted(f"{left} {right}" for left, right in bigrams), ensure_ascii=False
    )


def _decode_bigrams(payload: str) -> frozenset[tuple[str, str]]:
    return frozenset(tuple(item.split(" ", 1)) for item in json.loads(payload))


def _max_change_seq(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COALESCE(MAX(seq), 0) FROM changes").fetchone()[0])


def _mark_signatures_seq(conn: sqlite3.Connection, seq: int) -> None:
    """Record the change-log sequence up to which the relations are current."""

    conn.execute(
        "INSERT OR REPLACE INTO settings(key, value) VALUES('signatures_seq', ?)",
        (str(seq),),
    )


def read_signatures_snapshot() -> tuple[int, dict[str, tuple[int, frozenset[tuple[str, str]]]]]:
    """Consistent read-only snapshot of every stored signature.

    Returns the change-log high-water mark plus ``page_id -> (rev, bigrams)``.
    Runs in a plain read transaction: no write lock is taken, so concurrent
    page writes are never blocked while similarity is planned.
    """

    conn = connect()
    try:
        conn.execute("BEGIN")
        seq = _max_change_seq(conn)
        rows = conn.execute(
            "SELECT page_id, rev, bigrams FROM page_signatures"
        ).fetchall()
        conn.execute("COMMIT")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return seq, {
        str(row["page_id"]): (int(row["rev"]), _decode_bigrams(str(row["bigrams"])))
        for row in rows
    }


def plan_links(
    page_id: str,
    new_bigrams: frozenset[tuple[str, str]],
    signatures: dict[str, tuple[int, frozenset[tuple[str, str]]]],
) -> list[dict[str, Any]]:
    """Compute one page's links against a snapshot, without touching the DB."""

    links: list[dict[str, Any]] = []
    if new_bigrams:
        for other_id, (other_rev, other_bigrams) in signatures.items():
            if other_id == page_id or not other_bigrams:
                continue
            score = jaccard(new_bigrams, other_bigrams)
            if score >= SIMILARITY_THRESHOLD:
                links.append({"page_id": other_id, "rev": other_rev, "score": score})
    links.sort(key=lambda link: link["page_id"])
    return links


def assert_snapshot_current(conn: sqlite3.Connection, planned_seq: int) -> None:
    """Verify, inside the write transaction, that the planning snapshot is
    still the committed state.

    Called under the write lock before any write happens, so a mismatch can
    simply roll the transaction back and let the caller retry with a fresh
    snapshot.  This is what makes concurrent similar writes unable to miss
    their link without ever computing under the lock.
    """

    if _max_change_seq(conn) != planned_seq:
        raise SimilarityStateChanged(
            f"change log advanced past snapshot sequence {planned_seq}"
        )


def apply_upsert_relations(
    conn: sqlite3.Connection,
    *,
    page_id: str,
    rev: int,
    bigrams: frozenset[tuple[str, str]],
    links: list[dict[str, Any]],
) -> None:
    """Replace one page's signature and pair rows inside the write transaction."""

    conn.execute(
        """
        INSERT INTO page_signatures(page_id, rev, bigram_count, bigrams)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(page_id) DO UPDATE SET
            rev = excluded.rev,
            bigram_count = excluded.bigram_count,
            bigrams = excluded.bigrams
        """,
        (page_id, rev, len(bigrams), _encode_bigrams(bigrams)),
    )
    conn.execute(
        "DELETE FROM similar_pairs WHERE page_a = ? OR page_b = ?", (page_id, page_id)
    )
    for link in links:
        other_id = str(link["page_id"])
        page_a, page_b = sorted((page_id, other_id))
        rev_a = rev if page_a == page_id else int(link["rev"])
        rev_b = rev if page_b == page_id else int(link["rev"])
        conn.execute(
            """
            INSERT INTO similar_pairs(page_a, page_b, score, rev_a, rev_b)
            VALUES (?, ?, ?, ?, ?)
            """,
            (page_a, page_b, float(link["score"]), rev_a, rev_b),
        )
    _mark_signatures_seq(conn, rev)


def apply_delete_relations(conn: sqlite3.Connection, *, page_id: str, seq: int) -> None:
    """Remove every similarity trace of a page inside the delete transaction."""

    conn.execute("DELETE FROM page_signatures WHERE page_id = ?", (page_id,))
    conn.execute(
        "DELETE FROM similar_pairs WHERE page_a = ? OR page_b = ?", (page_id, page_id)
    )
    _mark_signatures_seq(conn, seq)


def ensure_similarities_current() -> bool:
    """Backfill signatures/pairs for pages that predate this feature.

    No-op when the stored relations already match the committed change log
    (every page write/delete keeps them current in the same transaction).
    The all-pairs computation runs outside any write transaction; the result
    is swapped in only if the change log has not moved since the snapshot,
    otherwise it is recomputed.  Safe to call concurrently from both API
    worker processes.  Returns True when a backfill was committed.
    """

    for _attempt in range(MAX_WRITE_ATTEMPTS):
        conn = connect()
        try:
            conn.execute("BEGIN")
            seq = _max_change_seq(conn)
            marker = conn.execute(
                "SELECT value FROM settings WHERE key = 'signatures_seq'"
            ).fetchone()
            if marker is not None and int(marker["value"]) == seq:
                conn.execute("COMMIT")
                return False
            rows = conn.execute(
                "SELECT page_id, rev, body FROM pages ORDER BY page_id"
            ).fetchall()
            conn.execute("COMMIT")
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        # CPU-only section: no connection held, no lock taken.
        signatures = {
            str(row["page_id"]): (int(row["rev"]), bigrams_for(str(row["body"])))
            for row in rows
        }
        pairs: list[tuple[str, str, float, int, int]] = []
        page_ids = sorted(signatures)
        for left_index, left_id in enumerate(page_ids):
            left_rev, left_bigrams = signatures[left_id]
            if not left_bigrams:
                continue
            for right_id in page_ids[left_index + 1 :]:
                right_rev, right_bigrams = signatures[right_id]
                score = jaccard(left_bigrams, right_bigrams)
                if score >= SIMILARITY_THRESHOLD:
                    pairs.append((left_id, right_id, score, left_rev, right_rev))

        conn = connect()
        try:
            begin_immediate(conn)
            if _max_change_seq(conn) != seq:
                conn.rollback()
                continue  # a write landed meanwhile; recompute from scratch
            conn.execute("DELETE FROM page_signatures")
            conn.executemany(
                "INSERT INTO page_signatures(page_id, rev, bigram_count, bigrams) "
                "VALUES (?, ?, ?, ?)",
                [
                    (page_id, rev, len(bigrams), _encode_bigrams(bigrams))
                    for page_id, (rev, bigrams) in signatures.items()
                ],
            )
            conn.execute("DELETE FROM similar_pairs")
            conn.executemany(
                "INSERT INTO similar_pairs(page_a, page_b, score, rev_a, rev_b) "
                "VALUES (?, ?, ?, ?, ?)",
                pairs,
            )
            _mark_signatures_seq(conn, seq)
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    raise sqlite3.OperationalError(
        "similarity backfill could not catch up with concurrent writes"
    )


def duplicate_groups() -> dict[str, Any]:
    """Connected duplicate groups, similar pairs and current page revisions.

    Everything is read in one SQLite snapshot, so the returned revisions and
    relations always come from the same committed state.  Groups are computed
    from the stored pairs on every call, which is what makes group splits
    after a bridge-page correction or deletion immediately visible.
    """

    ensure_similarities_current()
    conn = connect()
    try:
        conn.execute("BEGIN")
        seq = _max_change_seq(conn)
        pair_rows = conn.execute(
            "SELECT page_a, page_b, score, rev_a, rev_b FROM similar_pairs "
            "ORDER BY page_a, page_b"
        ).fetchall()
        page_rows = conn.execute(
            "SELECT page_id, rev, updated_at FROM pages"
        ).fetchall()
        conn.execute("COMMIT")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    pages = {
        str(row["page_id"]): {
            "page_id": str(row["page_id"]),
            "revision": int(row["rev"]),
            "updated_at": row["updated_at"],
        }
        for row in page_rows
    }

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    valid_rows: list[sqlite3.Row] = []
    for row in pair_rows:
        page_a, page_b = str(row["page_a"]), str(row["page_b"])
        if page_a not in pages or page_b not in pages:
            # Cannot happen while relations are maintained transactionally;
            # stay defensive instead of crashing the QA view.
            continue
        parent.setdefault(page_a, page_a)
        parent.setdefault(page_b, page_b)
        union(page_a, page_b)
        valid_rows.append(row)

    members: dict[str, list[str]] = defaultdict(list)
    for page_id in parent:
        members[find(page_id)].append(page_id)

    group_members = [sorted(ids) for ids in members.values() if len(ids) >= 2]
    group_members.sort(key=lambda ids: (-len(ids), ids[0]))

    group_of: dict[str, int] = {}
    groups: list[dict[str, Any]] = []
    for index, ids in enumerate(group_members, start=1):
        for page_id in ids:
            group_of[page_id] = index
        groups.append(
            {
                "group_id": index,
                "size": len(ids),
                "pages": [pages[page_id] for page_id in ids],
            }
        )

    pairs = [
        {
            "page_a": str(row["page_a"]),
            "page_b": str(row["page_b"]),
            "score": round(float(row["score"]), 6),
            "revision_a": pages[str(row["page_a"])]["revision"],
            "revision_b": pages[str(row["page_b"])]["revision"],
            "group_id": group_of[str(row["page_a"])],
        }
        for row in valid_rows
    ]
    pairs.sort(key=lambda pair: (-pair["score"], pair["page_a"], pair["page_b"]))

    return {
        "rule": RULE_DESCRIPTION,
        "threshold": SIMILARITY_THRESHOLD,
        "as_of_change_seq": seq,
        "page_count": len(pages),
        "linked_page_count": len(parent),
        "pair_count": len(pairs),
        "group_count": len(groups),
        "groups": groups,
        "pairs": pairs,
    }
