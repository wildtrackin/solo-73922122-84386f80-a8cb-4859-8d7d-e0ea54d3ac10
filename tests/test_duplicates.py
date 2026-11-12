"""Tests for the fixed-rule duplicate-scan grouping API."""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import SETTINGS


def configure_database(path: Path) -> None:
    object.__setattr__(SETTINGS, "db_path", path)
    object.__setattr__(SETTINGS, "rebuild_batch_delay_ms", 120)
    object.__setattr__(SETTINGS, "rebuild_batch_size", 2)
    object.__setattr__(SETTINGS, "rebuild_lease_seconds", 3)


@pytest.fixture
def client(tmp_path: Path):
    configure_database(tmp_path / "archive.db")
    from app.main import app
    from app.rebuild import cancel_local_rebuilds

    with TestClient(app) as test_client:
        yield test_client
    cancel_local_rebuilds()


def _put(client: TestClient, page_id: str, body: str, title: str = "t") -> dict:
    response = client.put(f"/api/pages/{page_id}", json={"title": title, "body": body})
    assert response.status_code == 200, response.text
    return response.json()


def _duplicates(client: TestClient) -> dict:
    response = client.get("/api/duplicates")
    assert response.status_code == 200, response.text
    return response.json()


def _wait_for_terminal(client: TestClient, job_id: str, timeout: float = 8.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get("/api/index/status").json()
        job = status["current_rebuild_job"] or status["latest_rebuild_job"]
        if job and job["job_id"] == job_id and job["state"] in {
            "succeeded",
            "failed",
            "interrupted",
        }:
            return status
        time.sleep(0.03)
    raise AssertionError(f"rebuild {job_id} did not finish within {timeout}s")


def test_group_pairs_and_current_revisions(client: TestClient):
    first = _put(client, "page-a", "alpha bravo charlie delta echo")
    second = _put(client, "page-b", "alpha bravo charlie delta echo foxtrot")
    _put(client, "page-c", "completely unrelated vocabulary shows up here")

    data = _duplicates(client)
    assert data["threshold"] == 0.8
    assert data["rule"]["accent_policy_independent"] is True
    assert data["page_count"] == 3
    assert data["group_count"] == 1
    assert data["pair_count"] == 1

    group = data["groups"][0]
    assert group["size"] == 2
    assert [page["page_id"] for page in group["pages"]] == ["page-a", "page-b"]
    revisions = {page["page_id"]: page["revision"] for page in group["pages"]}
    assert revisions == {"page-a": first["revision"], "page-b": second["revision"]}

    pair = data["pairs"][0]
    assert pair["page_a"] == "page-a"
    assert pair["page_b"] == "page-b"
    # 4 shared bigrams out of 5: exactly at the threshold.
    assert pair["score"] == pytest.approx(0.8)
    assert pair["revision_a"] == first["revision"]
    assert pair["revision_b"] == second["revision"]
    assert pair["group_id"] == group["group_id"]

    # The returned revisions match a fresh read of each page.
    for page_id, revision in revisions.items():
        assert client.get(f"/api/pages/{page_id}").json()["revision"] == revision


def test_threshold_boundary_and_empty_sets(client: TestClient):
    _put(client, "base", "t1 t2 t3 t4 t5")  # bigrams: 4
    _put(client, "exact", "t1 t2 t3 t4 t5 t6")  # 4/5 = 0.8 -> linked
    _put(client, "identical", "t1 t2 t3 t4 t5")  # 1.0 -> linked
    _put(client, "below", "t1 t2 t3 t4")  # 3/4 = 0.75 -> not linked
    _put(client, "single-1", "solitaire")  # empty bigram set
    _put(client, "single-2", "solitaire")  # identical, still never linked
    _put(client, "empty-1", "")
    _put(client, "empty-2", "")

    data = _duplicates(client)
    assert data["group_count"] == 1
    group_ids = [page["page_id"] for page in data["groups"][0]["pages"]]
    assert group_ids == ["base", "exact", "identical"]

    pair_ids = {(pair["page_a"], pair["page_b"]) for pair in data["pairs"]}
    assert pair_ids == {
        ("base", "exact"),
        ("base", "identical"),
        ("exact", "identical"),
    }
    scores = {(p["page_a"], p["page_b"]): p["score"] for p in data["pairs"]}
    assert scores[("base", "exact")] == pytest.approx(0.8)
    assert scores[("base", "identical")] == pytest.approx(1.0)
    assert scores[("exact", "identical")] == pytest.approx(0.8)


def test_rule_is_fixed_nfc_casefold_and_accent_sensitive(client: TestClient):
    _put(client, "accented", "Le café noir ensemble")
    _put(client, "upper", "LE CAFÉ NOIR ENSEMBLE")  # casefold -> linked
    _put(client, "decomposed", "Le café noir ensemble")  # NFC -> linked
    _put(client, "unaccented", "Le cafe noir ensemble")  # é != e -> not linked

    data = _duplicates(client)
    pair_ids = {(pair["page_a"], pair["page_b"]) for pair in data["pairs"]}
    assert pair_ids == {
        ("accented", "decomposed"),
        ("accented", "upper"),
        ("decomposed", "upper"),
    }
    assert [p["page_id"] for p in data["groups"][0]["pages"]] == [
        "accented",
        "decomposed",
        "upper",
    ]

    # Switch the search policy to fold: search folds accents, the duplicate
    # rule must not change.
    trigger = client.post("/api/index/rebuild", json={"accent_policy": "fold"})
    assert trigger.status_code == 202
    status = _wait_for_terminal(client, trigger.json()["job"]["job_id"])
    assert status["active_policy"] == "fold"

    folded = client.get("/api/search", params={"q": "cafe"}).json()
    assert {hit["page_id"] for hit in folded["hits"]} >= {"accented", "unaccented"}

    after = _duplicates(client)
    assert after["as_of_change_seq"] == data["as_of_change_seq"]
    assert {(p["page_a"], p["page_b"]) for p in after["pairs"]} == pair_ids
    assert after["group_count"] == 1


def test_bridge_page_correction_and_delete_regroups(client: TestClient):
    _put(client, "page-a", "w01 w02 w03 w04 w05 w06 w07 w08 w09 w10")
    _put(client, "page-b", "w02 w03 w04 w05 w06 w07 w08 w09 w10 w11")
    _put(client, "page-c", "w03 w04 w05 w06 w07 w08 w09 w10 w11 w12")

    data = _duplicates(client)
    # a~b and b~c at exactly 0.8; a~c is ~0.64 but connected through b.
    assert data["group_count"] == 1
    assert [p["page_id"] for p in data["groups"][0]["pages"]] == [
        "page-a",
        "page-b",
        "page-c",
    ]
    assert {(p["page_a"], p["page_b"]) for p in data["pairs"]} == {
        ("page-a", "page-b"),
        ("page-b", "page-c"),
    }

    # Correcting the bridge breaks both links; the group splits apart.
    _put(client, "page-b", "zulu yankee xray quebec uniform")
    data = _duplicates(client)
    assert data["group_count"] == 0
    assert data["pairs"] == []
    assert data["linked_page_count"] == 0

    # Restoring the bridge rebuilds the same group.
    _put(client, "page-b", "w02 w03 w04 w05 w06 w07 w08 w09 w10 w11")
    assert _duplicates(client)["group_count"] == 1

    # Deleting the bridge re-splits the groups; the other pages survive.
    assert client.delete("/api/pages/page-b").status_code == 200
    data = _duplicates(client)
    assert data["group_count"] == 0
    assert data["pairs"] == []
    assert data["page_count"] == 2
    assert client.get("/api/pages/page-b").status_code == 404
    assert client.get("/api/pages/page-a").status_code == 200


def test_update_moves_page_between_groups(client: TestClient):
    _put(client, "one", "red green blue yellow orange")
    _put(client, "two", "red green blue yellow orange purple")
    _put(client, "drifter", "mercury venus earth mars saturn")

    assert _duplicates(client)["group_count"] == 1

    # Rewrite the drifter so it joins the first group.
    _put(client, "drifter", "red green blue yellow orange pink")
    data = _duplicates(client)
    assert data["group_count"] == 1
    assert [p["page_id"] for p in data["groups"][0]["pages"]] == [
        "drifter",
        "one",
        "two",
    ]

    # Rewrite it again into something unique; it leaves the group.
    _put(client, "drifter", "completely fresh text without overlap")
    data = _duplicates(client)
    assert data["group_count"] == 1
    assert [p["page_id"] for p in data["groups"][0]["pages"]] == ["one", "two"]
    assert {(p["page_a"], p["page_b"]) for p in data["pairs"]} == {("one", "two")}


def test_concurrent_writes_never_miss_links(client: TestClient):
    from app.archive import upsert_page

    def body_for(index: int) -> str:
        # Sliding windows of width 10: neighbours share 8 of 10 bigrams (0.8),
        # pages two apart share 7 of 11 (~0.64) and must stay unlinked.
        return " ".join(f"w{n:02d}" for n in range(index, index + 10))

    count = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futures = [
            pool.submit(upsert_page, f"conc-{i:02d}", f"Concurrent {i}", body_for(i))
            for i in range(count)
        ]
        for future in futures:
            future.result(timeout=30)

    data = _duplicates(client)
    assert data["group_count"] == 1
    group = data["groups"][0]
    assert [p["page_id"] for p in group["pages"]] == [f"conc-{i:02d}" for i in range(count)]

    pair_ids = {(pair["page_a"], pair["page_b"]) for pair in data["pairs"]}
    expected = {(f"conc-{i:02d}", f"conc-{i + 1:02d}") for i in range(count - 1)}
    assert pair_ids == expected

    # Revisions in the response are the committed ones.
    for page in group["pages"]:
        fresh = client.get(f"/api/pages/{page['page_id']}").json()
        assert page["revision"] == fresh["revision"]


def test_failed_relation_step_rolls_back_everything(client: TestClient, monkeypatch):
    from app import archive
    from app.db import connect

    _put(client, "keep-1", "red green blue yellow orange")
    before = _duplicates(client)

    def count_rows() -> tuple[int, int, int]:
        conn = connect()
        try:
            changes = conn.execute("SELECT COUNT(*) FROM changes").fetchone()[0]
            signatures = conn.execute("SELECT COUNT(*) FROM page_signatures").fetchone()[0]
            pairs = conn.execute("SELECT COUNT(*) FROM similar_pairs").fetchone()[0]
            return int(changes), int(signatures), int(pairs)
        finally:
            conn.close()

    counts_before = count_rows()

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated relation-store failure")

    monkeypatch.setattr(archive, "apply_upsert_relations", boom)
    with pytest.raises(RuntimeError, match="simulated relation-store failure"):
        client.put(
            "/api/pages/keep-2",
            json={"title": "t", "body": "red green blue yellow orange purple"},
        )
    monkeypatch.undo()

    # Nothing changed: no page, no change-log entry, no FTS row, no relations.
    assert client.get("/api/pages/keep-2").status_code == 404
    assert client.get("/api/search", params={"q": "purple"}).json()["total"] == 0
    assert count_rows() == counts_before
    assert _duplicates(client) == before

    # With the failure gone the very same write commits cleanly and links.
    _put(client, "keep-2", "red green blue yellow orange purple")
    data = _duplicates(client)
    assert {(p["page_a"], p["page_b"]) for p in data["pairs"]} == {("keep-1", "keep-2")}


def test_preexisting_pages_are_backfilled(client: TestClient):
    _put(client, "old-1", "mercury venus earth mars jupiter")
    _put(client, "old-2", "mercury venus earth mars jupiter saturn")
    assert _duplicates(client)["pair_count"] == 1

    # Simulate a database written before this feature existed.
    from app.db import connect

    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM page_signatures")
        conn.execute("DELETE FROM similar_pairs")
        conn.execute("DELETE FROM settings WHERE key = 'signatures_seq'")
        conn.commit()
    finally:
        conn.close()

    from app.similar import ensure_similarities_current

    assert ensure_similarities_current() is True
    assert ensure_similarities_current() is False  # already current: no-op

    data = _duplicates(client)
    assert data["group_count"] == 1
    assert [p["page_id"] for p in data["groups"][0]["pages"]] == ["old-1", "old-2"]
    assert data["pairs"][0]["score"] == pytest.approx(0.8)


def test_delete_on_drifted_database_backfills_first(client: TestClient):
    _put(client, "old-1", "mercury venus earth mars jupiter")
    _put(client, "old-2", "mercury venus earth mars jupiter saturn")
    _put(client, "old-3", "unrelated page that will be deleted")

    # Simulate a pre-feature database, then delete a page before any other
    # request touches the similarity tables.
    from app.db import connect

    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM page_signatures")
        conn.execute("DELETE FROM similar_pairs")
        conn.execute("DELETE FROM settings WHERE key = 'signatures_seq'")
        conn.commit()
    finally:
        conn.close()

    assert client.delete("/api/pages/old-3").status_code == 200

    # The delete advanced the change log, and the relations must still be
    # complete for the surviving pre-existing pages.
    data = _duplicates(client)
    assert data["group_count"] == 1
    assert [p["page_id"] for p in data["groups"][0]["pages"]] == ["old-1", "old-2"]
    assert {(p["page_a"], p["page_b"]) for p in data["pairs"]} == {("old-1", "old-2")}
