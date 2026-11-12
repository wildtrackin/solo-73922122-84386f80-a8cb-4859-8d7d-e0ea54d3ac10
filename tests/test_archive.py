from __future__ import annotations

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


def _search(client: TestClient, query: str) -> dict:
    response = client.get("/api/search", params={"q": query})
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


def test_exact_then_folded_switch_with_correction_and_delete(client: TestClient):
    assert client.put(
        "/api/pages/ocr-001",
        json={
            "title": "Crème brûlée à Paris",
            "body": "Le café est près de la gare; le señor lit un naïve résumé.",
        },
    ).status_code == 200
    assert client.put(
        "/api/pages/ocr-002",
        json={"title": "A plain cafe", "body": "cafe naive without accents"},
    ).status_code == 200

    keep_cafe = _search(client, "café")
    assert keep_cafe["accent_policy"] == "keep"
    assert [hit["page_id"] for hit in keep_cafe["hits"]] == ["ocr-001"]

    plain_cafe = _search(client, "cafe")
    assert [hit["page_id"] for hit in plain_cafe["hits"]] == ["ocr-002"]

    trigger = client.post(
        "/api/index/rebuild", json={"accent_policy": "fold"}
    )
    assert trigger.status_code == 202, trigger.text
    first = trigger.json()
    assert first["created"] is True
    job_id = first["job"]["job_id"]

    duplicate = client.post(
        "/api/index/rebuild", json={"accent_policy": "fold"}
    )
    assert duplicate.status_code == 202
    assert duplicate.json()["job"]["job_id"] == job_id
    assert duplicate.json()["reused_existing_rebuild"] is True

    # These writes happen while the background scan/catch-up is in progress.
    corrected = client.put(
        "/api/pages/ocr-001",
        json={
            "title": "Crème brûlée — corrected",
            "body": "Le café fermé; le señor relit le résumé corrigé.",
        },
    )
    assert corrected.status_code == 200
    assert corrected.json()["accent_policy"] == "keep"
    destroyed = client.delete("/api/pages/ocr-002")
    assert destroyed.status_code == 200
    assert destroyed.json()["existed"] is True

    # The old generation still serves the correction immediately.
    live_old = _search(client, "café")
    assert live_old["accent_policy"] == "keep"
    assert [hit["page_id"] for hit in live_old["hits"]] == ["ocr-001"]
    assert "corrigé" in live_old["hits"][0]["body"]

    status = _wait_for_terminal(client, job_id)
    job = status["latest_rebuild_job"]
    assert job["state"] == "succeeded", job
    assert status["active_policy"] == "fold"
    assert status["rebuild_in_progress"] is False

    folded = _search(client, "cafe")
    assert folded["accent_policy"] == "fold"
    assert [hit["page_id"] for hit in folded["hits"]] == ["ocr-001"]
    assert folded["hits"][0]["body"] == (
        "Le café fermé; le señor relit le résumé corrigé."
    )
    assert folded["hits"][0]["revision"] == corrected.json()["revision"]

    assert _search(client, "resume")["hits"][0]["page_id"] == "ocr-001"
    # The word existed only in the stale baseline/deleted page.
    assert _search(client, "naive")["total"] == 0
    assert client.get("/api/pages/ocr-002").status_code == 404


def test_old_scan_cannot_resurrect_deleted_pages_or_overwrite_corrections(
    client: TestClient,
):
    for index in range(12):
        body = f"Le café numéro {index:02d} et le señor."
        if index == 11:
            body += " staleanchor"
        response = client.put(
            f"/api/pages/book-{index:02d}",
            json={
                "title": f"Archive page {index:02d}",
                "body": body,
            },
        )
        assert response.status_code == 200

    trigger = client.post(
        "/api/index/rebuild", json={"accent_policy": "fold"}
    ).json()
    job_id = trigger["job"]["job_id"]

    # Delete a page that the snapshot scan has already seen, and replace one
    # that it may still be about to see.
    assert client.delete("/api/pages/book-00").status_code == 200
    corrected = client.put(
        "/api/pages/book-11",
        json={
            "title": "Archive page 11 — revised",
            "body": "Le café remplacé; l'ancien mot disparaît.",
        },
    )
    assert corrected.status_code == 200

    status = _wait_for_terminal(client, job_id)
    assert status["latest_rebuild_job"]["state"] == "succeeded"
    assert status["active_policy"] == "fold"

    cafe = _search(client, "cafe")
    ids = [hit["page_id"] for hit in cafe["hits"]]
    assert "book-00" not in ids
    assert len(ids) == 11
    assert _search(client, "remplace")["hits"][0]["page_id"] == "book-11"
    assert _search(client, "disparait")["hits"][0]["page_id"] == "book-11"
    # The pre-correction text of book-11 must not survive in the new index.
    assert _search(client, "staleanchor")["total"] == 0


def test_interrupted_job_recovery_keeps_old_index_and_pages(tmp_path: Path):
    configure_database(tmp_path / "restart.db")
    from app.archive import search, upsert_page
    from app.db import claim_rebuild, connect, get_job, get_status, init_db
    from app.rebuild import start_rebuild_thread

    init_db()
    upsert_page("p1", "Café", "Le café et le señor.")

    claimed = claim_rebuild("fold")
    assert claimed["created"] is True
    interrupted_job = claimed["job"]
    assert get_status()["active_policy"] == "keep"

    # Simulate a container restart: the job owner is from a previous instance
    # and its heartbeat is fresh, but it is definitely not this process.
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE rebuild_jobs SET owner_host = 'previous-container', "
            "owner_pid = 999999, state = 'building' WHERE id = ?",
            (interrupted_job["job_id"],),
        )
        conn.commit()
    finally:
        conn.close()

    init_db()
    status = get_status()
    assert status["active_policy"] == "keep"
    assert status["active_generation_id"] == 1
    assert status["latest_rebuild_job"]["state"] == "interrupted"
    assert status["latest_rebuild_job"]["error"]

    retry = claim_rebuild("fold")
    assert retry["created"] is True
    job_id = retry["job"]["job_id"]
    start_rebuild_thread(
        job_id=job_id,
        generation_id=retry["job"]["generation_id"],
        policy="fold",
        claim_token=retry["worker_token"],
    )

    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        job = get_job(job_id)
        if job["state"] in {"succeeded", "failed", "interrupted"}:
            break
        time.sleep(0.03)
    else:
        raise AssertionError("retried rebuild did not finish")

    assert job["state"] == "succeeded", job
    status = get_status()
    assert status["active_policy"] == "fold"
    assert search("cafe", limit=10, offset=0)["hits"][0]["page_id"] == "p1"
    assert search("senor", limit=10, offset=0)["hits"][0]["body"] == "Le café et le señor."
