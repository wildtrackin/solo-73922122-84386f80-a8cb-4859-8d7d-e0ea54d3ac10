# OCR Accent-Aware Archive Search

FastAPI + SQLite FTS5 service for an OCR archive whose searchable accent
policy can be changed online. Page corrections and deletes continue to commit
while a new index is being built; the old index serves searches until the new
generation has caught up and is switched atomically.

## Start with Docker Compose

```bash
docker compose up --build -d
```

Open **<http://localhost:8000/docs>**. The container runs exactly two uvicorn
API worker processes against one WAL-mode SQLite file on the named volume
`archive-data`.

```bash
docker compose ps
docker compose exec api ps -ef | grep '[u]vicorn'
```

The `api` container is the single writer-capable deployment; SQLite is not
shared between separate containers. The two processes are uvicorn workers in
this one container.

To make a rebuild visibly long enough to submit edits by hand during it:

```bash
REBUILD_BATCH_DELAY_MS=250 docker compose up --build -d
```

The delay is a demo-only aid and is `0` by default. Reset demo data with:

```bash
docker compose down -v
```

## Swagger demo

Every step is available with **Try it out** on `/docs`. The OpenAPI description
also contains the full demo narrative and accented request examples.

### 1. Write two accented OCR pages under the initial `keep` policy

```bash
BASE=http://localhost:8000

curl -sS -X PUT "$BASE/api/pages/ocr-001" \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Crème brûlée à Paris",
    "body": "Le café est près de la gare; le señor lit un naïve résumé."
  }'

curl -sS -X PUT "$BASE/api/pages/ocr-002" \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "A plain cafe",
    "body": "This page uses cafe and naive without accents."
  }'
```

### 2. Search with the initial exact-accent rule

```bash
curl --get "$BASE/api/search" --data-urlencode 'q=café'
# ocr-001 is returned.

curl --get "$BASE/api/search" --data-urlencode 'q=cafe'
# Only ocr-002 matches. The accented café is not folded under keep.
```

Word queries are literal words. Multiple words are ANDed; FTS query operators
from the request are never interpreted.

### 3. Trigger the background switch to `fold`

```bash
curl -sS -X POST "$BASE/api/index/rebuild" \
  -H 'Content-Type: application/json' \
  -d '{"accent_policy": "fold"}'
```

The response is HTTP `202` and contains a `job.job_id`. A duplicate trigger
while the job is active returns the **same** job ID with
`"reused_existing_rebuild": true`.

### 4. Correct and delete during the rebuild

Run these while `GET /api/index/status` reports `building` or `catching_up`:

```bash
# Correction: replaces the page and becomes visible in both the old live index
# and, after catch-up, the new generation.
curl -sS -X PUT "$BASE/api/pages/ocr-001" \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Crème brûlée — corrected OCR",
    "body": "Le café fermé est près de la gare; le señor relit le résumé corrigé."
  }'

# Deletion must remain deleted; the old snapshot scan must not revive it.
curl -sS -X DELETE "$BASE/api/pages/ocr-002"
```

These writes commit immediately. The rebuild scan does not hold a lock for its
duration, so editing is not blocked by it.

### 5. Watch the live status and old index

```bash
curl -sS "$BASE/api/index/status"
curl --get "$BASE/api/search" --data-urlencode 'q=cafe'
```

While the job is running, `active_policy` is still `keep`,
`rebuild_in_progress` is `true`, and `current_rebuild_job` shows progress:
`queued → building → catching_up → activating → succeeded`.

### 6. Query after the switch

Once `current_rebuild_job` is `null` and `active_policy` is `fold`:

```bash
curl --get "$BASE/api/search" --data-urlencode 'q=cafe'
# Returns ocr-001 with the corrected body: café matches cafe.

curl --get "$BASE/api/search" --data-urlencode 'q=resume'
# Folded résumé matches.
```

Deleted `ocr-002` remains absent, and returned titles/bodies are the corrected
committed versions from the same SQLite snapshot as the hit.

## Accent examples

| Document word | `keep` query | `fold` query |
|---|---|---|
| café | café | cafe or café |
| señor | señor | senor or señor |
| naïve | naïve | naive or naïve |
| résumé | résumé | resume or résumé |
| Crème brûlée | crème / brûlée | creme / brulee |
| İstanbul | İstanbul | Istanbul |

`keep` normalises composed/decomposed spellings to NFC but never removes a
diacritic. `fold` removes combining marks attached to Latin base letters only;
non-Latin combining marks are preserved.

## API summary

| Method | Path | Purpose |
|---|---|---|
| `PUT` | `/api/pages/{page_id}` | Insert/replace an OCR title and body. |
| `DELETE` | `/api/pages/{page_id}` | Idempotent delete. |
| `GET` | `/api/pages/{page_id}` | Read a committed page. |
| `GET` | `/api/search?q=word` | Word search over the active FTS generation. |
| `POST` | `/api/index/rebuild` | Queue/switch accent policy through a background rebuild. |
| `GET` | `/api/index/status` | Active policy/generation, current job, progress, history. |
| `GET` | `/api/index/jobs/{job_id}` | One rebuild job. |
| `GET` | `/healthz` | Health and active policy. |

`POST /api/index/rebuild` also accepts `?accent_policy=fold` for convenience.
If the body/parameter is omitted, the opposite of the active policy is used.
The alias `POST /api/index/switch` is available but omitted from the Swagger
schema.

## Correctness model

* **Durable source of truth** — `pages` stores the latest committed title/body
  and `changes` is an append-only sequence log containing the full content of
  every write/delete.
* **One active generation** — each generation owns a separate FTS5 table.
  `settings.active_generation_id` and `settings.active_policy` are switched in
  the same short transaction that marks the new generation active.
* **Writes during rebuild** — a write transaction updates `pages`, appends to
  `changes`, and updates the currently active FTS table. It never writes to a
  generation under construction. Therefore no scan-length write lock is needed,
  and a failed worker cannot corrupt the serving index.
* **Snapshot baseline** — the rebuild reads its baseline in a pinned WAL read
  transaction. Writers continue to commit in WAL while that snapshot is open.
* **Catch-up** — the worker applies every change-log row after the snapshot,
  in sequence, to its private shadow FTS table. It does this in small write
  transactions, with heartbeats.
* **Atomic swap** — the final short `BEGIN IMMEDIATE` transaction applies the
  last residual changes, verifies shadow/page document counts, updates the
  pointer/policy, and marks the job succeeded. Only then does a new request see
  the new generation and policy.
* **Snapshot-consistent reads** — a search begins one SQLite read transaction,
  reads the active pointer/policy, the FTS hits and the canonical bodies in that
  same transaction. Hits and returned text therefore come from one committed
  state, even if a swap commits concurrently.
* **One rebuild across processes** — `rebuild_jobs` has a partial unique index
  on active states. Claiming is done under `BEGIN IMMEDIATE`, so two API
  processes cannot start two jobs; the loser returns the current job.
* **Failure/interruption** — the previous active generation is never changed
  until final commit. A failed job marks only its shadow generation failed. A
  new process marks a dead owner's job interrupted at startup (dead PID or a
  different container host), so the rebuild can be retriggered immediately.
  Canonical pages and the change log survive in SQLite.
* **Fencing** — the final activation re-checks the job's claim token and
  active state under the write lock, so a stale worker that resumes cannot
  switch an obsolete generation after a newer job has been claimed.

FTS rowids are taken from the canonical page rowid and every log entry carries
that rowid, making updates/deletes O(index lookup) rather than rescanning the
FTS table by page ID.

Retired FTS tables are deliberately retained rather than dropped in the swap
transaction. This avoids schema-lock surprises for an already-running query;
generation history is visible in `/api/index/status`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ARCHIVE_DB_PATH` | `./data/archive.db` locally, `/data/archive.db` in Docker | SQLite path. |
| `REBUILD_LEASE_SECONDS` | `20` | Heartbeat lease before a trigger may reclaim a job. |
| `REBUILD_BATCH_SIZE` | `500` | Scan/catch-up batch size. |
| `REBUILD_BATCH_DELAY_MS` | `0` | Demo-only pacing per batch. |
| `DB_BUSY_TIMEOUT_SECONDS` | `30` | SQLite writer wait. |

## Local development and tests

```bash
python3 -m pip install -r requirements.txt -r requirements-dev.txt
ARCHIVE_DB_PATH=/tmp/archive-dev.db \
  uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 2
```

```bash
pytest -q
```

The tests exercise exact vs folded search, a rebuild that finishes with a
correction and a deletion submitted during the scan, duplicate job idempotency,
and restart/stale-job recovery. They use temporary SQLite files.
