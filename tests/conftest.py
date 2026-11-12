"""Test configuration: point every test at an isolated SQLite file."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="ocr-archive-tests-"))
os.environ.setdefault("ARCHIVE_DB_PATH", str(_TEST_ROOT / "bootstrap.db"))
os.environ.setdefault("REBUILD_BATCH_DELAY_MS", "120")
os.environ.setdefault("REBUILD_BATCH_SIZE", "2")
os.environ.setdefault("REBUILD_LEASE_SECONDS", "3")

# Must be imported after the environment above is set.
from app.config import SETTINGS  # noqa: E402


def configure_database(path: Path) -> None:
    object.__setattr__(SETTINGS, "db_path", path)
    object.__setattr__(SETTINGS, "rebuild_batch_delay_ms", 120)
    object.__setattr__(SETTINGS, "rebuild_batch_size", 2)
    object.__setattr__(SETTINGS, "rebuild_lease_seconds", 3)


def pytest_sessionfinish(session, exitstatus):  # noqa: ANN001
    import shutil

    shutil.rmtree(_TEST_ROOT, ignore_errors=True)
