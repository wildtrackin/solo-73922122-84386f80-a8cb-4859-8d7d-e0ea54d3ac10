"""Runtime configuration, all overridable with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    db_path: Path
    rebuild_lease_seconds: float = 30.0
    rebuild_batch_size: int = 500
    rebuild_batch_delay_ms: int = 0
    db_busy_timeout_seconds: float = 30.0


def get_settings() -> Settings:
    default_path = Path(__file__).resolve().parent.parent / "data" / "archive.db"
    return Settings(
        db_path=Path(os.environ.get("ARCHIVE_DB_PATH", str(default_path)).strip() or default_path),
        rebuild_lease_seconds=_env_float("REBUILD_LEASE_SECONDS", 30.0),
        rebuild_batch_size=max(1, _env_int("REBUILD_BATCH_SIZE", 500)),
        rebuild_batch_delay_ms=max(0, _env_int("REBUILD_BATCH_DELAY_MS", 0)),
        db_busy_timeout_seconds=max(1.0, _env_float("DB_BUSY_TIMEOUT_SECONDS", 30.0)),
    )


SETTINGS = get_settings()
