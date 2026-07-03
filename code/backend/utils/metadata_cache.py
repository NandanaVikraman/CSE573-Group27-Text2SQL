import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any

import fcntl


class MetadataCacheLockError(RuntimeError):
    """Raised when a writer cannot acquire the metadata cache lock."""


def load_metadata_cache(cache_path: str) -> dict[str, Any]:
    with open(cache_path, "r", encoding="utf-8") as cache_file:
        payload = json.load(cache_file)

    if not isinstance(payload, dict):
        raise ValueError("Metadata cache payload must be a JSON object.")

    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("Metadata cache payload is missing a valid 'entries' object.")

    return payload


def build_metadata_cache_payload(
    *,
    cache_schema_version: int,
    entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "cache_schema_version": cache_schema_version,
        "entry_count": len(entries),
        "entries": entries,
    }


def write_metadata_cache_atomic(
    cache_path: str,
    *,
    cache_schema_version: int,
    entries: dict[str, dict[str, Any]],
) -> None:
    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)

    payload = build_metadata_cache_payload(
        cache_schema_version=cache_schema_version,
        entries=entries,
    )

    fd, temp_path = tempfile.mkstemp(
        dir=cache_dir,
        prefix=".phase1_metadata_cache_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            json.dump(payload, temp_file, indent=2, sort_keys=True)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, cache_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@contextmanager
def metadata_cache_write_lock(cache_path: str):
    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)

    lock_path = f"{cache_path}.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MetadataCacheLockError(lock_path) from exc

        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
