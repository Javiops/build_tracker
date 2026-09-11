"""Durable local artifact publication; unfinished generations remain invisible."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def replace_retry(source: Path, destination: Path):
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * 2 ** attempt)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.partial')
    with pending.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    # Preserve pending bytes on final failure for explicit recovery.
    replace_retry(pending, path)


def publish_directory(pending: Path, destination: Path):
    if destination.exists():
        raise FileExistsError(f'Immutable generation already exists: {destination}')
    # A same-volume directory rename exposes the whole completed generation.
    # Never use replace on an existing generation.
    pending.rename(destination)
