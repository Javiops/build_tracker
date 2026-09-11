"""A process-held lease for serial Riot collection on this machine."""
import os
from pathlib import Path
from app.config import DATA_DIR


class CollectorLease:
    def __init__(self, path: Path | None = None):
        self.path = path or DATA_DIR / 'riot_collector.lock'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        self.handle.seek(0, 2)
        if not self.handle.tell():
            self.handle.write(b'0')
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError('Another Riot collector holds the lease; retry after it finishes') from None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.handle.close()
