"""Run every assert script in scripts/ and report one summary.

The tests here are plain top-level assert scripts, not pytest cases. This runs
each one in its own subprocess from the repo root, so a failure in one cannot
hide the others, and returns a non-zero exit code if any failed.

    .venv\\Scripts\\python.exe scripts\\run_tests.py
    .venv\\Scripts\\python.exe scripts\\run_tests.py --only gold
    .venv\\Scripts\\python.exe scripts\\run_tests.py --list

Some tests need `data/tracker.db` or a frozen export and will fail on a clean
checkout. That is reported, not hidden.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "scripts"
TIMEOUT_S = 600


def discover(pattern: str | None) -> list[Path]:
    found = sorted(TESTS_DIR.glob("test_*.py"))
    if pattern:
        found = [p for p in found if pattern in p.stem]
    return found


def run_one(path: Path) -> tuple[str, float, str]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.monotonic() - started, f"exceeded {TIMEOUT_S}s"

    elapsed = time.monotonic() - started
    if proc.returncode == 0:
        return "PASS", elapsed, ""

    detail = (proc.stderr or proc.stdout or "").strip()
    tail = "\n".join(detail.splitlines()[-12:])
    return "FAIL", elapsed, tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run only tests whose name contains this substring")
    parser.add_argument("--list", action="store_true", help="list discovered tests and exit")
    args = parser.parse_args()

    tests = discover(args.only)
    if not tests:
        print("no tests matched")
        return 1

    if args.list:
        for path in tests:
            print(path.name)
        return 0

    failures: list[tuple[str, str]] = []
    total_started = time.monotonic()

    for path in tests:
        status, elapsed, detail = run_one(path)
        print(f"{status:<7} {path.name:<34} {elapsed:6.1f}s")
        if status != "PASS":
            failures.append((path.name, detail))

    total = time.monotonic() - total_started
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed in {total:.1f}s")

    for name, detail in failures:
        print(f"\n--- {name} ---")
        print(detail or "(no output)")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
