"""Print what is actually happening in this repo, right now.

Replaces the hand-maintained "what is running" and "uncommitted work" sections
of HANDOFF.md, which went stale between sessions. Everything here is read from
the machine, so it cannot be wrong.

    .venv\\Scripts\\python.exe scripts\\status.py

Read-only: it starts nothing, stops nothing, and never prints a secret value.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOG_TAIL_LINES = 5


def rule(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 68 - len(title)))


def sh(cmd: list[str], timeout: int = 20) -> str:
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(unavailable: {exc})"
    return (proc.stdout or proc.stderr or "").strip()


def show_git() -> None:
    rule("git")
    print(sh(["git", "log", "-1", "--format=%h %s (%cr)"]) or "(no commits)")
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    print(f"branch: {branch}")
    changes = sh(["git", "status", "--short"])
    if not changes.strip():
        print("working tree clean")
        return
    lines = changes.splitlines()
    for line in lines[:25]:
        print(f"  {line}")
    if len(lines) > 25:
        print(f"  ... and {len(lines) - 25} more changed paths")


def show_scheduled_tasks() -> None:
    rule("scheduled tasks")
    if platform.system() != "Windows":
        print(f"(not Windows — {platform.system()}; skipped)")
        return
    out = sh(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-ScheduledTask -TaskName 'BuildTracker*' -ErrorAction SilentlyContinue "
            "| Select-Object TaskName,State | Format-Table -AutoSize | Out-String",
        ]
    )
    print(out or "(no BuildTracker* tasks registered)")


def show_locks() -> None:
    rule("locks and markers")
    watched = sorted(DATA.glob("riot_collector.lock")) + sorted(DATA.glob("refresh_done*.txt"))
    if not watched:
        print("(none present)")
        return
    for path in watched:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        print(f"  {path.name:<34} modified {stamp:%Y-%m-%d %H:%M UTC}")


def show_logs() -> None:
    rule(f"active logs (last {LOG_TAIL_LINES} lines)")
    logs = sorted(DATA.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        print("(no logs in data/)")
        return
    for path in logs[:4]:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        print(f"\n  {path.name} — last write {stamp:%Y-%m-%d %H:%M UTC}")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"    (unreadable: {exc})")
            continue
        for line in text.splitlines()[-LOG_TAIL_LINES:]:
            print(f"    {line}")


def show_served_artifact() -> None:
    rule("served model")
    manifest = DATA / "ml" / "deployment_manifest.json"
    if not manifest.exists():
        print("(no deployment_manifest.json — nothing is bound for serving)")
        return
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"(manifest unreadable: {exc})")
        return
    for key in ("artifact", "artifact_sha256", "report", "promoted_at", "split"):
        if key in payload:
            print(f"  {key:<16} {payload[key]}")
    unknown = [k for k in payload if k not in
               ("artifact", "artifact_sha256", "report", "promoted_at", "split")]
    if unknown:
        print(f"  (other keys: {', '.join(sorted(unknown))})")


def show_generations() -> None:
    rule("corpora and export generations")
    for label, folder in (("corpora", DATA / "corpora"),
                          ("generations", DATA / "dataset_generations"),
                          ("experiments", DATA / "experiments")):
        if not folder.is_dir():
            print(f"  {label:<13} (absent)")
            continue
        entries = sorted(p.name for p in folder.iterdir() if p.is_dir())
        if not entries:
            print(f"  {label:<13} (empty)")
        else:
            print(f"  {label:<13} {', '.join(entries[-6:])}"
                  + (f"  (+{len(entries) - 6} older)" if len(entries) > 6 else ""))


def show_env() -> None:
    rule("environment")
    print(f"  python        {sys.version.split()[0]} on {platform.system()}")
    env_file = ROOT / ".env"
    has_key = False
    if env_file.exists():
        try:
            has_key = any(
                line.strip().startswith("RIOT_API_KEY=") and line.strip() != "RIOT_API_KEY="
                for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines()
            )
        except OSError:
            has_key = False
    has_key = has_key or bool(os.environ.get("RIOT_API_KEY"))
    # Never print the value — dev keys are credentials and rotate every 24h.
    print(f"  RIOT_API_KEY  {'present' if has_key else 'MISSING'} (value not shown)")
    db = DATA / "tracker.db"
    if db.exists():
        print(f"  tracker.db    {db.stat().st_size / 1e9:.1f} GB")
    else:
        print("  tracker.db    absent")


def main() -> int:
    print(f"build_tracker status — {datetime.now(tz=timezone.utc):%Y-%m-%d %H:%M UTC}")
    show_git()
    show_scheduled_tasks()
    show_locks()
    show_logs()
    show_served_artifact()
    show_generations()
    show_env()
    print("\nNothing above was started or stopped by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
