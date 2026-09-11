"""Entry-point holds precede DB/export work and training side effects."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import baseline
import train_prefix
import export_clean_probe
import train as comparison_train
import app.pipeline_guard as guard

def forbidden():
    raise AssertionError("side effect reached despite hold")

with tempfile.TemporaryDirectory() as directory:
    hold = Path(directory) / "hold.json"
    hold.write_text("operator hold", encoding="utf-8")
    old = guard.HOLD_PATH
    old_dragon, old_speed = baseline.default_dragon, train_prefix._windows_full_speed
    try:
        guard.HOLD_PATH = hold
        baseline.default_dragon = forbidden
        train_prefix._windows_full_speed = forbidden
        for entry in (baseline.main, train_prefix.main, export_clean_probe.main, comparison_train.main):
            try:
                entry()
            except SystemExit as exc:
                assert "PIPELINE HOLD" in str(exc), str(exc)
            else:
                raise AssertionError("held entry point returned normally")
        # Absence permits the guard only, never invoke real training here.
        guard.require_pipeline_clear("test", Path(directory) / "absent")
    finally:
        guard.HOLD_PATH = old
        baseline.default_dragon, train_prefix._windows_full_speed = old_dragon, old_speed
print("ok pipeline hold: export and training stop before work; no ingestion hook")
