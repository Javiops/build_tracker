"""Deliberately approve a validation-evaluated artifact for the live overlay.

Run this only after copying the chosen candidate to data/ml/prefix_model.pt.
It binds the exact served bytes to the frozen-split validation policy report;
the app refuses to emit advice unless that binding is present and valid.

    .venv\\Scripts\\python.exe scripts\\promote_served_model.py \
        --report policy_eval_val_candidate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from app.config import DATA_DIR
from app.deployment import DEPLOYMENT_MANIFEST_PATH, REQUIRED_PROVENANCE, sha256_file

ML_DIR = DATA_DIR / "ml"
SPLIT_MANIFEST_PATH = ML_DIR / "split_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="prefix_model.pt", help="served artifact in data/ml")
    parser.add_argument("--report", required=True, help="validation report filename in data/ml")
    parser.add_argument("--replace", action="store_true", help="replace an existing deployment manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = ML_DIR / args.artifact
    report_path = ML_DIR / args.report
    if not artifact.exists() or not report_path.exists() or not SPLIT_MANIFEST_PATH.exists():
        raise SystemExit("artifact, validation policy report, and split_manifest.json must all exist.")
    if DEPLOYMENT_MANIFEST_PATH.exists() and not args.replace:
        raise SystemExit(
            f"{DEPLOYMENT_MANIFEST_PATH} already exists. Review the promotion and pass --replace to change serving."
        )

    from app.predictor import Predictor
    import eval_policy

    predictor = Predictor(artifact)
    if predictor.provenance.get("probe") or predictor.provenance.get("status") == "probe_only":
        raise SystemExit("early-probe artifacts are diagnostic-only and can never be promoted.")
    split_manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    checked = eval_policy.verify_artifact_provenance(predictor, split_manifest, allow_legacy=False)
    if checked["status"] != "verified":
        raise SystemExit("only a provenance-verified candidate can be promoted.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("split") != "val":
        raise SystemExit("promotion needs a schema-v1 validation policy report, never a test or legacy report.")
    if report.get("artifact_sha256", {}).get("A") != predictor.artifact_sha256:
        raise SystemExit("the report's A artifact digest does not match the bytes being served.")
    if report.get("artifact_provenance", {}).get("A", {}).get("status") != "verified":
        raise SystemExit("the report did not verify the candidate against the frozen split manifest.")
    if int(report.get("metrics", {}).get("A", {}).get("unaffordable", -1)) != 0:
        raise SystemExit("cannot promote: the displayed-policy report failed the exact-budget hard stop.")
    if int(report.get("visits") or 0) <= 0:
        raise SystemExit("cannot promote an empty evaluation report.")

    approved = {key: predictor.provenance.get(key) for key in REQUIRED_PROVENANCE}
    payload = {
        "schema_version": 1,
        "artifact_name": artifact.name,
        "artifact_sha256": predictor.artifact_sha256,
        "artifact_provenance": approved,
        "evaluation_report": report_path.name,
        "evaluation_report_sha256": sha256_file(report_path),
        "evaluation_split": "val",
        "unaffordable_recommendations": 0,
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    DEPLOYMENT_MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"promoted {artifact.name} ({predictor.artifact_sha256[:12]}) for explicit live shop sessions")


if __name__ == "__main__":
    main()
