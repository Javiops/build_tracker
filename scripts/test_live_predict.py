"""Assert the deployed artifact drives the live path correctly.

Run after every deploy (and after touching predictor/live/train_prefix
featurization). Replays stored live-client snapshots through the real
Predictor and checks the invariants that silently break in production:

  * the artifact's config flags actually flip the module-level knobs
    train_prefix reads (a GOLDX artifact scored with USE_GOLDX off is
    fed a budget in the wrong regime and loses ~14pts — measured);
  * live rows carry exact gold in BOTH gold and gold_est;
  * every recommended basket is affordable at exact gold (the live UI
    must never suggest what the player cannot buy right now);
  * options are well-formed (a save card, or items with names/costs).

    .venv\\Scripts\\python.exe scripts\\test_live_predict.py [artifact.pt]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.live import snapshot_to_row
from app.predictor import ARTIFACT, Predictor

SNAP_DIR = DATA_DIR / "live_snapshots"


def main() -> None:
    artifact = Path(sys.argv[1]) if len(sys.argv) > 1 else ARTIFACT
    if not artifact.is_absolute():
        artifact = DATA_DIR / "ml" / artifact.name
    predictor = Predictor(artifact)
    config = predictor.config

    sys.path.insert(0, str(ROOT / "scripts"))
    import train_prefix as tp
    from eval_policy import check_displayed_purchases

    # config -> module knobs (the featurization must match training exactly)
    assert tp.USE_GOLDX == bool(config.get("gold_x", False)), "USE_GOLDX not synced from config"
    assert tp.USE_RUNES == bool(config.get("runes", False)), "USE_RUNES not synced from config"
    assert tp.USE_GOLDEST == bool(config.get("gold_est", False)), "USE_GOLDEST not synced"
    assert tp.QUERY_DIM == int(config.get("query_dim", tp.BASE_QUERY_DIM)), "QUERY_DIM mismatch"

    snaps = sorted(SNAP_DIR.glob("snap_*.json"))
    if len(snaps) < 3:
        raise SystemExit(f"need >=3 snapshots in {SNAP_DIR} (python -m app.live --dump)")
    picked = [snaps[0], snaps[len(snaps) // 3], snaps[len(snaps) // 2], snaps[-1]]

    scored = 0
    for path in picked:
        snap = json.loads(path.read_text(encoding="utf-8"))
        row = snapshot_to_row(snap, predictor.dragon)
        if not row:
            continue
        assert row["gold_est"] == row["gold"], "live row must carry exact gold in gold_est"
        # Production only produces an option inside a user-started shop
        # session. That is also the one context in which a historic no-buy
        # score may render an experimental hold card.
        options = predictor.predict_options(row, budget_slack=0, allow_save=True)
        assert options, f"no advice for {path.name}"
        assert not check_displayed_purchases(options, row, row["gold"], predictor.dragon)
        for opt in options:
            assert opt["kind"] in ("buy", "save"), opt
            if opt["kind"] == "save":
                assert not opt["items"]
                continue
            assert opt["items"], "buy option with no items"
            for item in opt["items"]:
                assert item.get("name"), item
                assert "prob" not in item and "score" not in item, "options view must not leak ranking scores"
            total = sum(item["cost"] for item in opt["items"])
            assert total <= row["gold"], f"{path.name}: basket {total}g over budget {row['gold']}g"
        scored += 1

    assert scored > 0, "no valid live snapshots were scored"
    print(
        f"ok live predict — {artifact.name} (trained {predictor.trained_at}), "
        f"gold_x={bool(config.get('gold_x'))} posw={config.get('posw_sched') or config.get('pos_weight_cap')} "
        f"tau={predictor.threshold} — {scored} snapshots"
    )


if __name__ == "__main__":
    main()
