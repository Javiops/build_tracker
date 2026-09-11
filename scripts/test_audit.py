"""Regression tests for the 2026-09-09 audit.

Each of these pins down a mistake that shipped once and produced a number
somebody believed. They are cheap; run them before touching gold, splits,
labels or evaluation.

    .venv\\Scripts\\python.exe scripts\\test_audit.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import baseline  # noqa: E402
import train_prefix as tp  # noqa: E402
from app.ddragon import default_dragon  # noqa: E402
from app.shop_econ import SAVE_ITEM  # noqa: E402

dragon = default_dragon()


# ---------------------------------------------------------------- 2. parity
# A harness that hand-copies featurization flags will eventually drop one; the
# sampled gold harness dropped USE_GOLDX and would have scored a GOLDX model on
# a stale budget without failing.
config = {
    "extras": True, "history": False, "gold_est": True, "gold_x": True,
    "runes": True, "query_dim": 55, "save_splice": False,
}
tp.apply_config(config)
assert (tp.USE_GOLDX, tp.USE_RUNES, tp.USE_GOLDEST, tp.QUERY_DIM) == (True, True, True, 55)
tp.apply_config({})
assert tp.USE_GOLDX is False and tp.USE_RUNES is False
assert tp.QUERY_DIM == tp.BASE_QUERY_DIM
import inspect  # noqa: E402

predictor_src = (ROOT / "app" / "predictor.py").read_text(encoding="utf-8")
assert "tp.apply_config(config)" in predictor_src, "Predictor must go through apply_config"
for harness in ("eval_live_gold_sampled.py", "eval_policy.py"):
    src = (ROOT / "scripts" / harness).read_text(encoding="utf-8")
    assert "apply_config" in src or "Predictor(" in src, f"{harness} bypasses the config path"
assert "USE_GOLDX" in inspect.getsource(tp.apply_config)
print("ok parity: one function sets featurization for training, serving and evals")


# ------------------------------------------------------- 3. GOLDX version gate
BASE_ROW = {
    "match_id": "KR_T", "champion": "Ahri", "champion_id": 103, "role": "MIDDLE",
    "team_id": 100, "gold": 400, "total_gold": 4000, "ally_obj": {}, "enemy_obj": {},
    "level": 6, "ts": 600000, "kills": 0, "deaths": 0, "inventory": [1056],
    "label_id": 1058, "label_ids": [1058], "can_complete": 0, "n_completable": 0,
    "cheapest_complete": 0, "gold_after_complete": 400, "n_inventory": 1,
    "decision": "component", "save_kind": "build_spend", "others": [],
}
champ_index, item_index = {103: 1}, {1056: 1, 1058: 2}
label_index = {1058: 0, SAVE_ITEM: 1, 2055: 2}

tp.apply_config({"gold_x": True})
for bad in (None, "leaky-v1", "prequential-v1"):
    row = {**BASE_ROW, "gold_est": 1500, "gold_est_version": bad}
    try:
        tp.ShopDataset([row], champ_index, item_index, label_index, dragon)
    except ValueError as exc:
        assert "gold_est_version" in str(exc)
    else:
        raise AssertionError(f"GOLDX accepted gold_est_version={bad!r}")
try:
    tp.ShopDataset([{**BASE_ROW, "gold_est": None, "gold_est_version": "prequential-v2"}],
                   champ_index, item_index, label_index, dragon)
except ValueError:
    pass
else:
    raise AssertionError("GOLDX accepted a null gold_est")
for good in ("prequential-v2", "live-exact-v2"):
    ds = tp.ShopDataset([{**BASE_ROW, "gold_est": 1500, "gold_est_version": good}],
                        champ_index, item_index, label_index, dragon)
    assert float(ds.budget[0]) == 1500.0, (good, float(ds.budget[0]))
tp.apply_config({})  # without GOLDX the same untagged row is fine, on stale gold
ds = tp.ShopDataset([{**BASE_ROW, "gold_est": 1500}], champ_index, item_index, label_index, dragon)
assert float(ds.budget[0]) == 400.0
print("ok gold gate: GOLDX refuses untagged/legacy rows instead of mixing regimes")


# ------------------------------------------------------------- 4. frozen split
created = {f"KR_{i}": 1_700_000_000_000 + i * 60_000 for i in range(100)}
created["KR_tie_a"] = created["KR_50"]  # simultaneous games must not be split
created["KR_tie_b"] = created["KR_50"]
assignment = baseline.temporal_split(created)
assert assignment["KR_tie_a"] == assignment["KR_tie_b"] == assignment["KR_50"]
splits = Counter(assignment.values())
assert splits["train"] > splits["val"] and splits["test"] >= 1, splits


def stamps(assign: dict[str, str], created_map: dict[str, int], split: str) -> list[int]:
    return [created_map[m] for m, s in assign.items() if s == split]


train_ts, val_ts, test_ts = (stamps(assignment, created, s) for s in ("train", "val", "test"))
assert max(train_ts) <= min(val_ts), "train must be older than val"
assert max(val_ts) <= min(test_ts), "val must be older than test"
groups = {split: {m for m, s in assignment.items() if s == split} for split in ("train", "val", "test")}
assert not (groups["train"] & groups["val"]) and not (groups["train"] & groups["test"])
assert not (groups["val"] & groups["test"])

with tempfile.TemporaryDirectory() as tmpdir:
    out = Path(tmpdir)
    old_out, old_manifest, old_assign = baseline.OUT_DIR, baseline.MANIFEST_PATH, baseline.ASSIGNMENT_PATH
    baseline.OUT_DIR = out
    baseline.MANIFEST_PATH = out / "split_manifest.json"
    baseline.ASSIGNMENT_PATH = out / "split_assignment.json"
    try:
        first, manifest = baseline.load_or_build_split(created, rebuild=False)
        baseline.ASSIGNMENT_PATH.write_text(json.dumps(first), encoding="utf-8")
        baseline.MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")
        grown = dict(created)
        for i in range(100, 130):  # games newer than the whole split arrive
            grown[f"KR_{i}"] = 1_700_000_000_000 + i * 60_000
        second, manifest2 = baseline.load_or_build_split(grown, rebuild=False)

        # nothing already assigned may move
        for match_id, split in first.items():
            assert second[match_id] == split, f"{match_id} moved {split} -> {second[match_id]}"
        # and nothing newer than the cutoff may enter ANY split — parking those
        # in train would put post-test games in the training set, which is the
        # contamination freezing the ids was supposed to prevent
        for i in range(100, 130):
            assert second[f"KR_{i}"] == "post_cutoff", (
                f"KR_{i} landed in {second[f'KR_{i}']}; newer games must be excluded, not trained on"
            )
        for split in ("train", "val", "test"):
            newest_in_split = max(stamps(second, grown, split))
            oldest_post = min(stamps(second, grown, "post_cutoff"))
            assert newest_in_split <= oldest_post, f"{split} contains a post-cutoff game"
        assert manifest2["frozen"] is True
        assert manifest2["post_cutoff_count"] == 30, manifest2["post_cutoff_count"]
        assert manifest2["cutoff_utc"] and manifest2["id_sha256"]["post_cutoff"]
        assert manifest["id_sha256"]["test"] == manifest2["id_sha256"]["test"], "test set changed"

        # route_split must drop post_cutoff rows from every artifact
        tmp_rows = out / "rows.jsonl"
        with tmp_rows.open("w", encoding="utf-8") as handle:
            for match_id in list(first)[:20] + [f"KR_{i}" for i in range(100, 105)]:
                handle.write(json.dumps({"match_id": match_id, "champion": "Ahri", "label_id": 3006}) + "\n")
        _bc, _gc, _val, rows_per_split = baseline.route_split(tmp_rows, second)
        assert rows_per_split.get("post_cutoff") == 5, rows_per_split
        written = sum(
            len((out / f"visits_{name}.jsonl").read_text(encoding="utf-8").strip().splitlines() or [])
            for name in ("train", "val", "test")
        )
        assert written == 20, f"{written} rows written, post_cutoff rows leaked into an artifact"

        # a rebuild starts a new evaluation version and archives the old manifest
        _third, manifest3 = baseline.load_or_build_split(grown, rebuild=True)
        assert manifest3["eval_version"] == manifest2["eval_version"] + 1, manifest3["eval_version"]
        assert manifest3["supersedes"]["test_id_sha256"] == manifest2["id_sha256"]["test"]
        assert (out / f"split_manifest_v{manifest2['eval_version']}.json").exists()
        assert manifest3["post_cutoff_count"] == 0, "a rebuild should absorb the backlog"
    finally:
        baseline.OUT_DIR, baseline.MANIFEST_PATH, baseline.ASSIGNMENT_PATH = old_out, old_manifest, old_assign
print("ok split: temporal ordering holds, newer games are excluded not trained on, rebuild is versioned")


# ------------------------------------------------ 5. no_buy_death vs ward_only
def example_for(bought: list[int], is_save: bool) -> dict:
    event = {
        "is_save": is_save,
        "bought": [{"item_id": i, "item_name": "x", "skip": False} for i in bought],
        "inventory_before": [{"item_id": 1056, "item_name": "x", "skip": False}],
        "board": [{"is_self": True, "champion_id": 103, "team_id": 100, "gold": 900, "items": []}],
        "gold": 900, "gold_left": 900, "level": 8, "cs": 40, "kills": 1, "deaths": 1,
        "assists": 0, "ts": 600000, "team_id": 100, "champion_name": "Ahri",
        "champion_id": 103, "team_position": "MIDDLE", "match_id": "KR_T", "score": {},
        "gold_est": 950, "gold_est_version": "prequential-v2",
    }
    return baseline._example(event, dragon)


no_buy = example_for([], True)
ward = example_for([2055], False)
build = example_for([1058], False)
assert no_buy["save_kind"] == "no_buy_death" and no_buy["label_ids"] == [SAVE_ITEM]
assert ward["save_kind"] == "ward_only" and ward["label_ids"] == [2055]
assert build["save_kind"] == "build_spend"
assert ward["label_ids"] != no_buy["label_ids"], "ward-only must keep real item labels"

tp.apply_config({})
rows = [
    {**BASE_ROW, "label_ids": [SAVE_ITEM], "label_id": SAVE_ITEM, "decision": "save", "save_kind": "no_buy_death"},
    {**BASE_ROW, "label_ids": [2055], "label_id": 2055, "decision": "save", "save_kind": "ward_only"},
]
ds = tp.ShopDataset(rows, champ_index, item_index, label_index, dragon)
assert int(ds.is_save[0]) == 1, "no_buy_death must train the save head"
assert int(ds.is_save[1]) == 0, "ward_only must NOT train the save head as a no-buy"
print("ok save: no_buy_death and ward_only are different actions end to end")


# -------------------------------------------- 6. policy eval strips the labels
import eval_policy  # noqa: E402

row_with_labels = {**BASE_ROW, "label_ids": [1058], "target_id": 3004, "gold_est": 1500,
                   "gold_est_version": "prequential-v2"}
served = eval_policy.serving_row(row_with_labels, budget=1500)
for field in eval_policy.LABEL_FIELDS:
    assert field not in served, f"{field} leaked into the served row"
assert served["gold"] == served["gold_est"] == 1500.0

tp.apply_config({})
labelled = tp.ShopDataset([{**BASE_ROW, "gold": 100, "label_ids": [1058]}],
                          champ_index, item_index, label_index, dragon)
stripped = tp.ShopDataset([{k: v for k, v in BASE_ROW.items() if k not in ("label_id", "label_ids")}
                           | {"gold": 100}], champ_index, item_index, label_index, dragon)
idx = label_index[1058]
assert bool(labelled.legal[0, idx]) is True, "training must keep the true label legal"
assert bool(stripped.legal[0, idx]) is False, (
    "an unaffordable item became legal without a label — stripping labels has no effect, "
    "so the policy evaluator would be scoring a privileged model"
)
print("ok policy eval: labels are stripped, and stripping actually removes the hint")


# --------------------------- 6b. point estimate and CI share one population
# The first version reported first_exact over purchases only while its bootstrap
# also consumed the no_buy_death visits, so the interval described a different
# quantity than the number beside it. One purchase (a miss) and one no-buy (a
# hit) is the minimal case that separates the two populations.
outcomes = {
    "game_1": [
        {  # a purchase the policy got wrong
            "first_action_exact": 0.0,
            "first_basket_exact_given_purchase": 0.0,
            "no_buy_death_recall": None,
        },
        {  # a no-buy the policy got right: defined for the action metric only
            "first_action_exact": 1.0,
            "first_basket_exact_given_purchase": None,
            "no_buy_death_recall": 1.0,
        },
    ]
}
action = eval_policy.summarize(outcomes, "first_action_exact", reps=200, seed=1)
basket = eval_policy.summarize(outcomes, "first_basket_exact_given_purchase", reps=200, seed=1)
no_buy = eval_policy.summarize(outcomes, "no_buy_death_recall", reps=200, seed=1)
assert action["n"] == 2 and action["value"] == 0.5, action
assert basket["n"] == 1 and basket["value"] == 0.0, basket
assert no_buy["n"] == 1 and no_buy["value"] == 1.0, no_buy
# the sharp one: every purchase in this population is a miss, so no resample can
# score above zero. A bootstrap that had swept in the no-buy hit would.
assert basket["ci95"][1] == 0.0, (
    f"basket CI upper bound {basket['ci95'][1]} > 0 — the interval is drawing from visits the "
    "point estimate excludes"
)
assert action["ci95"][0] <= action["value"] <= action["ci95"][1], action

# paired comparison: same games both sides, and a real difference is detected
worse = {"game_1": [{"first_action_exact": 0.0}, {"first_action_exact": 0.0}]}
better = {"game_1": [{"first_action_exact": 1.0}, {"first_action_exact": 1.0}]}
diff = eval_policy.paired_diff(better, worse, "first_action_exact", reps=200, seed=1)
assert diff["diff"] == 1.0 and diff["ci95"] == [1.0, 1.0] and diff["significant"] is True, diff
tie = eval_policy.paired_diff(better, better, "first_action_exact", reps=200, seed=1)
assert tie["diff"] == 0.0 and tie["significant"] is False, tie
print("ok policy metrics: point estimate and CI95 share one population; paired diff works")


# -------------------------------------------- 6c. save-card metric population
# The product may show Save as a situational alternative rather than the lead
# action. save_card_precision is about every displayed Save card, whereas
# first_action_exact and no_buy_death_recall remain explicitly first-action
# metrics. Keep those meanings separate.
class _AlternativeSavePredictor:
    def predict_options(self, _row, budget_slack=0, allow_save=False):
        assert budget_slack == 0
        assert allow_save is True
        return [
            {"kind": "buy", "items": [{"item_id": 1058, "cost": dragon.gold_block(1058)["total"]}]},
            {"kind": "save", "items": []},
        ]


alt_save, over = eval_policy.score_visit(
    _AlternativeSavePredictor(), BASE_ROW, Counter({1058: 1}), "build_spend",
    2000, dragon, None,
)
assert not over
assert alt_save["first_action_exact"] == 1.0, alt_save
assert alt_save["save_card_precision"] == 0.0, alt_save
assert alt_save["no_buy_death_recall"] is None, alt_save
print("ok save-card precision: counts a displayed alternative without changing lead-action metrics")


# ------------------------------------- 8. stability language / decomposition
import eval_stability  # noqa: E402

row_a = {"inventory": [1056], "level": 6, "gold": 400, "others": [], "kills": 0,
         "deaths": 0, "ally_obj": {}, "enemy_obj": {}}
row_b = {**row_a, "gold": 1300}          # only gold moved
row_c = {**row_a, "kills": 1}            # board moved
assert eval_stability.board_signature(row_a) == eval_stability.board_signature(row_b), (
    "board signature must ignore gold — otherwise every poll looks like a state change"
)
assert eval_stability.board_signature(row_a) != eval_stability.board_signature(row_c)
# 1058 (Long Sword line component) costs more than 400 and less than 1300, so a
# gold move across it is an affordability crossing, not flicker.
assert eval_stability.affordability_moved(row_a, row_b, {1058}, dragon) is True
assert eval_stability.affordability_moved(row_a, {**row_a, "gold": 410}, {1058}, dragon) is False
src = (ROOT / "scripts" / "eval_stability.py").read_text(encoding="utf-8")
assert "NOT 'unchanged game state'" in src, "the loose condition must be labelled as loose"
print("ok stability: inventory+level equality is not called an unchanged game state")

print()
print("audit suite passed")
