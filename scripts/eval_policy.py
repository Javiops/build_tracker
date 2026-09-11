"""Score the policy the user actually sees: Predictor.predict_options.

Every accuracy number this project has published came from predict_top3 or
predict_baskets against `label_id`, a canonical item picked by baseline._label
(the completed item, else the priciest). Neither is what ships: the overlay runs
predict_options, which applies a plan probe, recipe-consistency filters, three
alternatives with the lead banned, an exact-budget decode and a save card. This
harness scores that, and nothing else, as the primary product metric.

Rules it enforces that earlier harnesses did not:

  * labels are stripped before a row reaches the Predictor. ShopDataset marks
    the true labels legal so training never masks a positive; leaving them in
    hands the evaluated policy a hint no live row carries.
  * one budget feeds both the model's feature and the decoder, with zero slack.
  * every metric's point estimate and its bootstrap interval are computed over
    THE SAME population. Mixing them (a point over purchases, an interval over
    purchases plus no-buys) describes two different quantities and invites the
    reader to compare them.

Budget consistency is not exactness. Offline there is no exact gold: the number
used is `gold_est`, a causal estimate. The point is only that the feature and
the decoder cannot disagree about what the player can afford. Nothing here
measures live accuracy (HANDOFF.md §14).

    .venv\\Scripts\\python.exe scripts\\eval_policy.py [--split val] [--games 300]
    .venv\\Scripts\\python.exe scripts\\eval_policy.py --artifact A.pt --compare B.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.ddragon import default_dragon
from app.shop_econ import PINK, SAVE_ITEM, components_of, try_purchase

sys.path.insert(0, str(ROOT / "scripts"))
import train_prefix as tp  # noqa: E402

ML_DIR = Path(os.environ["PREFIX_ML_DIR"]).resolve() if os.environ.get("PREFIX_ML_DIR") else DATA_DIR / "ml"
RUN_DIR = Path(os.environ.get('PREFIX_RUN_DIR',str(ML_DIR))).resolve()
TEST_USAGE_LOG = RUN_DIR / "test_set_usage.log"
SPLIT_MANIFEST_PATH = ML_DIR / "split_manifest.json"
LABEL_FIELDS = (
    "label_id",
    "label_ids",
    "label_counts",
    "target_encoding",
    "label_name",
    "label_completed",
    "decision",
    "save_kind",
    "target_id",
    "gold_arrival_true",
)


def manifest_sha256(path: Path) -> str:
    """Hash the manifest's logical UTF-8 text, not platform line endings.

    ``Path.write_text`` writes CRLF on Windows while ``read_text`` normalises
    it to LF. Training correctly binds the normalised text; policy evaluation
    must use the identical representation or a harmless newline conversion
    falsely looks like a provenance violation.
    """
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
# Every metric declares the population it is defined on. A visit contributes
# None where the metric does not apply, and None is filtered identically by the
# point estimate and the bootstrap — that identity is the whole point.
METRICS = {
    "first_action_exact": "all scored visits: SAVE right on a no-buy, or the exact basket on a purchase",
    "first_basket_exact_given_purchase": "visits with a real basket only",
    "first_basket_precision_given_purchase": "visits with a real basket only",
    "first_basket_recall_given_purchase": "visits with a real basket only",
    "any_of_three_exact_given_purchase": "visits with a real basket only",
    "action_family_match": "purchases whose next completed final is known",
    "no_buy_death_recall": "no_buy_death visits only",
    "save_card_precision": "visits where a save card was shown in any displayed option",
    "ward_only_recall": "ward_only visits only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="prefix_model.pt", help="file in data/ml")
    parser.add_argument(
        "--compare",
        default=None,
        help="second artifact: reports A-B differences with a game-PAIRED bootstrap, "
        "which is the test of whether one candidate beats another (non-overlapping "
        "individual intervals is not that test)",
    )
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--games", type=int, default=300, help="games to sample (0 = all)")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=400, help="resamples (0 = off)")
    parser.add_argument("--confirm-final", action="store_true")
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="forensic only: score an artifact missing required split/gold/policy provenance; never use it for selection",
    )
    return parser.parse_args()


def serving_row(row: dict, budget: float | None) -> dict:
    """A row as production would see it: no labels, one budget everywhere."""
    out = {k: v for k, v in row.items() if k not in LABEL_FIELDS}
    if budget is not None:
        out["gold"] = float(budget)
        out["gold_est"] = float(budget)
    return out


def _ci(samples: list[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    if not ordered:
        return (float("nan"), float("nan"))
    return (
        ordered[int(0.025 * (len(ordered) - 1))],
        ordered[int(0.975 * (len(ordered) - 1))],
    )


def _defined(per_game: dict[str, list[dict]], key: str) -> dict[str, list[float]]:
    """The population of a metric: visits where it is defined, grouped by game."""
    out: dict[str, list[float]] = {}
    for match_id, visits in per_game.items():
        values = [v[key] for v in visits if v.get(key) is not None]
        if values:
            out[match_id] = values
    return out


def _mean(per_game: dict[str, list[float]]) -> tuple[float, int]:
    total = sum(sum(v) for v in per_game.values())
    n = sum(len(v) for v in per_game.values())
    return (total / n if n else float("nan"), n)


def summarize(per_game: dict[str, list[dict]], key: str, reps: int, seed: int) -> dict:
    """Point estimate and CI95 over one population, resampling GAMES.

    Visits inside a game are heavily correlated, so a per-visit interval would
    be far too narrow. Both numbers come from the same filtered population, so
    the interval always describes the value printed next to it.
    """
    population = _defined(per_game, key)
    mean, n = _mean(population)
    result = {"metric": key, "population": METRICS.get(key, ""), "n": n, "value": mean}
    keys = list(population)
    if reps > 0 and keys:
        rng = random.Random(seed)
        means = []
        for _ in range(reps):
            total = count = 0.0
            for _ in range(len(keys)):
                values = population[keys[rng.randrange(len(keys))]]
                total += sum(values)
                count += len(values)
            means.append(total / count if count else 0.0)
        lo, hi = _ci(means)
        result["ci95"] = [lo, hi]
    return result


def paired_diff(
    a: dict[str, list[dict]], b: dict[str, list[dict]], key: str, reps: int, seed: int
) -> dict:
    """CI95 of (A - B) resampling the SAME games for both.

    Comparing two independent intervals answers a different question and is far
    too conservative: the candidates are scored on identical visits, so the
    difference has much less variance than either estimate alone.
    """
    pa, pb = _defined(a, key), _defined(b, key)
    shared = sorted(set(pa) & set(pb))
    if not shared:
        return {"metric": key, "n_games": 0}
    mean_a, _ = _mean({k: pa[k] for k in shared})
    mean_b, _ = _mean({k: pb[k] for k in shared})
    out = {
        "metric": key,
        "population": METRICS.get(key, ""),
        "n_games": len(shared),
        "a": mean_a,
        "b": mean_b,
        "diff": mean_a - mean_b,
    }
    if reps > 0:
        rng = random.Random(seed)
        diffs = []
        for _ in range(reps):
            ta = tb = ca = cb = 0.0
            for _ in range(len(shared)):
                game = shared[rng.randrange(len(shared))]
                ta += sum(pa[game]); ca += len(pa[game])
                tb += sum(pb[game]); cb += len(pb[game])
            diffs.append((ta / ca if ca else 0.0) - (tb / cb if cb else 0.0))
        lo, hi = _ci(diffs)
        out["ci95"] = [lo, hi]
        out["significant"] = bool(lo > 0 or hi < 0)
    return out


def check_displayed_purchases(options: list[dict], row: dict, budget: float, dragon) -> list[dict]:
    """Replay each displayed alternative independently of decoder bookkeeping.

    Reprice from pinned recipes rather than trusting displayed costs. Invalid
    slots, stacks, restrictions or misleading prices abort evaluation; wallet
    violations remain explicit in the existing unaffordable-report counter.
    """
    over_budget = []
    for option in options:
        entries = option.get("items") or []
        if option.get("kind") == "save":
            if entries:
                raise ValueError("Displayed save action contains purchases")
            continue
        if option.get("kind") != "buy":
            raise ValueError("Unknown displayed action kind")
        inventory = list(row.get("inventory") or [])
        total = 0
        protected = Counter()
        for entry in entries:
            item_id = int(entry["item_id"])
            purchase = try_purchase(item_id, inventory, float("inf"), dragon,
                                    role=row.get("role") or "",
                                    bot_quest_complete=row.get("bot_quest_complete"),
                                    unmodeled_regular_slots=row.get("unmodeled_regular_slots") or 0,
                                    role_slot_inventory_unknown=bool(row.get("role_slot_inventory_unknown")),
                                    protected=protected)
            if purchase is None:
                raise ValueError(f"Displayed purchase {item_id} violates inventory legality")
            cost, owned = purchase
            if entry.get("cost") != cost:
                raise ValueError(f"Displayed purchase {item_id} reports the wrong recipe cost")
            inventory = list(owned.elements())
            total += cost
            protected[item_id] += 1
        if total > float(budget):
            over_budget.append({"cost": total, "budget": float(budget)})
    return over_budget


def score_visit(predictor, row: dict, actual: Counter, save_kind: str, budget: float, dragon,
                family: set[int] | None) -> tuple[dict, list[dict]]:
    """One visit through the deployed policy → per-metric outcomes (None where
    the metric does not apply to this visit)."""
    # A policy evaluation represents an explicit shop decision, matching the
    # only production context where a hold card may be shown.
    options = predictor.predict_options(serving_row(row, budget), budget_slack=0, allow_save=True)
    first = options[0] if options else {"kind": "buy", "items": []}
    pred = Counter(int(i["item_id"]) for i in first.get("items") or [])
    predicted_save = first.get("kind") == "save"
    save_card_shown = any(option.get("kind") == "save" for option in options)

    over_budget = check_displayed_purchases(options, row, budget, dragon)

    truth = Counter({k: v for k, v in actual.items() if k != SAVE_ITEM})
    is_no_buy = save_kind == "no_buy_death"
    out: dict[str, float | None] = {key: None for key in METRICS}

    # defined on every visit: did the policy pick the right ACTION
    if is_no_buy:
        out["first_action_exact"] = float(predicted_save)
    else:
        out["first_action_exact"] = float(not predicted_save and pred == truth)

    if not is_no_buy and truth:
        hit = sum((truth & pred).values())
        out["first_basket_exact_given_purchase"] = float(pred == truth)
        out["first_basket_recall_given_purchase"] = hit / sum(truth.values())
        out["first_basket_precision_given_purchase"] = hit / sum(pred.values()) if pred else 0.0
        any_exact = 0.0
        for opt in options:
            if Counter(int(i["item_id"]) for i in opt.get("items") or []) == truth:
                any_exact = 1.0
                break
        out["any_of_three_exact_given_purchase"] = any_exact
        if family:
            out["action_family_match"] = float(bool(pred) and any(i in family for i in pred))
    if is_no_buy:
        out["no_buy_death_recall"] = float(predicted_save)
    # This measures the card the user was shown, not just the lead action.
    # A save card can deliberately be the situational third option while a
    # buying line remains first; scoring only `predicted_save` made the named
    # precision metric silently exclude those displayed cards.
    if save_card_shown:
        out["save_card_precision"] = float(is_no_buy)
    if save_kind == "ward_only":
        out["ward_only_recall"] = float(any(i in PINK for i in pred))
    return out, over_budget


def verify_artifact_provenance(predictor, manifest: dict, allow_legacy: bool) -> dict:
    """Refuse metrics that cannot be tied to the export they claim to score."""
    provenance = predictor.provenance
    if provenance.get("status") == "legacy_unverified":
        if not allow_legacy:
            raise SystemExit(
                f"{predictor.artifact.name} is legacy/unverified: {provenance.get('reason')}. "
                "It has no binding to a split or gold-input regime. Re-train on the frozen export, "
                "or pass --allow-legacy for forensic output that must not select a model."
            )
        return {"status": "legacy_forensic_only", "provenance": provenance}
    expected = {
        "manifest_sha256": manifest_sha256(SPLIT_MANIFEST_PATH),
        "export_fingerprint": manifest.get("export_fingerprint"),
        "eval_version": manifest.get("eval_version"),
    }
    # Legacy probe reports remain inspectable, but newly versioned artifacts
    # must carry the complete reconstruction/static-data binding through eval.
    if provenance.get("reconstruction_version"):
        expected.update({
            "reconstruction_version": tp.RECONSTRUCTION_VERSION,
            "patch": manifest.get("patch"),
            "ddragon_version": manifest.get("ddragon_version"),
            "static_data_sha256": manifest.get("static_data_sha256"),
        })
    mismatches = {
        key: {"artifact": provenance.get(key), "current": value}
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    config_gold = predictor.config.get("gold_input") or (
        "prequential-v2" if predictor.config.get("gold_x") else "stale-frame-gold"
    )
    if provenance.get("gold_input") != config_gold:
        mismatches["gold_input"] = {"artifact": provenance.get("gold_input"), "config": config_gold}
    if provenance.get("training_split") != "train" or provenance.get("selection_split") != "validation":
        mismatches["split_protocol"] = {
            "training_split": provenance.get("training_split"),
            "selection_split": provenance.get("selection_split"),
        }
    if mismatches:
        raise SystemExit(
            f"{predictor.artifact.name} provenance does not match the current frozen export: "
            f"{json.dumps(mismatches, sort_keys=True)}. Do not compare across manifests."
        )
    status = "probe_only" if provenance.get("probe") else "verified"
    return {"status": status, "provenance": provenance}


def main() -> None:
    args = parse_args()
    if (ML_DIR / 'generation_manifest.json').exists():
        from app.dataset_generations import verify_generation, validate_run_directory
        verify_generation(ML_DIR)
        validate_run_directory(ML_DIR,RUN_DIR)
        if args.split != 'val':
            raise SystemExit('Pipeline smoke artifacts may only evaluate validation')
    path = ML_DIR / f"visits_{args.split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing — run scripts/baseline.py first.")
    if args.split == "test" and not args.confirm_final:
        raise SystemExit(
            "Refusing to read the final test set without --confirm-final.\n"
            "Selection belongs on --split val. Every test read is appended to "
            f"{TEST_USAGE_LOG}, because a test set looked at repeatedly stops being one."
        )
    if not SPLIT_MANIFEST_PATH.exists():
        raise SystemExit(f"{SPLIT_MANIFEST_PATH} missing — run scripts/baseline.py first.")
    manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))

    from app.predictor import Predictor

    models = [("A", args.artifact, Predictor(RUN_DIR / args.artifact))]
    if args.compare:
        models.append(("B", args.compare, Predictor(RUN_DIR / args.compare)))
    provenance_checks = {
        tag: verify_artifact_provenance(predictor, manifest, args.allow_legacy)
        for tag, _name, predictor in models
    }
    for tag, name, predictor in models:
        config = predictor.config
        gold_input = config.get("gold_input") or (
            "prequential-v2" if config.get("gold_x") else "stale-frame-gold"
        )
        print(
            f"{tag}: {name} (trained {predictor.trained_at})  gold_input={gold_input}  "
            f"tau={predictor.threshold}  provenance={provenance_checks[tag]['status']}",
            flush=True,
        )
    print(f"split={args.split}", flush=True)

    if args.split == "test":
        TEST_USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TEST_USAGE_LOG.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{datetime.now().isoformat(timespec='seconds')}\t"
                f"{'+'.join(n for _t, n, _p in models)}\tgames={args.games}\n"
            )

    dragon = models[0][2].dragon
    if any(p.dragon.signature != dragon.signature for _tag, _name, p in models):
        raise SystemExit("Cannot compare artifacts using different static data")
    all_games: list[str] = []
    seen: set[str] = set()
    for row in tp.stream_rows(path):
        if row.get("match_id") not in seen:
            seen.add(row["match_id"])
            all_games.append(row["match_id"])
    rng = random.Random(args.seed)
    rng.shuffle(all_games)
    chosen = set(all_games if args.games <= 0 else all_games[: args.games])
    print(f"scoring {len(chosen)} of {len(all_games)} games", flush=True)

    finals_cache: dict[int, set[int]] = {}

    def family_of(target_id: int) -> set[int]:
        if target_id not in finals_cache:
            finals_cache[target_id] = components_of(target_id, dragon) | {target_id}
        return finals_cache[target_id]

    outcomes: dict[str, dict[str, list[dict]]] = {tag: defaultdict(list) for tag, _n, _p in models}
    over_budget: dict[str, list[dict]] = {tag: [] for tag, _n, _p in models}
    scored = 0
    for row in tp.with_targets(tp.stream_rows(path), dragon):
        if row.get("match_id") not in chosen:
            continue
        actual = Counter(int(i) for i in (row.get("label_ids") or []) if i)
        if not actual:
            continue
        save_kind = row.get("save_kind") or (
            "no_buy_death" if SAVE_ITEM in actual
            else ("ward_only" if all(i in PINK for i in actual) else "build_spend")
        )
        target_id = int(row.get("target_id") or 0)
        family = family_of(target_id) if target_id else None
        for tag, name, predictor in models:
            use_gold_est = bool(predictor.config.get("gold_x"))
            budget = row.get("gold_est") if use_gold_est else row.get("gold")
            if budget is None:
                budget = row.get("gold")
            try:
                result, over = score_visit(predictor, row, actual, save_kind, budget, dragon, family)
            except ValueError as exc:  # Input-version or displayed-purchase hard stop.
                raise SystemExit(f"cannot score this export with {name}: {exc}")
            outcomes[tag][row["match_id"]].append(result)
            over_budget[tag].extend(over)
        scored += 1
        if scored % 2000 == 0:
            print(f"  …{scored} visits", flush=True)

    report: dict = {
        "schema_version": 1,
        "split": args.split,
        "visits": scored,
        "games": len(outcomes["A"]),
        "artifacts": {tag: name for tag, name, _p in models},
        "artifact_sha256": {tag: predictor.artifact_sha256 for tag, _name, predictor in models},
        "artifact_provenance": provenance_checks,
        "manifest": {
            "eval_version": manifest.get("eval_version"),
            "export_fingerprint": manifest.get("export_fingerprint"),
            "probe": bool(manifest.get("probe")),
        },
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": {},
        "purchase_legality": {
            "version": "role-slots-v1",
            "checks": ["recipe_cost", "wallet", "regular_slots", "ward_stacks", "unique_boots", "purchase_blocks"],
            "scope": "supplied inventory and observed slot state; missing completion grants no extra capacity",
        },
    }
    print(flush=True)
    print(f"scored {scored} visits from {len(outcomes['A'])} games", flush=True)
    for tag, name, _p in models:
        print(flush=True)
        print(f"=== {tag}: {name} ===", flush=True)
        summaries = {}
        for key in METRICS:
            summary = summarize(outcomes[tag], key, args.bootstrap, args.seed)
            summaries[key] = summary
            if not summary["n"]:
                continue
            ci = summary.get("ci95")
            ci_txt = f"  CI95 [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            print(f"  {key:<38} {summary['value']:.3f}{ci_txt}   n={summary['n']}", flush=True)
        print(
            f"  {'unaffordable recommendations':<38} {len(over_budget[tag])}   (must be 0)",
            flush=True,
        )
        summaries["unaffordable"] = len(over_budget[tag])
        report["metrics"][tag] = summaries

    if args.compare:
        print(flush=True)
        print("=== A - B, paired bootstrap by game (the comparison that counts) ===", flush=True)
        diffs = {}
        for key in METRICS:
            diff = paired_diff(outcomes["A"], outcomes["B"], key, args.bootstrap, args.seed)
            if not diff.get("n_games"):
                continue
            diffs[key] = diff
            ci = diff.get("ci95")
            ci_txt = f"  CI95 [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
            verdict = "significant" if diff.get("significant") else "not significant"
            print(
                f"  {key:<38} {diff['diff']:+.3f}{ci_txt}   {verdict}  "
                f"(A {diff['a']:.3f} vs B {diff['b']:.3f})",
                flush=True,
            )
        report["paired_diff_a_minus_b"] = diffs
        print(flush=True)
        print(
            "A candidate wins only where the paired interval excludes zero. "
            "Overlapping individual intervals prove nothing either way.",
            flush=True,
        )

    stem = Path(args.artifact).stem + (f"_vs_{Path(args.compare).stem}" if args.compare else "")
    out = RUN_DIR / f"policy_eval_{args.split}_{stem}.json"
    if out.exists():
        raise SystemExit('Policy evidence already exists; use a new run directory')
    from app.dataset_artifacts import write_json
    write_json(out,report)
    print(flush=True)
    print(f"wrote {out}", flush=True)

    # A number without <metric, split, n, report path> is inadmissible, so print
    # the envelope next to the numbers rather than leaving it to be reassembled
    # from scrollback. Quote this block whenever you quote a metric above.
    print(flush=True)
    print("=" * 72, flush=True)
    print("CITATION — copy this with any number above", flush=True)
    for tag, name, predictor in models:
        config = predictor.config
        gold_input = config.get("gold_input") or (
            "prequential-v2" if config.get("gold_x") else "stale-frame-gold"
        )
        print(
            f"  {tag}: {name}  sha256={predictor.artifact_sha256[:12]}  "
            f"gold_input={gold_input}  provenance={provenance_checks[tag]['status']}",
            flush=True,
        )
    print(
        f"  split={args.split}  games={len(outcomes['A'])}  visits={scored}  "
        f"bootstrap={args.bootstrap}  seed={args.seed}",
        flush=True,
    )
    print(f"  report={out}", flush=True)
    if not args.compare:
        print(
            "  UNPAIRED: single-artifact run. These values describe this "
            "artifact only;\n  they are not a comparison and no difference "
            "may be inferred from them.",
            flush=True,
        )
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
