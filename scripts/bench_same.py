"""same_{decision} benchmark: multiset intersection between the model's
predicted basket and what the player actually bought, per decision kind.

Usage: python scripts/bench_same.py data/ml/prefix_model_B.pt

Builds the test dataset fresh from the current export (no cache writes, so a
concurrently training run's caches are untouched), decodes budget-constrained
baskets exactly like training eval (count head only when the artifact trained
one), and reports intersection-size distributions plus gold-weighted and
order-aware views.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import torch

artifact = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "ml" / "prefix_model.pt"
blob = torch.load(artifact, map_location="cpu", weights_only=False)
cfg = blob.get("config") or {}

# featurization must match the artifact before train_prefix is imported
os.environ["PREFIX_GOLDEST"] = "1" if cfg.get("gold_est") else "0"
os.environ["PREFIX_RUNES"] = "1" if cfg.get("runes") else "0"
import train_prefix as tp  # noqa: E402

from app.ddragon import default_dragon  # noqa: E402
from app.shop_econ import GOLD_DRIFT, SAVE_ITEM, combine_cost  # noqa: E402

if cfg.get("runes"):
    while tp.QUERY_DIM > int(cfg["query_dim"]) and tp.KEYSTONE_IDS:
        tp.KEYSTONE_IDS.pop()  # artifact predates newer keystones
        tp.RUNE_DIM -= 1
        tp.QUERY_DIM -= 1
assert tp.QUERY_DIM == int(cfg.get("query_dim", tp.BASE_QUERY_DIM)), (tp.QUERY_DIM, cfg.get("query_dim"))

dragon = default_dragon()
test_path = ROOT / "data" / "ml" / "visits_test.jsonl"
labels = blob["label_ids"]
label_index = {i: idx for idx, i in enumerate(labels)}

n = sum(1 for _ in test_path.open(encoding="utf-8"))
slim: list[dict] = []


def gen():
    for row in tp.fix_champ_ids(tp.with_history(tp.stream_rows(test_path)), blob["champ_index"]):
        slim.append(
            {
                "decision": row.get("decision"),
                "label_ids": [int(i) for i in (row.get("label_ids") or []) if i],
                "label_id": row.get("label_id"),
                "match_id": row.get("match_id"),
                "champion_id": row.get("champion_id"),
                "ts": row.get("ts") or 0,
            }
        )
        yield row


print(f"building test tensors for {artifact.name} ({n} rows)…", flush=True)
ds = tp.ShopDataset.from_stream(gen(), n, blob["champ_index"], blob["item_index"], label_index, dragon)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = tp.PrefixModel(
    len(blob["champ_index"]), len(blob["item_index"]), len(labels), len(blob["item_index"]) + 1,
    d_model=int(cfg.get("d_model", 64)), layers=int(cfg.get("layers", 2)),
    ff_dim=int(cfg.get("ff_dim", 128)), heads=int(cfg.get("heads", 4)), query_dim=tp.QUERY_DIM,
).to(device)
missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
assert not unexpected, unexpected
use_counts = bool(cfg.get("counts"))
threshold = float(cfg.get("basket_threshold", tp.BASKET_THRESHOLD))
ds.to_device(device)
model.eval()

baskets: list[list[int]] = []
save_probs: list[float] = []
save_idx = label_index.get(SAVE_ITEM)
r = 0
with torch.no_grad():
    for batch in ds.batches(tp.BATCH, shuffle=False):
        out = model(
            batch["champs"], batch["sides"], batch["items"], batch["query"], batch["inv"], batch["hist"]
        )
        logits, count_logits = out["items"], out["counts"]
        if cfg.get("target_head") and float(cfg.get("target_blend") or 0) > 0:
            global _TM
            if "_TM" not in globals():
                _TM = tp.target_matrix(labels, dragon).to(device)
            logits = logits + float(cfg["target_blend"]) * torch.log(torch.softmax(out["target"], dim=1) @ _TM + 1e-4)
        if cfg.get("save_head"):
            save_probs.extend(torch.sigmoid(out["save"][:, 0]).cpu().tolist())
            if cfg.get("save_splice", True):  # splice after blend: plan mass must not punish SAVE
                logits = tp.splice_save(logits, out["save"], save_idx, tilt=float(cfg.get("save_tilt") or tp.SAVE_TILT))
        logits = logits.masked_fill(~batch["legal"], -1e4)
        probs = torch.sigmoid(logits).cpu()
        if save_idx is not None and not cfg.get("save_head"):
            save_probs.extend(probs[:, save_idx].tolist())
        want = (count_logits.argmax(dim=2) + 1).cpu() if use_counts else None
        for b in range(probs.size(0)):
            row_p = probs[b]
            budget = float(ds.budget[r]) + GOLD_DRIFT
            inv_c = Counter(ds.inventories[r])
            basket: list[int] = []
            for idx in torch.argsort(row_p, descending=True).tolist():
                if float(row_p[idx]) < threshold or len(basket) >= 5:
                    break
                item_id = labels[idx]
                if item_id == SAVE_ITEM:
                    if not basket:
                        basket = []  # model's primary call is: buy nothing
                    break
                copies = int(want[b, idx]) if want is not None else 1
                for _c in range(copies):
                    if len(basket) >= 5:
                        break
                    cost = combine_cost(item_id, inv_c, dragon)
                    if cost > budget:
                        break
                    combine_cost(item_id, inv_c, dragon, consume=True)
                    inv_c[item_id] += 1
                    budget -= cost
                    basket.append(item_id)
            baskets.append(basket)
            r += 1

print(f"decoded {len(baskets)} baskets (threshold {threshold}, counts {'on' if use_counts else 'off'})\n")

KINDS = ("complete", "component", "start")
hist: dict[str, Counter] = {k: Counter() for k in KINDS}
joint: dict[str, Counter] = {k: Counter() for k in KINDS}  # (actual_size, inter)
exact: dict[str, list] = defaultdict(lambda: [0, 0])
gold_hit: dict[str, list] = defaultdict(lambda: [0.0, 0.0])
first_hit: dict[str, list] = defaultdict(lambda: [0, 0])
save_stats = Counter()


def item_gold(item_id: int) -> float:
    g = dragon.gold_block(item_id)
    return float(g.get("total") or g.get("base") or 0)


for row, basket in zip(slim, baskets):
    kind = row["decision"]
    actual_list = [i for i in row["label_ids"] if i != SAVE_ITEM]
    if kind == "save":
        save_stats["n"] += 1
        save_stats["model_agrees (empty basket)"] += not basket
        continue
    if kind not in hist or not actual_list:
        continue
    actual = Counter(actual_list)
    pred = Counter(basket)
    inter = sum((actual & pred).values())
    asz = sum(actual.values())
    hist[kind][min(inter, 4)] += 1
    joint[kind][(min(asz, 4), min(inter, 4))] += 1
    exact[kind][0] += pred == actual
    exact[kind][1] += 1
    matched_gold = sum(item_gold(i) * c for i, c in (actual & pred).items())
    total_gold = sum(item_gold(i) * c for i, c in actual.items())
    gold_hit[kind][0] += matched_gold
    gold_hit[kind][1] += total_gold
    if basket and actual_list:
        first_hit[kind][0] += basket[0] == actual_list[0]
        first_hit[kind][1] += 1

for kind in KINDS:
    total = sum(hist[kind].values())
    if not total:
        continue
    print(f"same_{kind}  (n={total})")
    for size in range(0, 5):
        c = hist[kind][size]
        tag = f"{size}" if size < 4 else "4+"
        print(f"  intersection {tag}: {c:>7}  ({100 * c / total:5.1f}%)")
    print(f"  exact multiset match: {100 * exact[kind][0] / exact[kind][1]:.1f}%")
    print(f"  gold-weighted hit:    {100 * gold_hit[kind][0] / max(gold_hit[kind][1], 1):.1f}%  (matched item value / bought item value)")
    if first_hit[kind][1]:
        print(f"  first-buy order hit:  {100 * first_hit[kind][0] / first_hit[kind][1]:.1f}%  (basket head == first item bought)")
    print("  intersection by actual basket size (rows = they bought k items):")
    sizes = sorted({a for (a, _i) in joint[kind]})
    for a in sizes:
        rowc = Counter({i: joint[kind][(a, i)] for (aa, i) in joint[kind] if aa == a})
        tot = sum(rowc.values())
        cells = "  ".join(f"∩{i}:{100 * rowc.get(i, 0) / tot:4.0f}%" for i in range(0, min(a, 4) + 1))
        atag = f"{a}" if a < 4 else "4+"
        print(f"    bought {atag} ({tot:>6}): {cells}")
    print()

if save_stats["n"]:
    print(f"save visits (n={save_stats['n']}): model also said buy-nothing {100 * save_stats['model_agrees (empty basket)'] / save_stats['n']:.1f}%")

# ---------------------------------------------------------------- same_target
# A bought component's ground-truth target is the finished item THIS player
# completed later in THIS game whose recipe consumed it; a bought final is its
# own target. Plan agreement: any predicted item equals, or is a component of,
# any ground-truth target — "wrong piece, right plan" counts as agreement.
from app.shop_econ import components_of  # noqa: E402

comp_cache: dict[int, set] = {}


def comps(fid: int) -> set:
    if fid not in comp_cache:
        comp_cache[fid] = components_of(fid, dragon)
    return comp_cache[fid]


def is_final(item_id: int) -> bool:
    cls = dragon.classify(item_id)
    return bool(cls.get("is_completed") or cls.get("is_boots"))


# completions per (match, champion): [(ts, final_id)] in time order
completions: dict[tuple, list] = defaultdict(list)
for row in slim:
    for item in row["label_ids"]:
        if item != SAVE_ITEM and is_final(item):
            completions[(row["match_id"], row["champion_id"])].append((row["ts"], item))

target_stats = {k: Counter() for k in KINDS}
for row, basket in zip(slim, baskets):
    kind = row["decision"]
    if kind not in target_stats:
        continue
    actual_list = [i for i in row["label_ids"] if i != SAVE_ITEM]
    if not actual_list:
        continue
    timeline = completions[(row["match_id"], row["champion_id"])]
    targets: set[int] = set()
    for item in actual_list:
        if is_final(item):
            targets.add(item)
        else:
            for ts2, fid in timeline:
                if ts2 >= row["ts"] and item in comps(fid):
                    targets.add(fid)
                    break
    key_known = "known" if targets else "unknown"
    target_stats[kind][f"target_{key_known}"] += 1
    if not targets:
        continue
    hit = any(p in targets or any(p in comps(f) or p == f for f in targets) for p in set(basket))
    target_stats[kind]["hit"] += hit
    inter = sum((Counter(actual_list) & Counter(basket)).values())
    if inter == 0:
        target_stats[kind]["i0"] += 1
        target_stats[kind]["i0_hit"] += hit

print()
print("same_target  (plan agreement: prediction advances a ground-truth target)")
for kind in KINDS:
    st = target_stats[kind]
    known = st["target_known"]
    if not known:
        continue
    cov = known / (known + st["target_unknown"])
    print(f"  {kind:<10} target known {100 * cov:.0f}%  |  plan-hit {100 * st['hit'] / known:.1f}%"
          f"  |  plan-hit when item-intersection=0: {100 * st['i0_hit'] / max(st['i0'], 1):.1f}%  (n={st['i0']})")

# ------------------------------------------------------------ SAVE classifier
# The SAVE probability as a score for "this visit buys nothing", decoupled
# from the basket decode. AUC via rank statistic + operating points.
if save_idx is not None and save_probs:
    pairs = [(p, 1 if row["decision"] == "save" else 0) for p, row in zip(save_probs, slim)]
    pairs.sort(key=lambda x: x[0])
    n_pos = sum(y for _p, y in pairs)
    n_neg = len(pairs) - n_pos
    rank_sum = sum(i + 1 for i, (_p, y) in enumerate(pairs) if y)
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    print()
    print(f"SAVE classifier  (n={len(pairs)}, save rate {100 * n_pos / len(pairs):.1f}%)")
    print(f"  ROC AUC {auc:.3f}")
    print(f"  {'tau':>5} {'precision':>10} {'recall':>8} {'flagged':>9}")
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        flagged = [(p, y) for p, y in pairs if p >= tau]
        tp_ = sum(y for _p, y in flagged)
        prec = tp_ / len(flagged) if flagged else 0.0
        rec = tp_ / n_pos
        print(f"  {tau:>5.1f} {prec:>10.2f} {rec:>8.2f} {len(flagged):>9}")
