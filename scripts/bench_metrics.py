"""Metrics 1-10 for a trained artifact: calibration, confidence bands, MRR,
lookahead credit, churn, gold-sensitivity, basket size/gold error, count-head
accuracy, accuracy vs training-data volume. Writes aggregates to
data/ml/bench_metrics_<artifact>.json for charting.

Usage: python scripts/bench_metrics.py data/ml/prefix_model_C.pt
"""

from __future__ import annotations

import json
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

artifact = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "ml" / "prefix_model_C.pt"
blob = torch.load(artifact, map_location="cpu", weights_only=False)
cfg = blob.get("config") or {}
os.environ["PREFIX_GOLDEST"] = "1" if cfg.get("gold_est") else "0"
os.environ["PREFIX_RUNES"] = "1" if cfg.get("runes") else "0"
import train_prefix as tp  # noqa: E402

from app.ddragon import default_dragon  # noqa: E402
from app.shop_econ import GOLD_DRIFT, SAVE_ITEM, combine_cost  # noqa: E402

if cfg.get("runes"):
    while tp.QUERY_DIM > int(cfg["query_dim"]) and tp.KEYSTONE_IDS:
        tp.KEYSTONE_IDS.pop()
        tp.RUNE_DIM -= 1
        tp.QUERY_DIM -= 1
assert tp.QUERY_DIM == int(cfg.get("query_dim", tp.BASE_QUERY_DIM))

dragon = default_dragon()
test_path = ROOT / "data" / "ml" / "visits_test.jsonl"
train_path = ROOT / "data" / "ml" / "visits_train.jsonl"
labels = blob["label_ids"]
label_index = {i: idx for idx, i in enumerate(labels)}
KINDS = ("complete", "component", "start", "save")

n = sum(1 for _ in test_path.open(encoding="utf-8"))
slim: list[dict] = []
sample_rows: list[dict] = []  # every 30th full row, for the sensitivity probe


def gen():
    for k, row in enumerate(tp.fix_champ_ids(tp.with_history(tp.stream_rows(test_path)), blob["champ_index"])):
        slim.append(
            {
                "decision": row.get("decision"),
                "champion": row.get("champion") or "?",
                "label_ids": [int(i) for i in (row.get("label_ids") or []) if i],
                "label_id": int(row.get("label_id") or 0),
                "match_id": row.get("match_id"),
                "champion_id": row.get("champion_id"),
                "ts": row.get("ts") or 0,
            }
        )
        if k % 30 == 0:
            sample_rows.append(json.loads(json.dumps(row)))
        yield row


print(f"building test tensors ({n} rows)…", flush=True)
ds = tp.ShopDataset.from_stream(gen(), n, blob["champ_index"], blob["item_index"], label_index, dragon)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = tp.PrefixModel(
    len(blob["champ_index"]), len(blob["item_index"]), len(labels), len(blob["item_index"]) + 1,
    d_model=int(cfg.get("d_model", 64)), layers=int(cfg.get("layers", 2)),
    ff_dim=int(cfg.get("ff_dim", 128)), heads=int(cfg.get("heads", 4)), query_dim=tp.QUERY_DIM,
).to(device)
model.load_state_dict(blob["state_dict"], strict=False)
use_counts = bool(cfg.get("counts"))
threshold = float(cfg.get("basket_threshold", tp.BASKET_THRESHOLD))
ds.to_device(device)
model.eval()

N_BINS = 15
bin_count = torch.zeros(N_BINS, dtype=torch.float64, device=device)
bin_hit = torch.zeros(N_BINS, dtype=torch.float64, device=device)
bin_conf = torch.zeros(N_BINS, dtype=torch.float64, device=device)
top1_item: list[int] = []
top1_prob: list[float] = []
ranks: list[int] = []
baskets: list[list[int]] = []
dup_count_pred: Counter = Counter()  # (actual_count, predicted_count) on duplicated items
r = 0
with torch.no_grad():
    for batch in ds.batches(tp.BATCH, shuffle=False):
        out = model(
            batch["champs"], batch["sides"], batch["items"], batch["query"], batch["inv"], batch["hist"]
        )
        logits, count_logits = out["items"], out["counts"]
        if cfg.get("save_head"):
            logits = tp.splice_save(logits, out["save"], label_index.get(SAVE_ITEM))
        logits = logits.masked_fill(~batch["legal"], -1e4)
        probs = torch.sigmoid(logits)
        bought = batch["counts"] > 0
        # metric 1: reliability bins over every (visit, item) probability
        bins = (probs * N_BINS).long().clamp(max=N_BINS - 1)
        for b in range(N_BINS):
            m = bins == b
            bin_count[b] += m.sum()
            bin_hit[b] += (m & bought).sum()
            bin_conf[b] += probs[m].sum()
        p_cpu = probs.cpu()
        want = (count_logits.argmax(dim=2) + 1).cpu() if use_counts else None
        top = p_cpu.argmax(dim=1)
        for b in range(p_cpu.size(0)):
            row = slim[r]
            ti = int(top[b])
            top1_item.append(labels[ti])
            top1_prob.append(float(p_cpu[b, ti]))
            # metric 3: rank of the main label
            li = label_index.get(row["label_id"])
            ranks.append(int((p_cpu[b] > p_cpu[b, li]).sum()) + 1 if li is not None else 0)
            # metric 9: predicted copies where the player duplicated
            if want is not None:
                for item, c in Counter(row["label_ids"]).items():
                    if c >= 2 and item in label_index:
                        dup_count_pred[(min(c, 3), int(want[b, label_index[item]]))] += 1
            # basket decode (metrics 7, 8)
            budget = float(ds.budget[r]) + GOLD_DRIFT
            inv_c = Counter(ds.inventories[r])
            basket: list[int] = []
            for idx in torch.argsort(p_cpu[b], descending=True).tolist():
                if float(p_cpu[b, idx]) < threshold or len(basket) >= 5:
                    break
                item_id = labels[idx]
                if item_id == SAVE_ITEM:
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

out: dict = {"artifact": artifact.name, "test_rows": n, "threshold": threshold}

# 1 reliability + ECE
total = float(bin_count.sum())
rel = []
ece = 0.0
for b in range(N_BINS):
    c = float(bin_count[b])
    if c == 0:
        continue
    conf = float(bin_conf[b]) / c
    acc = float(bin_hit[b]) / c
    ece += (c / total) * abs(conf - acc)
    rel.append({"bin_lo": b / N_BINS, "confidence": round(conf, 4), "accuracy": round(acc, 4), "n": int(c)})
out["reliability"] = rel
out["ece"] = round(ece, 5)

# 2 confidence-stratified top-1 accuracy (hit = top pick was actually bought)
bands = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
band_stats = []
for lo, hi in bands:
    hits = tot = 0
    for row, item, p in zip(slim, top1_item, top1_prob):
        if lo <= p < hi:
            tot += 1
            hits += item in row["label_ids"]
    band_stats.append({"band": f"{lo:.2f}-{hi if hi <= 1 else 1.0:.2f}", "n": tot, "hit": round(hits / tot, 4) if tot else None})
out["confidence_bands"] = band_stats

# 3 MRR + rank histogram per decision
mrr: dict = {}
rank_hist: dict = {}
for kind in KINDS:
    rs = [rk for row, rk in zip(slim, ranks) if row["decision"] == kind and rk > 0]
    if not rs:
        continue
    mrr[kind] = round(sum(1.0 / rk for rk in rs) / len(rs), 4)
    hist = Counter(min(rk, 11) for rk in rs)
    rank_hist[kind] = {str(k) if k <= 10 else "11+": hist.get(k, 0) for k in range(1, 12)}
out["mrr"] = mrr
out["rank_hist"] = rank_hist

# 4 lookahead credit + 5 churn (need per-player sequences)
seq: dict[tuple, list[int]] = defaultdict(list)
for i, row in enumerate(slim):
    seq[(row["match_id"], row["champion_id"])].append(i)
look = {k: Counter() for k in ("complete", "component", "start")}
churn = Counter()
for key, idxs in seq.items():
    for j, i in enumerate(idxs):
        row = slim[i]
        pred = top1_item[i]
        kind = row["decision"]
        if kind in look and pred != row["label_id"]:
            look[kind]["miss"] += 1
            future = set()
            for nxt in idxs[j + 1 : j + 3]:
                future.update(slim[nxt]["label_ids"])
            look[kind]["hit_next2"] += pred in future
        if j + 1 < len(idxs):
            nxt = idxs[j + 1]
            churn["pairs"] += 1
            flip = top1_item[nxt] != pred
            churn["flips"] += flip
            if flip and pred not in row["label_ids"]:
                churn["unforced_flips"] += 1
            if pred not in row["label_ids"]:
                churn["unforced_pairs"] += 1
out["lookahead"] = {
    k: {"misses": v["miss"], "recovered_next2": v["hit_next2"], "rate": round(v["hit_next2"] / v["miss"], 4) if v["miss"] else None}
    for k, v in look.items()
}
out["churn"] = {
    "pairs": churn["pairs"],
    "flip_rate": round(churn["flips"] / max(churn["pairs"], 1), 4),
    "unforced_flip_rate": round(churn["unforced_flips"] / max(churn["unforced_pairs"], 1), 4),
}

# 6 gold sensitivity on the sample: does +/-150g flip the top-1?
flips = tot = 0
for delta in (150, -150):
    pert = []
    for row in sample_rows:
        q = dict(row)
        q["gold"] = max(0, (q.get("gold") or 0) + delta)
        pert.append(q)
    pds = tp.ShopDataset(pert, blob["champ_index"], blob["item_index"], label_index, dragon)
    pds.to_device(device)
    pi = 0
    with torch.no_grad():
        for batch in pds.batches(tp.BATCH, shuffle=False):
            o = model(batch["champs"], batch["sides"], batch["items"], batch["query"], batch["inv"], batch["hist"])
            lg = o["items"]
            if cfg.get("save_head"):
                lg = tp.splice_save(lg, o["save"], label_index.get(SAVE_ITEM))
            lg = lg.masked_fill(~batch["legal"], -1e4)
            tops = lg.argmax(dim=1).cpu()
            for t in tops.tolist():
                base_idx = sample_rows[pi]["_row_index"] if "_row_index" in sample_rows[pi] else pi * 30
                flips += labels[t] != top1_item[base_idx]
                tot += 1
                pi += 1
out["gold_sensitivity"] = {"perturbation": 150, "probes": tot, "flip_rate": round(flips / max(tot, 1), 4)}

# 7 basket size error and 8 gold gap
size_err: dict = {k: Counter() for k in ("complete", "component", "start")}
gold_gap: dict = {k: [] for k in ("complete", "component", "start")}


def item_gold(i: int) -> float:
    g = dragon.gold_block(i)
    return float(g.get("total") or g.get("base") or 0)


for row, basket in zip(slim, baskets):
    kind = row["decision"]
    if kind not in size_err:
        continue
    actual = [i for i in row["label_ids"] if i != SAVE_ITEM]
    if not actual:
        continue
    err = len(basket) - len(actual)
    size_err[kind][max(-3, min(3, err))] += 1
    gold_gap[kind].append(sum(item_gold(i) for i in basket) - sum(item_gold(i) for i in actual))
out["size_error"] = {k: {str(e): v[e] for e in sorted(v)} for k, v in size_err.items()}
gap_summary = {}
for k, gaps in gold_gap.items():
    gaps.sort()
    m = len(gaps)
    gap_summary[k] = {
        "median": round(gaps[m // 2], 0),
        "p25": round(gaps[m // 4], 0),
        "p75": round(gaps[3 * m // 4], 0),
        "mean": round(sum(gaps) / m, 0),
    }
out["gold_gap"] = gap_summary

# 9 count-head confusion on duplicated items
out["count_confusion"] = {f"actual{a}_pred{p}": c for (a, p), c in sorted(dup_count_pred.items())}

# 10 accuracy vs training volume per champion
print("counting train champions…", flush=True)
train_counts: Counter = Counter()
with train_path.open(encoding="utf-8") as fh:
    for line in fh:
        i = line.find('"champion": "')
        if i >= 0:
            j = line.find('"', i + 13)
            train_counts[line[i + 13 : j]] += 1
buckets = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 12000), (12000, 10**9)]
vol = []
for lo, hi in buckets:
    hits = tot = 0
    for row, item in zip(slim, top1_item):
        if lo <= train_counts.get(row["champion"], 0) < hi:
            tot += 1
            hits += item == row["label_id"]
    vol.append({"bucket": f"{lo}-{hi if hi < 10**8 else '+'}", "n": tot, "top1": round(hits / tot, 4) if tot else None})
out["volume_curve"] = vol

out_path = ROOT / "data" / "ml" / f"bench_metrics_{artifact.stem}.json"
out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1)[:2400])
print(f"\nwrote {out_path}")
