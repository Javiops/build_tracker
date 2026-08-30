"""Train a small sequence model over earlier shops (the prefix) and score it like the forest."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR

ML_DIR = DATA_DIR / "ml"
MAX_PREFIX = 16
D_MODEL = 64
BATCH = 256
EPOCHS = 4
SEED = 16
ROLES = {"TOP": 1, "JUNGLE": 2, "MIDDLE": 3, "BOTTOM": 4, "UTILITY": 5}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    keep = (
        "champion",
        "champion_id",
        "role",
        "team_id",
        "gold",
        "level",
        "ts",
        "kills",
        "deaths",
        "inventory",
        "label_id",
    )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            prefix = []
            for step in (raw.get("prefix") or [])[-MAX_PREFIX:]:
                bought = step.get("bought") or []
                prefix.append(
                    {
                        "champion": step.get("champion") or "",
                        "champion_id": step.get("champion_id"),
                        "team_id": step.get("team_id"),
                        "bought": bought[:1],
                    }
                )
            row = {key: raw.get(key) for key in keep}
            row["prefix"] = prefix
            rows.append(row)
    return rows


def score_guesses(rows: list[dict], guesses: list[list[int]]) -> tuple[float, float]:
    hits1 = hits3 = 0
    for row, guess in zip(rows, guesses):
        if guess and row["label_id"] == guess[0]:
            hits1 += 1
        if row["label_id"] in guess:
            hits3 += 1
    n = len(rows) or 1
    return hits1 / n, hits3 / n


def baseline_guesses(train: list[dict], test: list[dict]) -> list[list[int]]:
    by_champ: dict[str, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    for row in train:
        by_champ[row["champion"]][row["label_id"]] += 1
        global_counts[row["label_id"]] += 1
    fallback = [item_id for item_id, _n in global_counts.most_common(3)]
    out = []
    for row in test:
        local = [item_id for item_id, _n in by_champ[row["champion"]].most_common(3)]
        extra = [item_id for item_id in fallback if item_id not in local]
        out.append((local + extra)[:3])
    return out


def _index_map(values: list[int]) -> dict[int, int]:
    unique = sorted({int(v) for v in values if v})
    return {item_id: i + 1 for i, item_id in enumerate(unique)}


class ShopDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        champ_index: dict[int, int],
        item_index: dict[int, int],
        label_index: dict[int, int],
    ):
        self.rows = rows
        self.champ_index = champ_index
        self.item_index = item_index
        self.label_index = label_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        row = self.rows[i]
        team = row.get("team_id")
        champs = torch.zeros(MAX_PREFIX, dtype=torch.long)
        sides = torch.zeros(MAX_PREFIX, dtype=torch.long)
        items = torch.zeros(MAX_PREFIX, dtype=torch.long)
        prefix = (row.get("prefix") or [])[-MAX_PREFIX:]
        for pos, step in enumerate(prefix):
            champs[pos] = self.champ_index.get(int(step.get("champion_id") or 0), 0)
            # prefix rows from export may only have champion name; map later if missing
            if champs[pos] == 0:
                champs[pos] = self.champ_index.get(hash(step.get("champion") or "") % 100000, 0)
            sides[pos] = 1 if step.get("team_id") == team else 2
            bought = step.get("bought") or []
            items[pos] = self.item_index.get(int(bought[0]), 0) if bought else 0
        query = torch.tensor(
            [
                float(self.champ_index.get(int(row.get("champion_id") or 0), 0)),
                float(ROLES.get(row.get("role") or "", 0)),
                float(row.get("gold") or 0) / 3000.0,
                float(row.get("level") or 0) / 18.0,
                float((row.get("ts") or 0) / 60000.0),
                float(row.get("kills") or 0) / 10.0,
                float(row.get("deaths") or 0) / 10.0,
                float(len(prefix)) / MAX_PREFIX,
                float(len(row.get("inventory") or [])) / 6.0,
            ],
            dtype=torch.float32,
        )
        inv = torch.zeros(len(self.item_index) + 1, dtype=torch.float32)
        for item_id in row.get("inventory") or []:
            idx = self.item_index.get(int(item_id))
            if idx:
                inv[idx] = 1.0
        label = self.label_index.get(int(row["label_id"]), 0)
        return {
            "champs": champs,
            "sides": sides,
            "items": items,
            "query": query,
            "inv": inv,
            "label": torch.tensor(label, dtype=torch.long),
        }


class PrefixModel(nn.Module):
    def __init__(self, n_champ: int, n_item: int, n_label: int, inv_dim: int):
        super().__init__()
        self.champ_emb = nn.Embedding(n_champ + 1, D_MODEL, padding_idx=0)
        self.item_emb = nn.Embedding(n_item + 1, D_MODEL, padding_idx=0)
        self.side_emb = nn.Embedding(3, D_MODEL, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.query_proj = nn.Sequential(
            nn.Linear(9 + inv_dim, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.head = nn.Linear(D_MODEL, n_label)

    def forward(self, champs, sides, items, query, inv):
        prefix = self.champ_emb(champs) + self.side_emb(sides) + self.item_emb(items)
        q = self.query_proj(torch.cat([query, inv], dim=1)).unsqueeze(1)
        seq = torch.cat([q, prefix], dim=1)
        pad = champs == 0
        key_pad = torch.cat([torch.zeros(champs.size(0), 1, dtype=torch.bool, device=champs.device), pad], dim=1)
        encoded = self.encoder(seq, src_key_padding_mask=key_pad)
        return self.head(encoded[:, 0])


def predict_top3(model, loader, device, labels: list[int]) -> list[list[int]]:
    model.eval()
    guesses: list[list[int]] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["champs"].to(device),
                batch["sides"].to(device),
                batch["items"].to(device),
                batch["query"].to(device),
                batch["inv"].to(device),
            )
            top = logits.topk(k=min(3, logits.size(1)), dim=1).indices.cpu().tolist()
            for idxs in top:
                guesses.append([labels[i] for i in idxs])
    return guesses


def per_champ(test: list[dict], guesses: list[list[int]], limit: int = 8) -> None:
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row, guess in zip(test, guesses):
        champ = row["champion"] or "?"
        per[champ][2] += 1
        if guess and row["label_id"] == guess[0]:
            per[champ][0] += 1
        if row["label_id"] in guess:
            per[champ][1] += 1
    ranked = sorted(per.items(), key=lambda kv: -kv[1][2])[:limit]
    for champ, (c1, c3, count) in ranked:
        print(f"  {champ:<14} top-1 {c1 / count:.2f}  top-3 {c3 / count:.2f}   n={count}")


def main() -> None:
    train_path = ML_DIR / "visits_train.jsonl"
    test_path = ML_DIR / "visits_test.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit("Run scripts/baseline.py first so the jsonl files exist.")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("Loading rows…", flush=True)
    train = load_jsonl(train_path)
    print(f"loaded train {len(train)}", flush=True)
    test = load_jsonl(test_path)
    print(f"train {len(train)} shops  test {len(test)} shops", flush=True)

    base = baseline_guesses(train, test)
    b1, b3 = score_guesses(test, base)
    print(flush=True)
    print("baseline  (champion frequency only)", flush=True)
    print(f"  top-1  {b1:.2f}", flush=True)
    print(f"  top-3  {b3:.2f}", flush=True)

    champ_ids = [int(row.get("champion_id") or 0) for row in train]
    item_ids = [int(row["label_id"]) for row in train]
    for row in train:
        item_ids.extend(int(i) for i in (row.get("inventory") or []))
        for step in row.get("prefix") or []:
            item_ids.extend(int(i) for i in (step.get("bought") or []))
    champ_index = _index_map(champ_ids)
    item_index = _index_map(item_ids)
    label_ids = sorted({int(row["label_id"]) for row in train})
    label_index = {item_id: i for i, item_id in enumerate(label_ids)}

    # Prefix export has champion names, not ids. Attach a stable stand-in from the name.
    for row in train + test:
        for step in row.get("prefix") or []:
            if not step.get("champion_id"):
                step["champion_id"] = abs(hash(step.get("champion") or "")) % 10000 + 1
                if step["champion_id"] not in champ_index:
                    champ_index[step["champion_id"]] = len(champ_index) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(flush=True)
    print(f"Sequence model  prefix<= {MAX_PREFIX}  device={device}", flush=True)
    train_ds = ShopDataset(train, champ_index, item_index, label_index)
    test_ds = ShopDataset(test, champ_index, item_index, label_index)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

    model = PrefixModel(len(champ_index), len(item_index), len(label_ids), len(item_index) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            opt.zero_grad()
            logits = model(
                batch["champs"].to(device),
                batch["sides"].to(device),
                batch["items"].to(device),
                batch["query"].to(device),
                batch["inv"].to(device),
            )
            loss = loss_fn(logits, batch["label"].to(device))
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(batch["label"])
            n += len(batch["label"])
        print(f"  epoch {epoch}/{EPOCHS}  loss {total / max(n, 1):.3f}", flush=True)
    print(f"trained in {time.monotonic() - started:.1f}s", flush=True)

    guesses = predict_top3(model, test_loader, device, label_ids)
    m1, m3 = score_guesses(test, guesses)
    print(flush=True)
    print("prefix model  (ordered earlier shops + you are in base)", flush=True)
    print(f"  top-1  {m1:.2f}   vs baseline {b1:.2f}", flush=True)
    print(f"  top-3  {m3:.2f}   vs baseline {b3:.2f}", flush=True)
    print(flush=True)
    per_champ(test, guesses)


if __name__ == "__main__":
    main()
