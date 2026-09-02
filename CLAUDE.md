# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Build an AI-powered recommender that, given a live game state at a shop visit (your champion, gold, inventory, all ten players' builds, score), suggests what a Challenger player would buy in that spot. "Optimal" deliberately means **imitation of high-elo behavior**, not counterfactual outcome optimization — the labels are what Challenger/GM/Master players actually bought. The live path exists: `app/live.py` polls the Riot Live Client Data API (`https://127.0.0.1:2999`, works in Practice Tool), `app/predictor.py` loads `prefix_model.pt` and reuses the training featurization (ShopDataset on a one-row list — never featurize live data any other way), and `/live` serves an auto-refreshing recommendation panel. Live gold is exact (better than the frame-stale training feature); other players' gold is unavailable live, those features zero out (measured neutral).

## Commands

Windows; no activation needed — call the venv interpreter directly:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload   # web UI at http://127.0.0.1:8000
.venv\Scripts\python.exe -m app.ingest                      # ingest Faker's recent ranked games
.venv\Scripts\python.exe -m app.ingest --ladder --region both --daily  # daily incremental ladder pull (Chall+GM, since local midnight)
.venv\Scripts\python.exe -m app.ingest --ladder --region kr           # full ladder pull for current patch
.venv\Scripts\python.exe -m app.ingest --pros               # LCK/LEC accounts (roster hardcoded in app/pros.py)
.venv\Scripts\python.exe -m app.ingest --backfill           # re-fetch timelines so stored games have all 10 players' shops
.venv\Scripts\python.exe scripts\daily_pull.py              # nightly pipeline: KR+EUW Chall/GM (26h window, uncapped) + pros → export → retrain
```

A Windows scheduled task ("BuildTracker Daily Pull", daily 04:30, runs `scripts/daily_pull.py` via pythonw) keeps everything current: pull (match lists uncapped — every ranked game in the window), then re-export JSONL, then retrain `train_prefix`, which saves `data/ml/prefix_model.pt` (weights + vocab indices + metrics). Logs to `data/daily_pull.log`; aborts on 401/403 (expired API key). `--no-train` pulls only; `--hours N` widens the window for catch-up runs.

ML pipeline (order matters — baseline.py exports the JSONL the trainers read):

```powershell
.venv\Scripts\python.exe scripts\baseline.py       # export data\ml\visits_{train,test}.jsonl + score champion-frequency baseline
.venv\Scripts\python.exe scripts\train.py          # RandomForest comparison
.venv\Scripts\python.exe scripts\train_prefix.py   # PyTorch board/basket model (the main experiment)
```

Tests are plain assert scripts, run directly (no pytest):

```powershell
.venv\Scripts\python.exe scripts\test_shop_econ.py
.venv\Scripts\python.exe scripts\test_visits.py
```

Requires `RIOT_API_KEY` in `.env` at repo root (dev keys expire every 24h; 401/403 aborts ingest).

## Architecture

Data flow: **Riot API → timeline reconstruction → SQLite → JSONL export → models.**

1. **`app/riot.py`** — httpx client with rate limiting (1.3s gap, 95 req/2min from `app/config.py`), retry on 429/5xx. 404 returns `None`, it does not raise.
2. **`app/reconstruct.py`** — the core. Replays a match-v5 timeline and reconstructs *shop visits*: item events by one player within a 10s idle window (`SHOP_IDLE_MS`) form one visit, capturing inventory before/after, bought/consumed, gold, level, CS, KDA, a snapshot of all 10 players' builds ("board"), and objective score. `reconstruct_game` produces the full-lobby view; `reconstruct_visits` extracts one player's perspective from it.
3. **`app/shop_econ.py`** — gold economics on top of Data Dragon recipes: infers gold-on-arrival from leftover + net spend (recipe base cost when components are consumed, else total cost, minus sell value), computes what an inventory can afford to complete, and tags each visit's decision kind (`complete`/`component`/`start`/`save`).
4. **`app/db.py`** — SQLite at `data/tracker.db` (multi-GB; WAL mode). Two parallel schemas: `matches` + `shop_visits` are *per-player perspectives* (PK `match_id, puuid`); `games` + `game_events` are the *full-lobby* reconstruction (one row per match, shop events as JSON payloads). `init_db()` performs in-place schema migrations — be careful editing it.
5. **`app/ingest.py`** — CLI dispatcher for all ingest modes (see commands above). Ladder ingest (`app/ladder.py`) dedups match IDs across players, filters to ranked solo queue + current patch, and only fetches timelines for matches with missing perspectives.
6. **`scripts/baseline.py`** — exports only matches where all 10 players have shop data (`HAVING COUNT(DISTINCT puuid) = 10`), splits 80/20 *by match* to avoid leakage, writes `data/ml/visits_{train,test}.jsonl` (hundreds of MB).
7. **`scripts/train_prefix.py`** — transformer over a 10-token board: token 0 is the shopper (own champion embedding + inventory items), then the 9 others (champion + side + item embeddings; the enemy laner with the shopper's role gets a distinct "lane opponent" side id). The query vector carries the shopper's state, `shop_econ` affordability features, enemy AP/AD share, and lane-opponent level diff. Env knobs: `PREFIX_EPOCHS`, `PREFIX_COSINE=1`, `PREFIX_EXTRAS=0` (zero game-state features), `PREFIX_HISTORY=1` (prior-purchase tokens — tested ≈ no gain, off by default), `PREFIX_OUT` (artifact filename). Precomputed tensors are cached as `data/ml/tensors_{split}_*.pt` keyed on export mtime + knobs. Output is both ranked top-3 and a budget-constrained variable-size basket (`predict_baskets`: greedy over sigmoid probs, recipe-aware costs, stops at confidence/budget). Multi-label BCE over the full shopping basket, with **legality masking**: items the player can't afford (inventory-aware recipe cost vs. arrival gold) are masked to -1e4 at train and predict time. Always report metrics against the champion-frequency baseline, overall and per decision kind — "complete" decisions are near-deterministic, so the number that matters is uplift on `start`/`component` decisions.

## Domain gotchas

- **Support item line** (Atlas 3865 → Compass 3866 → Bounty 3867 → finished 3869/3870/3871/3876/3877): match-v5 never emits ITEM_PURCHASED for it. It's granted at game start and upgrades via ITEM_DESTROYED; `reconstruct.py` seeds it into inventories and handles the upgrade chain specially.
- **Participant frames arrive every 60s and reconstruct.py stamps visits with the frame BEFORE the visit's events**, so every stored gold/level/cs snapshot is pre-visit and up to a minute stale. Consequences: `gold_left` in stored payloads is really *pre-visit* gold (verified empirically — players "afford" items above the stored gold), and any arrival-gold estimate that adds the visit's own spend to it leaks the label into the feature. The ML export therefore uses pre-visit gold directly as the budget feature, with `shop_econ.GOLD_DRIFT` (~60s income allowance) applied only to affordability masks. Never reintroduce `gold_left + net_spent` as a model input.
- **Combine costs must be recursive** (`shop_econ.combine_cost`): owning *some* of a recipe discounts the price like in game; the old base-vs-total binary overcharged partial-component completions.
- **Patch identity**: `patch_from_version("16.17.x.y") == "16.17"`; ladder ingest filters DTOs by patch and uses `PATCH_STARTS_UTC` in `app/config.py` as the match-v5 `startTime` — add an entry there when a new patch goes live.
- Item IDs 2055/772043 are control wards ("PINK") — the only consumables not skipped by classification; buying only wards counts as decision kind `save`.
- Train/test JSONL, `data/tracker.db`, logs, and the Data Dragon cache all live under `data/` — never commit it; regenerate JSONL via `scripts/baseline.py` after new ingests.
