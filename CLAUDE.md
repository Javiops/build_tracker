# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Read `HANDOFF.md` first.** This file carries the durable implementation
> contract and was synchronised after the 2026-09-09 audit; `HANDOFF.md` remains
> authoritative for active jobs, corpus status, gold semantics, split version,
> and the current recovery queue. Quantified exact-gold gains remain withdrawn.

## Operating contract (binding on every session in this repo)

The long form is `docs/astra_prompt.md` §0, §7, §11, §12. It was written for an
external model; it applies to Claude Code sessions here identically. The short
form:

**The failure mode in this project is not breakage, it is numbers that look
right.** No crash, no traceback, no red test — every serious failure here
produced plausible output that a competent person acted on (leaky gold looked
like a breakthrough; a contaminated split looked like a good score; a RAM
thrash looked like a slow machine). Behave accordingly:

1. **Lead. Do not wait to be steered.** Arrive with the next move already
   chosen and the reasoning attached — not a menu of options for him to sort
   through. Make the routine calls myself and say what I decided. Escalate only
   what is genuinely his: money, priorities between workstreams, anything
   irreversible or outward-facing. End on a move, not a question. When work
   finishes, propose what follows rather than reporting done and stopping.
2. **No false hope.** Unmeasured is "not measured". A number without
   `<metric>, <split>, <population n>, <report path>` is **inadmissible** —
   say so rather than repeat it. Never compare a training-log number to a
   harness number; that already cost this project a retraction.
3. **Object once, on substance.** If a request costs real time, money or
   correctness, say so in the first two sentences with the cost and the
   alternative — then, reaffirmed or not obviously wrong, get on with it. This
   is for consequential calls, not for style, naming, or things I would merely
   have done differently. Friction where it does not buy anything is a defect.
4. **Name what to cut when the plan is large enough to have fat**, and state
   the strongest counterargument to my own recommendation before he has to.
   Not required on a one-step task.
5. **Verify in code before asserting.** Documents here go stale; the audit
   findings were all confirmed in code within minutes. Read the file.
6. **Lead with the finding or decision.** No praise openings, no restating his
   message. Short and direct; Spanish if he writes in Spanish.
7. **Report negative results loudly and early.** Several of the most valuable
   results in this project were negative. A negative result on day one beats a
   positive one on day five.
8. **Be hostile to my own output.** My code is the most likely source of the
   next silent wrong number.

Response shape for planning, evaluation and promotion work: **FINDING →
EVIDENCE (with provenance) → WHAT WOULD CHANGE MY MIND → COST → NOT DOING THIS
CYCLE → RISK → NEXT.** For ordinary implementation, just the finding and the
next move. Skip what does not apply; do not pad. The ceremony exists to stop
bad numbers shipping, not to decorate a two-line change.

Hard prohibitions: no test-set reads outside one final confirmatory check; no
promotion on training or canonical metrics; no unpaired model comparisons; no
shop model on a poll; no invented labels for what Match-V5 cannot observe; no
weighted-BCE output shown as a probability; no calling the product "optimal
builds"; no long jobs started inside an agent session; no racing or stopping a
running Riot sequence without asking; no "simplifying" the support-item chain,
recursive combine cost, or `init_db()` migrations; no unrequested shutdown.

## Project goal

Build an AI-powered **behavioural-imitation** advisor: at an explicit shop
decision, show what high-elo players historically bought in similar reconstructed
states. It does not optimise outcomes or establish that a purchase is best.

**Serving contract (2026-09-09):** `GET /api/live` only shapes live state and
observes an active decision session. It must never run the shop-only model.
Only `POST /api/live/session`, initiated by the player while choosing in shop,
may score one exact-wallet snapshot. The 90-second session records local-only,
anonymous telemetry: displayed actions and a later inventory addition or an
explicit no-buy timeout; cancellation and removal-only changes are unscored.
Historic `SAVE_ITEM` labels are post-death no-buys, not every manual recall.
Weighted-BCE outputs are ranking scores, never calibrated confidence shown to
the player. `prefix_model.pt` may serve only when its exact digest is bound to
a provenance-checked validation policy report in
`data/ml/deployment_manifest.json`; legacy artifacts are forensic-only.

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

A Windows scheduled task ("BuildTracker Daily Pull", daily 04:30, runs
`scripts/daily_pull.py` via pythonw) pulls uncapped daily match lists and may
train a **candidate**, never auto-promotes or overwrites the served artifact.
Logs to `data/daily_pull.log`; 401/403 aborts ingest. Current recovery work is
the causal re-reconstruction: `app.ingest --backfill --refresh` must finish
before creating a new export. Do not race it or stop its rate-limit sequence.

ML pipeline (order matters — baseline.py exports the JSONL the trainers read):

```powershell
.venv\Scripts\python.exe scripts\baseline.py --rebuild-split # first causal export: frozen 70/15/15 temporal train/val/test
.venv\Scripts\python.exe scripts\train.py          # RandomForest comparison
.venv\Scripts\python.exe scripts\train_prefix.py   # writes a provenance-bound candidate; select on validation only
.venv\Scripts\python.exe scripts\eval_policy.py --artifact <candidate> --split val
.venv\Scripts\python.exe scripts\promote_served_model.py --report <validation-report>
```

Tests are plain assert scripts, run directly (no pytest):

```powershell
.venv\Scripts\python.exe scripts\test_shop_econ.py
.venv\Scripts\python.exe scripts\test_visits.py
```

Requires `RIOT_API_KEY` in `.env` at repo root (dev keys expire every 24h; 401/403 aborts ingest).

## Architecture

Data flow: **Riot API → timeline reconstruction → SQLite → JSONL export → models.**

1. **`app/riot.py`** — httpx client with rate limiting (1.3s gap, 95 req/2min from `app/config.py`), retry on 429/5xx. 404 returns `None`, it does not raise. Successful Match-V5 match/timeline responses are atomically gzip-archived in ignored `data/raw_match_v5/`; replay from that cache rather than starting a duplicate raw pull. Set `RAW_MATCH_V5_CACHE=0` only for an explicitly ephemeral run.
2. **`app/reconstruct.py`** — the core. Replays a match-v5 timeline and reconstructs *shop visits*: item events by one player within a 10s idle window (`SHOP_IDLE_MS`) form one visit, capturing inventory before/after, bought/consumed, gold, level, CS, KDA, a snapshot of all 10 players' builds ("board"), and objective score. `reconstruct_game` produces the full-lobby view; `reconstruct_visits` extracts one player's perspective from it.
3. **`app/shop_econ.py`** — recursive recipe-aware affordability and decision
   kinds. `gold_left + visit spend` is label-derived and banned. Offline
   `gold_est` is the causal, versioned `prequential-v2` approximation; live
   `currentGold` is exact and tagged `live-exact-v2`.
4. **`app/db.py`** — SQLite at `data/tracker.db` (multi-GB; WAL mode). Two parallel schemas: `matches` + `shop_visits` are *per-player perspectives* (PK `match_id, puuid`); `games` + `game_events` are the *full-lobby* reconstruction (one row per match, shop events as JSON payloads). `init_db()` performs in-place schema migrations — be careful editing it.
5. **`app/ingest.py`** — CLI dispatcher for all ingest modes (see commands above). Ladder ingest (`app/ladder.py`) dedups match IDs across players, filters to ranked solo queue + current patch, and only fetches timelines for matches with missing perspectives.
6. **`scripts/baseline.py`** — exports only complete ten-player matches into a
   frozen, whole-game, temporal 70/15/15 train/validation/test split. The
   manifest is versioned; later games are `post_cutoff`, never silently train.
7. **`scripts/train_prefix.py`** — 10-token board transformer using
   `ShopDataset` for both train and live featurisation. It refuses GOLDX rows
   unless every offline estimate is `prequential-v2`, records provenance, and
   scores validation only. `scripts/eval_policy.py` evaluates the actual
   `predict_options` policy with game bootstrap; canonical top-k is diagnostic.

## Domain gotchas

- **Support item line** (Atlas 3865 → Compass 3866 → Bounty 3867 → finished 3869/3870/3871/3876/3877): match-v5 never emits ITEM_PURCHASED for it. It's granted at game start and upgrades via ITEM_DESTROYED; `reconstruct.py` seeds it into inventories and handles the upgrade chain specially.
- **Participant frames arrive every 60s.** Pre-visit `gold_left` is legal but
  stale. Never use `gold_left + visit net_spend` (label-derived), a later frame,
  or post-decision events. GOLDX accepts only `prequential-v2` offline rows;
  live rows use the exact client wallet.
- **Combine costs must be recursive** (`shop_econ.combine_cost`): owning *some* of a recipe discounts the price like in game; the old base-vs-total binary overcharged partial-component completions.
- **Patch identity**: `patch_from_version("16.17.x.y") == "16.17"`; ladder ingest filters DTOs by patch and uses `PATCH_STARTS_UTC` in `app/config.py` as the match-v5 `startTime` — add an entry there when a new patch goes live.
- Item IDs 2055/772043 are control wards ("PINK"). Ward-only is distinct from
  `no_buy_death`; only the latter is a historic save label.
- Never promote a model from training/canonical metrics. It needs a matching
  frozen-validation displayed-policy report with zero unaffordable baskets,
  then `scripts/promote_served_model.py` binds its digest for live serving.
- Train/test JSONL, `data/tracker.db`, logs, and the Data Dragon cache all live under `data/` — never commit it; regenerate JSONL via `scripts/baseline.py` after new ingests.
