# CLAUDE.md

Guidance for AI assistants in this repository. `AGENTS.md` points here — Claude
and Codex get identical instructions. Keep it that way.

## What this is

A behavioural-imitation shop advisor for League of Legends: at an explicit shop
decision, show what high-elo players historically bought in similar
reconstructed states. It does **not** optimise outcomes and does not establish
that a purchase is best. Never call it "optimal builds".

The failure mode here is not breakage — it is numbers that look right. Leaky
gold looked like a breakthrough; a contaminated split looked like a good score;
a RAM thrash looked like a slow machine. No crash, no traceback, no red test.

## Three rules that override your defaults

1. **Unmeasured is "not measured."** A number without `<metric, split, n,
   report path>` is inadmissible — say so rather than repeat it.
2. **Never compare a training-log number to a harness number.** That already
   cost this project a public retraction.
3. **Verify in code before asserting.** Docs here go stale. Read the file.

Everything else about tone and format — leading with the finding, skipping
preamble, reporting negative results early — you already do by default. Don't
perform it.

## Hard prohibitions

- No test-set reads outside one final confirmatory check.
- No promotion on training or canonical metrics. A candidate needs a matching
  frozen-validation displayed-policy report with zero unaffordable baskets,
  then `scripts/promote_served_model.py` binds its digest.
- No unpaired model comparisons, and never compare across different exports.
- No shop model on a poll. `GET /api/live` shapes live state only; only
  `POST /api/live/session`, started by the player in shop, may score one
  exact-wallet snapshot.
- No weighted-BCE output shown to the player as a probability. Ranking scores
  are not calibrated confidence.
- No invented labels for what Match-V5 cannot observe. `SAVE_ITEM` is a
  post-death no-buy, not every manual recall.
- No long jobs started inside an agent session — they are killed at ~3 min.
- No racing or stopping a running Riot sequence without asking.
- No "simplifying" the support-item chain, recursive combine cost, or
  `init_db()` migrations.
- No unrequested shutdown. Abort a pending one with `shutdown /a`.
- Never commit anything under `data/`.

## Commands

Windows; call the venv interpreter directly, no activation needed.

| Task               | Command                                                     |
| ------------------ | ----------------------------------------------------------- |
| Run tests          | `.venv\Scripts\python.exe scripts\run_tests.py`              |
| Current state      | `.venv\Scripts\python.exe scripts\status.py`                 |
| Web UI             | `.venv\Scripts\python.exe -m uvicorn app.main:app --reload`  |
| Ingest ladder      | `... -m app.ingest --ladder --region both --daily`           |
| Full patch pull    | `... -m app.ingest --ladder --region kr`                     |
| Backfill timelines | `... -m app.ingest --backfill`                               |
| Export + split     | `... scripts\baseline.py --rebuild-split`                    |
| Train candidate    | `... scripts\train_prefix.py`                                |
| Evaluate           | `... scripts\eval_policy.py --artifact <c> --split val`      |
| Promote            | `... scripts\promote_served_model.py --report <r>`           |

Order matters: `baseline.py` exports the JSONL the trainers read. Requires
`RIOT_API_KEY` in `.env`; dev keys expire every 24h and 401/403 aborts ingest.

**Run `scripts\status.py` before starting anything.** It reports running
scheduled tasks, active log tails, the served artifact digest, corpus
generations and `git status` — the things that used to live in `HANDOFF.md` and
went stale between sessions.

A Windows scheduled task ("BuildTracker Daily Pull", 04:30) runs
`scripts/daily_pull.py`; it may train a **candidate** but never auto-promotes.

## Architecture

Riot API → timeline reconstruction → SQLite → JSONL export → models.

| File                      | Role                                                                                                                                                     |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/riot.py`             | Rate-limited httpx client (1.3s gap, 95 req/2min). 404 returns `None`, does not raise. Responses gzip-archived to `data/raw_match_v5/`; replay from cache. |
| `app/reconstruct.py`      | **The core.** Replays a timeline into *shop visits* — item events within a 10s idle window — capturing inventory before/after, gold, level, CS, KDA and a 10-player board snapshot. |
| `app/shop_econ.py`        | Recursive recipe-aware affordability, slot legality, decision kinds. Offline gold is `prequential-v2`; live is `live-exact-v2`.                            |
| `app/db.py`               | SQLite at `data/tracker.db` (multi-GB, WAL). Two schemas: `matches`+`shop_visits` per-player, `games`+`game_events` full-lobby. `init_db()` migrates in place. |
| `app/ingest.py`           | CLI dispatcher for all ingest modes. `app/ladder.py` dedups match IDs and filters to ranked solo + current patch.                                          |
| `scripts/baseline.py`     | Exports complete ten-player matches into a frozen temporal 70/15/15 split. Later games are `post_cutoff` and never silently train.                         |
| `scripts/train_prefix.py` | 10-token board transformer. Refuses non-`prequential-v2` GOLDX rows, records provenance, scores validation only.                                          |
| `scripts/eval_policy.py`  | Evaluates the real `predict_options` policy with game bootstrap. Canonical top-k is diagnostic only.                                                      |

## Domain gotchas

- **Support item line** (3865 → 3866 → 3867 → 3869/70/71/76/77): match-v5 never
  emits ITEM_PURCHASED for it. It is granted at game start and upgrades via
  ITEM_DESTROYED; `reconstruct.py` seeds and chains it specially.
- **Participant frames arrive every 60s.** Pre-visit `gold_left` is legal but
  stale. `gold_left + net_spend` is label-derived and banned, as is any later
  frame or post-decision event.
- **Combine costs are recursive** (`shop_econ.combine_cost`): owning part of a
  recipe discounts the price, as in game.
- **Patch identity**: `patch_from_version("16.17.x.y") == "16.17"`. Add an entry
  to `PATCH_STARTS_UTC` in `app/config.py` on patch day, or ingest warns and
  falls back to a 16-day window.
- Items 2055 / 772043 are control wards. Ward-only is *not* `no_buy_death`;
  only the latter is a historic save label.
- Shop targets are **multisets** — `label_counts`, `target_encoding=item-counts-v1`.
  Preserve counts; reject capacity overflow rather than truncating.
- Train/test JSONL, `data/tracker.db`, logs and the Data Dragon cache all live
  under `data/`. Regenerate JSONL via `scripts/baseline.py` after new ingests.

## Where the rest lives

- **`docs/history.md`** — the ledger of withdrawn beliefs and the traps that
  cost real hours. Read it before proposing anything that sounds like a
  breakthrough; most of them have already been tried and disproved.
- **`HANDOFF.md`** — live job state, corpus status and the current queue. Detail
  only; `scripts/status.py` is the authority on what is actually running.
- **`.claude/skills/`** — procedures that load when relevant: `patch-day`,
  `run-experiment`, `promote-model`, `incident`.
- **`docs/archive/`** — superseded documents kept for reference, not binding.
