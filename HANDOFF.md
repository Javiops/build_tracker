# HANDOFF — Build Tracker (live League shop-decision advisor)

**Active update, 2026-09-11:** the completed 16.18 sample is frozen at
`data/corpora/16.18_initial/` (1,311 games; collection, not training acceptance).
The owner authorized seven ordered gates: freeze → semantic audit → coverage →
atomic export generations → isolated pipeline smoke → another 1–2 nights → one
baseline/candidate validation comparison. Gates 1–3 are complete for the qualified
subset, not the whole collection. The derived corpus is
`data/corpora/16.18_repaired_v2/`. Its successful semantic recheck is in
`data/acceptance/16.18_repaired_v2/recheck/`. The first derived audit found one
validator error (last quest marker instead of first completion); its failed
report is preserved in the parent directory. The authoritative final qualification
is now `data/acceptance/16.18_repaired_v2/inventory_recheck/qualification.json`:
1,063 accepted games, 248 quarantined; before-board, before-shop and after-shop
inventory checks pass for the accepted set. Coverage in its sibling `coverage/`
contains 1,049 export-eligible games and 126,325 rows (735/157/157 games in
train/val/test). Exclusions and pro coverage remain explicit.

Gates 4–5 completed at 08:51 UTC / 10:51 Madrid. The immutable export is
`data/dataset_generations/16.18-initial-smoke-v2/`, fingerprint
`5f24376f311742a8ffdec6633a0d7cc1d184875d3d251eaa8a491f729693447f`.
`data/experiments/16.18-initial-smoke-v2/completion.json` binds the one-epoch
trunk, one-epoch frozen-trunk graft and displayed-policy smoke: 1,252 visits
across ten validation games, zero unaffordable baskets. Candidate SHA256:
`202b5800e489199c93bb37d1db6ad01b324856b99641944c13bdcc3ca4721e9a`.
Serving/hold artifacts stayed unchanged; no final-test evaluation or promotion.
This is mechanics testing, not a quality comparison. Keep its code snapshots and
results immutable rather than reusing this experiment directory.
The existing daily collection finished at 10:11 Madrid with zero stage failures;
its additional rows used older code and require a new frozen/audited generation.
The production pipeline hold remains in force.
See `docs/data_acceptance_16_18.md`; the earlier broad readiness assessment is
withdrawn. Source collection completeness did not establish semantic correctness.

The original semantic report found inhibitors counted as towers. Subsequent
review found labels inferred from unpurchased inventory gains, final-DTO support
choice leakage, incomplete role-quest/ward handling, and incomplete combine-undo
restoration. The uncommitted `decision-start-v2` reconstruction uses causal support
tier proxies, timestamped role-quest transitions, source-reported undo gold,
strictly earlier gold frames, and purchase-backed labels. Tests cover those
properties; the full-corpus recheck has zero hard failures. Opaque undo IDs and
unreconciled terminal inventories must be explicitly excluded from any accepted
training subset. Do not equate an accepted subset with a clean entire collection.

The owner confirmed the inventory extremes after raw inspection: eight modeled
copies can mean six regular items plus two support wards; the five-copy target
is five source-recorded Long Swords (EUW1_7979347871, participant 1, 267579ms).
`coverage_extremes.json` preserves the traces. **Shop targets are multisets:**
qualified JSONL uses `label_counts` (item ID → positive integer quantity),
`target_encoding=item-counts-v1`, without an ordered `label_ids` target. Tensor
adapters preserve counts, canonicalize inventory order for new artifacts, and
reject capacity overflow rather than silently truncate/clip. Legacy canonical
labels remain diagnostic only. The immutable displayed-policy smoke used
conservative regular-slot decoding; it remains mechanics-only evidence.

**Slot-legality follow-up, 2026-09-11:** both Predictor paths and the conditional
baseline now share recipe, wallet, regular-slot, ward-stack and boot-family
checks in `app/shop_econ.py`. Basket entries are retained quantities: a later
combine cannot consume an earlier displayed purchase. Conditional baskets are
unordered and their legality searches for a valid purchase sequence. Policy
evaluation independently replays each displayed option and checks reported
prices, capacity and wallet. New reconstruction/export rows carry
`slot_state_version=role-slots-v1` and strictly pre-decision
`bot_quest_complete`; skipped held consumables also retain their regular-slot
occupancy. Existing immutable generations were not rewritten and lack this
new slot state: use a freshly reconstructed/audited generation for gate 7.

Live Client docs and existing captures do not establish a bot-quest completion
or separate role-inventory signal. Live completion stays unknown, and absent
BOTTOM boots / UTILITY wards are marked as unknown ownership; affected buys
are withheld rather than assuming empty slots. Observed skipped regular-slot
items count against capacity. This is a conservative live limitation, not a
claim of complete live inventory observability. See the new
`test_purchase_legality.py`, `test_slot_state_causal.py` and
`test_live_slot_state.py` regression scripts. Production hold, served artifacts,
frozen evidence and final-test payloads remain untouched.
Validation: 18 focused regression scripts passed (purchase legality, slot-state
causality/live parsing, multiset, economics, visits/quests, gold/decision causality,
audit, conditional policy, sessions/API, provenance, dataset generations, corpus
oracle, hold and live prediction). The unchanged initial smoke candidate also
passed the updated live path on four stored snapshots; its digest still matches
the completion report above. This is regression evidence, not a new quality
comparison or a rerun of the immutable experiment.

The owner's separate request for larger Codex context was applied to
`C:/Users/Javier/.codex/config.toml`: model_context_window=1050000,
model_auto_compact_token_limit=900000, with a backup. Active-session adoption is
not verified; do not claim this restored any already-compacted history.

Gate 6 follow-up is scheduled in this task at 07:00 Madrid daily, automation ID
`16-18-next-two-nights`, for the September 12 and 13 collections, then the
conditional gate 7 comparison. It must stop after those are handled. The dated
samples do not exist yet; no future collection or comparison is claimed done.
See the tested next-night collection/freezing procedure in
`docs/data_acceptance_16_18.md`. The collector can inherit the original reviewed
cohorts, retains unresolved-pro outcomes, and reconstructs only once per game.

The owner accepted the 16 unresolved tracked pro accounts as a minor limitation;
do not spend another cycle resolving them. The previous shutdown completed;
there is no new shutdown authorization for this work. No final-test evaluation,
model promotion, or commit has been authorized by the seven-step request.

Written 2026-09-09, revised the same day after an external audit. `CLAUDE.md` /
`AGENTS.md` carry the durable contract; this file covers mutable corpus state,
the recovery queue, and the long-form reasoning behind them.

**This file is not the binding contract and is not read first.** Start at
`CLAUDE.md` — it is short, and it is what an assistant loads every session.
Come here for depth on a specific question, and run `scripts/status.py` for
anything about what is happening right now.

**If you read only one thing, read §5.** An earlier version of this handoff
claimed the project had gained "+14 points in actual games" from exact-gold
training. That claim was withdrawn: it compared an eval harness against a
training log, and the estimator behind it was reading a frame from *after* the
decision. §5 explains what is actually known.

Read order: §1 (what the product is) → `scripts/status.py` (what is running,
so you don't collide with it) → §5 (gold) → §6 (the audit and what it changed)
→ `docs/history.md` (traps) → §11 (the queue).

**The development history lives in two places now.** §2.1 keeps the timeline;
the ledger of withdrawn beliefs, the engineering failures and the traps moved
verbatim to `docs/history.md` so they survive independently of this file's
live-status sections. When something here turns out to be wrong, move it into
that ledger instead of quietly deleting it.

---

## 1. What this product is

A desktop companion for League of Legends that answers only when the player
explicitly starts a shop-decision session: **"what would a KR Challenger buy
right here?"** — including *"nothing, hold your gold"* as a first-class
answer. The Live Client API has no trustworthy shop/location flag, so it must
not be inferred from continuous polling.

**The engraved question (owner's framing, 2026-09-02, never drift from it):**

> given that a KR Challenger — a really good player — visited the shop, WHICH
> INCLUDES basing for health/resources and buying nothing, what is the optimal
> action?

Consequences that are not negotiable:

- **Save/no-buy is a real action**, not a UI heuristic. `reconstruct.py` emits
  save events (respawned at fountain, ≥400g, bought nothing within 90s) with
  the `SAVE_ITEM = 999999` sentinel label.
- **"Optimal" means imitation of high-elo behaviour**, not outcome
  optimisation. Labels are what Challenger/GM players actually bought; there
  are no counterfactuals and no signal that any purchase was good. Calling the
  output "optimal" outside this narrow sense is rhetoric — the audit was right
  about that, and product copy should say "what a Challenger did here".
- **Confidence must be trustworthy.** Rare-item probability inflation "boosts
  great matchup choices but also highlights mistakes" (owner). Hence the
  `pos_weight` cap and the unweighted save head.
- **Poll-to-poll stability matters as much as accuracy** — but see §8: the
  previously quoted "2.9% flicker with identical state" was a mismeasurement.

Data flow: **Riot API → timeline reconstruction → SQLite → JSONL export →
transformer → live overlay.** Training data is offline match timelines;
inference reads the League client's own local API. No memory reading, no
scraping.

---

## 2. Development history

This section is deliberately long, and it records the mistakes as carefully as
the wins. Almost everything expensive this project learned, it learned by
believing a number that turned out to be wrong — and a handoff that documents
only the current state hands the next person the conclusions without the
reasoning that earned them. Keep appending to it. When a belief here is
overturned, move it to the withdrawn-belief ledger in `docs/history.md`
 rather than deleting it.

Commits exist through 2026-09-06; **everything after that is uncommitted** (`git status`).

### 2.1 Timeline

| When | What happened |
|---|---|
| **08-30** | Initial commit: Riot API client, SQLite, a web UI over Faker's ranked games. A match browser, not yet an advisor. |
| **09-02** | The architecture day, and the day the product became itself. Shop-economics layer (`shop_econ.py`: recursive combine costs, affordability, decision kinds). Board-model overhaul (self token, lane opponent, game-state features). Nightly pipeline. **First gold label leak found and fixed** — the budget feature was being reconstructed from the visit's own spend. Live recommender born (`live.py`, `predictor.py`, `/live`). Iterative basket decode with in-game purchase blocks. Plan probe and plan-consistency (stop burning gold off-build). **Save taught as a first-class action.** `pos_weight 8` promoted for trustworthy confidence; self-tuning basket threshold. The engraved question was stated and has not moved since. |
| **09-03** | 37× faster mask preprocessing (memoised Data Dragon, hoisted per-label metadata). Trainer opts itself out of Windows background throttling. README so friends could run it. |
| **09-04** | Experiments **A** (bigger arch → 0.61) and **B** (A + runes + an arrival-gold estimate → 0.63). **RAM crisis**: the export outgrew 15.8 GB as Python dicts and hung the machine twice; the trainer became streaming-only. |
| **09-05** | **C** (multiset count head — each buy is a multiset, ~7% of visits repeat an item). **D** (`pos_weight=1`: ECE 0.0002 but −6pts — the weighting is load-bearing and its tilt is analytically invertible). **E/F**: save and plan heads trained jointly cost the trunk 4-5pts at *any* loss weight → the **two-stage graft recipe** (train the trunk clean, freeze it, graft the heads) that is still in use. |
| **09-06** | `pos_weight` sweep (1/4/8/16/24/32) → **H24 deployed**. Live panel reworked into three baskets with no probability numbers, on the reasoning that a tilted logit is not a confidence and should not be shown as one. **In-game overlay** (frameless pywebview). `/api/live` serves options only (halves per-poll work). Health check moved off `/api/status`, which scans a multi-GB table and outlives any timeout. |
| **09-07** | Five owner-requested improvements: nightly-task auto-recovery, streaming `baseline.py`, a gold eval harness, per-label `pos_weight`, and a stability metric with hysteresis. Then a product decision: **the owner chose a closed beta over more model work**, on the grounds that the open questions had stopped being model-shaped. Launcher, bundle and runbook followed. |
| **09-08** | Fresh export (**17.7k games / 2.13M visits**) — the streaming rewrite from the day before is what let it complete at all. H24 retrained on it. Three-way weight comparison: the simple monotone **8:24 schedule beat both H24 and a hand-shaped alternative**. Beta bundle built, and debugged (`docs/history.md`). And the gold work began: a first measurement, two corrections, a forward walk, a corpus refresh. |
| **09-09 (am)** | GOLDX candidate trained on that forward walk and deployed; nightly recipe updated to match. A "+14 points in actual games" claim was made and is now **withdrawn** (`docs/history.md`, §5). |
| **09-09 (pm)** | **External audit.** Four fatal findings, all confirmed in code within minutes. Fixed the same day: causal gold estimator with a no-future test, versioned gold input with a hard gate, featurization parity through one function, frozen temporal 3-way split, policy-level evaluation, save disaggregation, honest stability decomposition. A follow-up round fixed the split's `post_cutoff` protocol and a metric whose point estimate and confidence interval described different populations. |
| **09-09 (eve)** | Serving contract tightened: scoring only happens inside an explicit shop session, artifacts need provenance and a digest bound in `deployment_manifest.json` to serve, and **raw Match-V5 responses are archived locally as gzip** so a future reconstruction change costs no API time. |
| **09-09 (night)** | **First honest measurement in the project's life**: a 3k-game causal probe, two trained artifacts on one export, paired by game, `probe_only`. Causal gold 32.0% first-action-exact vs stale gold 20.4%, paired +11.6 [+11.3, +11.9], zero unaffordable. And the finding that matters more than the headline (`docs/history.md`, last row of the ledger). |

### 2.2–2.4 Withdrawn beliefs, engineering failures, what they share

Moved verbatim to **`docs/history.md`**. Append new withdrawn beliefs there.

---

## 3. What is running right now

Run **`scripts/status.py`**. It reports scheduled tasks, active log tails,
the served artifact digest, corpus generations and `git status` by reading
the machine, so it cannot be stale the way this section always was.

---

## 4. Data and metrics — with the caveats attached

**Export** (regenerated by `baseline.py`; the numbers below predate the 3-way
split, so re-export before quoting them):

```
games 17,685   usable shops 2,130,985   labels 200   completed 450,625
decisions   complete 668,855   component 506,254   start 828,202   save 127,674
basket      avg 1.49 items per shop
```

**Every accuracy number this project has published shares three defects**, all
confirmed by the audit:

1. They score `predict_top3` or `predict_baskets` against `label_id` — a
   canonical item picked by `baseline._label` (the completed item, else the
   priciest). **That is not the action the user sees.** `predict_options` —
   plan probe, recipe filters, three alternatives, exact budget, save card — was
   never scored until `scripts/eval_policy.py` (new, §7.2).
2. They were computed on a split that had already been used to choose
   architecture, `pos_weight`, threshold, blend and graft recipe. It was a
   validation set being reported as a test set.
3. The `complete` class (0.79) is largely the determinism of that
   canonicalisation, so the headline is flattered by the easiest decisions.

So treat the table below as **historical model-selection signal**, not
performance:

| Model | canonical top-1 | top-3 | complete | component | start | save | basket F1 | exact |
|---|---|---|---|---|---|---|---|---|
| champion-frequency baseline | 0.13 | 0.30 | — | — | — | — | — | — |
| A (big arch) | 0.61 | 0.85 | — | 0.48 | — | 0.44 | 0.555 | — |
| B (+runes, +gold est) | 0.63 | 0.86 | 0.77 | 0.50 | — | 0.47 | 0.571 | — |
| H24 (deployed 09-06→09-09) | 0.63 | 0.87 | 0.79 | **0.54** | 0.59 | 0.37 | 0.558 | 0.29 |
| sched 8:24 | 0.63 | 0.86 | 0.79 | 0.52 | 0.60 | 0.42 | 0.573 | 0.32 |
| custom bump 8→24→12 | 0.63 | 0.87 | 0.79 | 0.53 | 0.59 | 0.38 | 0.567 | 0.31 |
| GOLDX + 8:24 (deployed, leaky gold) | 0.65 | 0.88 | 0.79 | 0.53 | 0.63 | 0.55 | 0.574 | 0.35 |

The 0.13 baseline is also too weak to carry weight: it ignores inventory, gold,
minute, role and legality. `scripts/eval_conditional_baseline.py` (new) builds
the baseline the model should actually have to beat.

---

## 5. Gold — the whole story, current as of the audit

Four quantities are called "gold". Confusing them has now produced two wrong
conclusions, so this section is the reference.

### 5.1 The four quantities

| Name | Where | What it is | Model input? |
|---|---|---|---|
| `leftover` / `gold_left` / export `gold` | `reconstruct.close_visit` | `currentGold` from the frame **before** the visit. Frames are 60 s apart, so it is stale by up to a minute. | ✅ yes (the pre-09-09 budget) |
| `gold` (payload) / export `gold_arrival_true` | `shop_econ.arrival_from_parts` | `leftover + the visit's own net spend`. **Not arrival gold** — it adds the spend back without removing the gap's income, ≈ **+630 g** too high — and it is derived from the label. | ❌ **banned**, and not a valid eval reference either |
| **`gold_est`** | `reconstruct._attach_gold_estimates` | A **causal approximation** of gold on arrival, built only from pre-decision information. Tagged `prequential-v2`. | ✅ yes, under GOLDX |
| live `currentGold` | `app/live.py` | The client's exact gold. Written to `gold` and `gold_est`, tagged `live-exact-v2`. | ✅ yes |

The engraved anti-leak rule stands: **never feed a quantity reconstructed from
the visit's own purchase.** The audit added a second rule of equal weight:
**never feed a quantity reconstructed from anything after the decision.**

### 5.2 What `gold_est` is now (prequential-v2)

```
gold_est(t) = f0.currentGold                      # last frame at or before t
            + passive income f0 → t               # deterministic, 2.04 g/s after 110 s
            + kill/shutdown bounties in (f0, t]   # events, exact ts and amounts
            − purchases/sells/undos in (f0, t)    # the visit's own buys share t exactly
            + prior_residual × (active / window)  # forecast, see below
```

`prior_residual = max(0, (f0.totalGold − fprev.totalGold) − passive(fprev,f0) −
bounties(fprev,f0))` — the farm/assist/objective income the player was earning
over the **previous complete** frame window, extrapolated forward at that rate,
not credited for the last 12 s (`SHOP_DEAD_TIME_MS`, spent walking or recalling).

`totalGold` is cumulative income, so spending never enters that term.

The dead time is subtracted from the credited span but not from the window the
rate is measured over. That asymmetry is deliberate: the prior window had no
reason to end in a shop stop, so its rate is a wall-clock rate. The effect is a
conservative under-credit of roughly one dead-time of income (order 60-120 g at
mid-game farm rates) — a calibration bias, not leakage, and one that can only
under-state the budget.

### 5.3 What it used to be, and why that was fatal (leaky-v1)

The 09-08 version computed the residual from `series[i+1]` — the frame **after**
the visit — and prorated part of it into the estimate. Two problems:

- **Availability**: at time t that frame does not exist. Live inference reads
  exact gold, not a share of future income, so training and serving were
  constructing the feature differently.
- **Label correlation**: income between t and f1 partly *results from* the
  purchase being predicted. Bounties were excluded, which removed the strongest
  channel, but farm, assists and objective gold after the buy were not.

The audit called it temporal leakage. Correct. `scripts/test_gold_causal.py`
now pins the fix: mutate any later frame or any post-t event and `gold_est`
must not move, with controls proving pre-decision information *does* move it.

### 5.4 What is validated, and what is not

| Check | What it proves | What it does NOT prove |
|---|---|---|
| Wallet identity: `cur(f1) == cur(f0) + Δtotal − spends(f0,f1]` — exact (≤5 g) on **92.6 %** of windows, p90 residual 5 g | The **spend pricing** is right (recipe discounts, undos, sells) | Nothing about the income terms. It reads both frames, so it cannot check how income is split between them |
| Afford coverage: whatever true arrival was, it covered the visit's spend — stale+500 **90.5 %**, est+150 **95.8 %**, est+300 **98.4 %** (leaky-v1 numbers; re-run for v2) | The budget is usable: recommendations are not priced out | That the estimate is close to the truth |
| Error vs `gold_arrival_true` | **Nothing.** That reference is inflated and label-derived | — |

**Validating the income forecast requires live telemetry** — exact gold at a
known timestamp with the purchase that followed. Match-V5 cannot provide it.
Until then, `gold_est` is a defensible approximation and nothing more.

### 5.5 What is claimed and what is withdrawn

| Earlier claim | Status |
|---|---|
| "gold_est is exact / leak-free arrival gold" | **withdrawn** — it is causal and approximate |
| "the live regime costs −19.4 points" | **withdrawn** — measured against the inflated field |
| "the live regime costs −12.0 points" | **weakened** — real harness, but the "live" arm was the leaky estimate, scored on canonical labels, on a contaminated split |
| "GOLDX scores 0.65 live → +14 points in game" | **withdrawn** — a training-log number against a harness number, different metrics, unpaired, leaky budget |
| "the stale-gold budget is systematically low" | **stands** — frames are 60 s old; the direction is not in doubt, the magnitude is |

What replaces them: nothing yet. The honest comparison is the ablation in
§11.2, on the frozen split, with the deployed decoder.

### 5.6 The version gate

`gold_est` alone is not enough — a corpus can hold two estimator generations at
once, which is how a leak survives a refactor. So every row carries
`gold_est_version`, and `PREFIX_GOLDX=1` accepts only `prequential-v2` (offline)
or `live-exact-v2` (live). Anything else — including the untagged leaky-v1 rows
that fill the corpus today — **raises**, rather than silently falling back to
stale gold and producing a model trained on a mixture nobody can describe.

Practical consequence: **GOLDX cannot train until the corpus is
re-reconstructed** (`--backfill --refresh`, ~18 h of API time, checkpointed).
The artifact records `gold_input: prequential-v2 | stale-frame-gold`, never
"exact".

---

## 6. What the audit changed (this session)

All eight required regression tests pass (`scripts/test_audit.py`,
`scripts/test_gold_causal.py`).

**Phase 1 — causal gold and parity**

- `reconstruct.py`: `gold_est` rebuilt as prequential-v2, `GOLD_EST_VERSION`
  exported, debug payload carries `prior_residual`/`frame_ts`/`bounty`/`spent`/
  `version`. All reads of the later frame removed.
- Version tag propagated through `baseline.py` → `stream_rows` → `ShopDataset`;
  `live.py` tags rows `live-exact-v2`; cache key bumped to `v8`; artifacts
  record `gold_input`.
- `train_prefix.apply_config()` is now the single place featurization knobs are
  set, used by `Predictor` and every harness. It previously lived in three
  hand-copied lists, and the sampled gold harness had dropped `USE_GOLDX`.
- `eval_live_gold_sampled.py`: parity asserts, both gold fields forced per arm,
  the "honest live regime" arm renamed to "causal offline gold estimate (NOT
  exact live gold)".
- `daily_pull.py`: the nightly now writes `prefix_candidate_<date>.pt` and
  **never overwrites the served model**. Promotion is manual, after evaluation.

**Phase 2 — honest evaluation**

- `baseline.py` writes **train / val / test** (70/15/15), split **temporally by
  `game_creation`**, whole games, ties kept together, with
  `data/ml/split_manifest.json` (eval version, strategy, cutoff, counts,
  per-split id hashes, fingerprint) and `split_assignment.json`. The split is
  **frozen**, and freezing has a protocol (§6.1).
- `train_prefix.py` scores itself on **validation only** and no longer opens
  `visits_test.jsonl`.
- `scripts/eval_policy.py` (new) scores `predict_options` — the real policy —
  with labels stripped from the row, one budget in every channel, and a hard
  check that no shown basket exceeds the budget. Two distinct headline metrics,
  each with its population declared: **`first_action_exact`** (all visits: SAVE
  right on a no-buy, exact basket on a purchase) and
  **`first_basket_exact_given_purchase`** (purchases only). Point estimate and
  bootstrap always come from the same filtered population — the first version
  reported a purchases-only point next to an interval that had also eaten the
  no-buys. `--compare B.pt` adds a **game-paired A−B bootstrap**, which is the
  test for choosing between candidates. All intervals go into the JSON, not
  just the console. `--split test` requires `--confirm-final` and appends to
  `data/ml/test_set_usage.log`.
- `app/predictor.py` re-applies its artifact's config before every forward
  pass, so loading two Predictors in one process (which `--compare` does) can't
  leave one of them featurizing rows for the other's model.
- `scripts/eval_conditional_baseline.py` (new): 5-level backoff baseline built
  on train only, plus `conditional_action_concentration` (modal share, distinct
  actions, normalised entropy per state bucket) — explicitly *not* a ceiling.

### 6.1 The split protocol (read before touching `baseline.py`)

A frozen split is only meaningful if it stays temporally sound, and the first
attempt at this got it wrong: it parked newly ingested games in **train**, to
protect the test ids. That protects the wrong thing. Games arriving after the
manifest was drawn are newer than the test cut, so training on them lets the
model see the future relative to val and test — the exact contamination the
freeze exists to prevent, with the test ids left intact as a fig leaf.

The rule now:

- games with no assignment go to **`post_cutoff`** and are written to **no
  artifact** — not train, not val, not test;
- the manifest records `post_cutoff_count`, `cutoff_utc`, the date range and an
  id hash for that bucket;
- `route_split` drops those rows, and anything unassigned defaults to
  `post_cutoff` rather than to train.

**Operational consequence, state it to the owner:** while a manifest is frozen,
the nightly pipeline keeps growing the corpus but the *trainable* data stands
still. Fresh games only become training data on a deliberate `--rebuild-split`,
which bumps `eval_version`, archives the old manifest as
`split_manifest_v<N>.json`, records what it supersedes, and prints a banner
saying results are no longer comparable. The nightly never passes that flag. So
periodically someone must decide: keep the benchmark, or take the new data and
start a new evaluation version. Both are defensible; drifting into it silently
is not.

**Phase 3 — save and stability**

- `save_kind` ∈ {`no_buy_death`, `ward_only`, `build_spend`} in the export. The
  save head trains **only** on `no_buy_death`; ward-only visits keep their ward
  items as real labels. `save_aux` is renamed `no_build_spend` and labelled a
  diagnostic, not save precision.
- `eval_stability.py` decomposes churn by cause and states in its own docstring
  that equal inventory and level is not an unchanged game state.

**Phase 4 — serving contract and prospective beta evidence**

- `app/main.py` no longer forwards the shop-only policy on a two-second poll.
  The player opens a 90-second decision session in the shop; only then is the
  model run, once, against the exact client wallet. Polling subsequently only
  observes its outcome.
- `app/telemetry.py` writes local-only, append-only JSONL: anonymous decision
  context, exact query wallet, the displayed actions and a later net inventory
  addition or explicit no-buy timeout. A removal-only delta is **unscored**;
  it is not mislabelled as a purchase. Cancellation/supersession is abandonment,
  never a no-buy. `scripts/eval_shop_telemetry.py --strict` makes any displayed
  basket over the exact wallet a hard-stop failure.
- The old plan decoder's 3,500g synthetic probe was removed. Plan arrows now
  use the next-final head at the observed state; it does not pretend to be an
  answer to a fabricated high-gold state.
- `app/deployment.py` fail-closes live serving. The manual
  `scripts/promote_served_model.py --report ...` command binds the exact served
  artifact digest to a provenance-checked **validation** policy report with
  zero unaffordable options. `build_beta.ps1` refuses to package any other
  model.

---

## 7. Results from the new tooling

### 7.1 The 3k causal probe (2026-09-09 night) — the first trustworthy number

Two artifacts trained on one export and scored against each other, paired by
game. Report:
`data/ml/probes/causal_v2_newest_3000/policy_eval_val_probe_candidate_prequential_vs_probe_candidate_stale.json`.

| | A: prequential-v2 | B: stale frame gold | paired A−B (CI95) |
|---|---|---|---|
| first_action_exact | **0.320** | 0.204 | **+0.117** [+0.113, +0.120] |
| first_basket_exact_given_purchase | 0.319 | 0.202 | +0.116 [+0.113, +0.120] |
| basket precision / recall | 0.520 / 0.491 | 0.402 / 0.352 | +0.118 / +0.139 |
| any_of_three_exact | 0.402 | 0.252 | +0.150 |
| action_family_match | 0.613 | 0.511 | +0.102 |
| no_buy_death_recall | 0.413 | 0.292 | +0.121 |
| save_card_precision | 0.098 | 0.037 | +0.060 |
| unaffordable baskets | 0 | 0 | — |

591 validation games, 69,857 visits, both artifacts `probe_only`, same
`export_fingerprint`, neither promoted.

What this is: the first result in the project's history where the comparison is
between two *trained* models, on identical data, paired, with provenance. The
direction of the gold work is supported. What it is not: a product number, or a
result on the full corpus — it is 3,000 of ~20,000 games, and the newest ones,
which is the most homogeneous slice available.

**Three things in it matter more than the headline:**

1. **The stale-gold model (0.204) ties the conditional lookup table (0.208).**
   On the metric that describes what the user sees, the architecture we have
   been refining for a week was worth nothing over a backoff table — until the
   budget was fixed. This is the single most important finding so far, and it
   reframes every earlier "the model beats baseline" claim.
2. **The save card is the weakest part of the product.** It is shown on 5,304
   visits when only 1,253 are real no-buys, and 90% of the time it is shown it
   is wrong. Since "hold your gold" is the differentiator, this is a product
   problem, not a metric footnote. Note also that `save_card_precision` counts
   the card appearing in *any* of the three options, which mixes "we advise
   saving" with "we offer saving as a situational alternative" — measure it per
   tier before drawing conclusions.
3. **The intervals are narrow for a structural reason, not a magical one.** A
   game bundles ten players' decisions, so a per-game mean already averages
   heterogeneous trajectories and the between-game variance is small. The
   interval is correct, but it describes sampling variation *within this 3k
   slice*; it says nothing about variation across patches or the rest of the
   corpus. Expect movement well beyond ±0.3 points at full scale.

Open question worth answering before the full run, and it costs no training:
**why is causal gold worth eleven points?** A breakdown of the error by price
band and basket size would say whether the model learns better items, or
whether the stale-budget decoder was simply truncating baskets it could not
afford.

### 7.2 Stability decomposition

The stability decomposition ran on 300 consecutive snapshots (2 s cadence,
deployed artifact, real hysteresis):

```
poll pairs 299   inventory+level unchanged: 283
   of those, fully identical discrete state: 79
lead churn, all polls            0.060
with inventory+level unchanged:  0.035   (0.028 as displayed)
why those 10 lead changes happened
  an on-screen item crossed its price   6
  the board / KDA / objectives moved    0
  gold moved, no on-screen crossing     4
  nothing discrete changed (FLICKER)    0
```

**Zero residual flicker.** Every change had a cause in the input. So the
"2.9 % flicker with identical state" framing was wrong. This is now a replay
diagnostic only: Phase 4 removed continuous live re-ranking and its hysteresis
from the serving path. Do not reintroduce it unless a real shop detector or a
separately validated session-state update policy exists.

---

## 8. Repo map (with the gotchas attached)

- **`app/riot.py`** — rate-limited httpx client (1.3 s gap, 95 req/2 min). 404
  returns `None`; 401/403 raises and aborts ingest. Successful Match-V5 match
  and timeline responses are atomically archived in ignored
  `data/raw_match_v5/`; replays reuse those immutable source payloads instead
  of making a duplicate API crawl (`RAW_MATCH_V5_CACHE=0` is an explicit
  ephemeral-run opt-out).
- **`app/reconstruct.py`** — the replay: shop visits, inventories, ten-player
  board, objectives, save events, `gold_est`. The support-item line (Atlas 3865
  → Compass → Bounty → finished) never emits `ITEM_PURCHASED`; it is seeded and
  upgraded via `ITEM_DESTROYED` — special-cased, don't "simplify" it.
- **`app/shop_econ.py`** — recursive combine costs, affordability, decision
  kinds, `GOLD_DRIFT = 500`, `SAVE_ITEM`, `PINK` (control wards, the only
  consumables not skipped).
- **`app/db.py`** — SQLite (multi-GB, WAL). `matches` + `shop_visits` are
  per-player perspectives; `games` + `game_events` the full lobby. `init_db()`
  migrates in place.
- **`app/ladder.py` / `pros.py` / `backfill.py` / `ingest.py`** — ingest modes.
  `--backfill --refresh` re-reconstructs the whole corpus, checkpointed in
  `data/refresh_done.txt`.
- **`scripts/baseline.py`** — export + split + manifest. Streaming — see the traps in `docs/history.md`.
- **`scripts/train_prefix.py`** — the model: 10-token board (token 0 the
  shopper; the enemy in the shopper's role gets a distinct side id), query
  vector with state/affordability/damage-share/runes, multi-label BCE with
  legality masking, count + save + plan heads, two-stage graft. Knobs:
  `PREFIX_EPOCHS/DMODEL/LAYERS/FF/HEADS`, `PREFIX_COSINE`, `PREFIX_RUNES`,
  `PREFIX_GOLDEST`, `PREFIX_GOLDX`, `PREFIX_POSW`, `PREFIX_POSW_SCHED`,
  `PREFIX_POSW_CUSTOM`, `PREFIX_SAVEW/TARGETW`, `PREFIX_INIT_FROM`,
  `PREFIX_FREEZE`, `PREFIX_OUT`, `PREFIX_HISTORY` (no gain, off).
- **`app/predictor.py`** — loads an artifact and **reuses the training
  featurization** (`ShopDataset` on a one-row list). Never featurize live data
  any other way. `predict_options` is the shipped policy.
- **`app/main.py`** — FastAPI. `/api/live` shapes state and observes an active
  session but emits no advice; `POST /api/live/session` runs the model once
  after the player's explicit request. `/api/status` scans the events table;
  never health-check on it.
- **`app/telemetry.py`** — local-only prospective session evidence; no Riot
  identity, raw snapshots or opponent data. **`app/deployment.py`** — the
  fail-closed promotion binding for a served artifact.
- **`app/live.py`** — Live Client Data API poller. `python -m app.live --dump`
  saves snapshots (20 k in `data/live_snapshots/`, all Practice Tool).
- **`app/overlay.py`** (dev) / **`app/advisor.py`** (beta, in-process server,
  PyInstaller-safe).

---

## 9. Traps

Moved verbatim to **`docs/history.md`**.

---

## 10. Beta status

The owner decided on 2026-09-08 to run a closed beta (10–30 testers, 2–4 weeks)
rather than keep polishing the model, because the open questions are
product-level: does the advice *feel* trustworthy, is the overlay in the way, is
the save card understood.

Built and verified: `dist/BuildAdvisor-beta.zip` (174 MB, frozen
`app/advisor.py`, CPU torch, assets beside the exe — launch tested),
`scripts/build_beta.ps1`, `docs/beta.md` (tester instructions, patch-day
runbook, checklist, **Personal API Key application draft**).

**Testers need no Riot API key**: the overlay talks only to `127.0.0.1:2999`
and the Data Dragon CDN.

**Blocker: the Personal API Key.** Also pending: a freshly reconstructed,
provenance-valid candidate, deliberate promotion, validation in the owner's
own ranked games, a feedback channel, minimal branding. The legacy artifact
will now be blocked rather than silently packaged.

The audit's beta advice is worth heeding: instrument first, and start smaller
than 30 testers. Phase 4 now records the minimum actionable facts — artifact
version, displayed options, exact query gold, role/minute and an observed
inventory addition or no-buy timeout — without retaining identity or a raw
snapshot. Latency and a one-tap useful/wrong rating are **not implemented**;
do not claim a feedback or satisfaction stopping rule until they are. The
existing evidence supports affordability and behavioural-match analysis, not
whether users found the advice helpful.

---

## 11. The queue, in priority order

### 11.1 Re-reconstruct the corpus, then re-export (P0, blocks everything)

Nothing GOLDX can train until rows carry `prequential-v2`, and the frozen split
does not exist until `baseline.py` runs once.

```powershell
# finish/redo the corpus refresh (checkpointed; needs a live key)
.venv\Scripts\python.exe -m app.ingest --backfill --refresh
# rebuild the export, and create the frozen 3-way split the first time
.venv\Scripts\python.exe scripts\baseline.py --rebuild-split
```

Check afterwards: `data/ml/split_manifest.json` shows
`gold_est_versions: {"prequential-v2": <all rows>}`, three non-empty splits and
`post_cutoff_count: 0`. After that, plain `baseline.py` (no flag) is the routine
command; it parks newer games in `post_cutoff` and leaves the benchmark alone.
Only redraw when you have decided the backlog is worth a new eval version
(§6.1).

### 11.2 The gold ablation (P1 — the only modelling question that matters now)

Three candidates, identical but for the budget, scored with the deployed
decoder on validation:

```powershell
$env:PREFIX_COSINE="1"; $env:PREFIX_DMODEL="128"; $env:PREFIX_LAYERS="4"
$env:PREFIX_FF="256"; $env:PREFIX_HEADS="8"; $env:PREFIX_RUNES="1"
$env:PREFIX_POSW_SCHED="8:24"; $env:PREFIX_EPOCHS="16"
$env:PREFIX_SAVEW="0"; $env:PREFIX_TARGETW="0"

# A: stale frame gold (no GOLDX, no interpolation)
$env:PREFIX_GOLDEST="0"; $env:PREFIX_GOLDX="0"; $env:PREFIX_OUT="trunk_stale.pt"
.venv\Scripts\python.exe scripts\train_prefix.py
# B: own-rate interpolation only (the model B feature)
$env:PREFIX_GOLDEST="1"; $env:PREFIX_GOLDX="0"; $env:PREFIX_OUT="trunk_interp.pt"
.venv\Scripts\python.exe scripts\train_prefix.py
# C: causal prequential estimate
$env:PREFIX_GOLDEST="1"; $env:PREFIX_GOLDX="1"; $env:PREFIX_OUT="trunk_prequential.pt"
.venv\Scripts\python.exe scripts\train_prefix.py
```

Graft heads onto each (`PREFIX_INIT_FROM=<trunk>`, `PREFIX_FREEZE=1`,
`PREFIX_EPOCHS=6`, `PREFIX_SAVEW=0.5`, `PREFIX_TARGETW=0.25`,
`PREFIX_OUT=cand_<name>.pt`), then compare them **pairwise**:

```powershell
.venv\Scripts\python.exe scripts\eval_policy.py --artifact cand_prequential.pt --compare cand_stale.pt --split val
.venv\Scripts\python.exe scripts\eval_policy.py --artifact cand_prequential.pt --compare cand_interp.pt --split val
```

**Success criterion: the paired A−B bootstrap interval excludes zero** on
`first_action_exact` and `action_family_match`. Do *not* use "the two
candidates' individual CI95 don't overlap" — the candidates are scored on
identical visits, so the difference has far less variance than either estimate,
and independent intervals are both the wrong question and much too
conservative. `--compare` resamples the same games for both models and reports
the interval of the difference.

If prequential does not win, the whole GOLDX line was a distributional change
with no measurable payoff — say so and revert the recipe. ~90 min per trunk on
the 4060; do not run two at once.

### 11.3 Establish where the model actually stands (P1, no training)

```powershell
.venv\Scripts\python.exe scripts\eval_policy.py --split val --games 300
.venv\Scripts\python.exe scripts\eval_conditional_baseline.py --split val
.venv\Scripts\python.exe scripts\eval_stability.py --stride 1
```

The first two together answer the question nobody has answered: does the
transformer beat a conditional lookup table on the policy the user sees? If the
gap is under ~2 points of family-match with overlapping intervals, the
architecture is not earning its cost.

### 11.4 Only after all of the above (P2)

**Do not read the final test set yet.** Selection happens on validation, full
stop; `--split test --confirm-final` is for one confirmatory read of the single
artifact you have already chosen, and every read is logged to
`data/ml/test_set_usage.log`. Then: the save↔ward hysteresis
dial, a `pos_weight` schedule sweep, the plan-head decode blend re-measured
under the new budget. Capacity is saturated (d160 flat, d192 worse) — not there.

---

## 12. Uncommitted work

Use `git status`. This section was a hand-typed file list that went stale
between sessions; `scripts/status.py` prints the live one.

---

## 13. Verifying you haven't broken anything

```powershell
.venv\Scripts\python.exe scripts\run_tests.py
```

Runs every `scripts/test_*.py` in its own subprocess and exits non-zero if
any fail. The two that catch the expensive class of mistake are
`test_gold_causal.py` (a feature that reads the future) and `test_audit.py`
(a harness scoring a model in a regime it was not trained in — nothing
crashes, the number is just wrong).

---

## 14. What still cannot be demonstrated

Moved verbatim to **`docs/history.md`**. It is the honest boundary of the
project's claims; keep it current there.

---

## 15. Working notes on the owner

- Spanish speaker; short, direct asks. Technical, comfortable with model-level
  detail, cares about *why* a number moved.
- Pushed back — correctly — when the previous agent proposed a statistical
  patch (gold jitter) instead of solving the gold problem: *"estamos hablando de
  un problema real, que requiere solución real y difícil… quiero el oro
  exacto"*. Prefer the real fix; bring evidence.
- Commissioned this audit and asked for it to be applied without commits and
  without stopping running jobs. Respect both.
- Turns the PC off at night and has asked for automatic shutdown after
  overnight work. **Don't schedule a shutdown unless asked for that session**,
  and abort a pending one (`shutdown /a`) if they come back to the machine.
- Reports honestly what failed, and expects the same. Several of the most
  useful results here were negative: the first gold measurement was invalid,
  the hand-tuned weight schedule lost to the simple one, and the headline gold
  win is currently unproven.

---

## 16. Audit update — 2026-09-10 (historical initial finding; current action in §17)

All eight Phase 0 scripts passed on CPU, but a separate synthetic check
demonstrated that a post-decision kill changes the exported shopper input.
`close_visit` reads `end_kda` while stamping the decision at `ts_start`;
`baseline._example` passes that value into `ShopDataset.query`. A kill at 153s
changed the input for a 150s decision whose final purchase occurred at 155s,
with identical labels and identical causal gold. The end-frame/objective paths
also need review. Corpus prevalence and metric impact are **not measured**.
The existing gold test proves less than whole-input causality.

See `docs/audits/2026-09-10-evidence.md`, the verbatim eight-test output at
`docs/audits/2026-09-10-phase0.txt`, and the bounded reproducer at
`data/audit_reconstruction_probe_20260910.py`. The 3k report values and model
hashes match the ledger; this does not resolve the new input-validity finding.
A newer 10k probe report exists, documented in that audit; it uses a different
population and is still ineligible for promotion. No original saved evidence
for the quoted wallet-identity percentage or zero-flicker run was located in
the searched local reports/logs; those statistics were not reproduced.

At the audit check, Task Scheduler showed `BuildTracker Causal Recovery`
Running; both probes and the daily pull were Ready. The latest logged full
refresh attempt began at 18:26:12; `backfill.log` was advancing through its
5,178-game remaining batch. The supervisor already queues catch-up, export,
three trunk/graft pairs, and policy comparisons. **This audit did not stop or
alter that sequence.** The recommendation is to defer selection and address
the start-state defect first; the owner has not yet decided how to change the
queued stages. Do not start another collector or infer a stop was implemented.

Only audit/procedure files and scratch diagnostic output were written. No
production code, reconstruction checkpoint, model, export, or deployment was
changed, and the final test set was not read. The operating procedures now
live under `docs/skills/`; they are not authorization to launch another job.

## 17. Authorized repair hold and 16.18 transition — 2026-09-10

The owner approved the repair-first action after §16. The code now fixes the
decision-start input defect and tags new events `decision-start-v1`, independently
of `prequential-v2` gold. The existing corpus has NOT been repaired or certified.
`data/ml/pipeline_hold.json` blocks all main export/training entry points. Leave
it in place until the staging repair and export publication gates in
`docs/audits/2026-09-10-repair-and-patch.md` are complete. The active refetch and
serial catch-up are untouched; the supervisor will stop at guarded export.

Patch provenance is now explicit: match DTO patch selects pinned static data;
16.17 uses 16.17.1 and 16.18 uses 16.18.1. Unknown patches fail closed. Daily
collection accepts both transition patches in a single pass. Export requires
`--patch <single-patch>`; daily export/training also needs `--export-patch`.
Changing export patch requires an explicit new split/evaluation version. Do not
clear the hold merely to make the old supervisor's command line work.

Backfill and ladder now publish game/events and requested perspectives in one
transaction, after reconstruction succeeds. Already-running Python retains its
old code. Nothing here retroactively fixes checkpointed games.

The read-only source audit at `data/ml/repair_source_audit.json` found 11,207
raw pairs present among 20,857 stored games and 9,650 missing pairs at its
timestamp. Presence is not integrity; twelve sampled pairs were successfully
replayed. Reinventory after collection finishes. Repair from verified local
sources into staging, explicitly account for exclusions, and decide targeted
retrieval only then. Do not start a duplicate broad API pull. Thirteen regression
scripts passed; details and limitations are in the linked repair report.

## 18. Queued 16.18 sample — 2026-09-10

The owner requested the complete small new-patch sample and explicitly included
tracked pros. `BuildTracker Patch 16.18 Sample` is a one-shot Windows task using
`scripts/run_patch_sample.ps1`; it waits for Causal Recovery and Daily Pull to
stop, then allows the prior rate-limit window to expire. It does not interrupt
them. Do not launch a second copy or infer that a Running scheduler state means
the sample is already fetching: inspect its job log.

Sample directory: `data/patch_samples/16.18_20260910/`. `state.json` freezes the
cutoff and tracked-pro snapshot; future discovery freezes KR/EUW Challenger and
Grandmaster roster responses, uncapped paginated lists, and overlapping cohort
membership. Pros use their configured regional searches, including LEC's Asia
search. It is a tracked-account population, not all professional players globally.
Ladder membership is observed at discovery time, not retrospectively at match time.

Raw match/timeline responses reuse the shared archive. All ten perspectives go
to the sample's own `tracker.db`. Main DB, refresh checkpoint, exports, models
and deployment are untouched by the sample. `report.json` will separate stored,
excluded, quarantined and unresolved-pro cases, with overlapping cohort counts.
Scope is not certified complete when a pro is unresolved or a source quarantined.
No real sample count or quality result is available before the worker finishes.

New ingestion CLI and daily entry points share `app/collector_lock.py` with the
sample. Existing processes predate this lease, so the scheduler-state gate must
also remain. Offline tests passed for frozen-window pagination, pro/ladder
deduplication, restart behavior, invalid source rejection, collector exclusion,
and real reconstruction into a temporary ten-perspective staging DB.

To inspect: read the sample's `job.log` and task state. After an auth failure,
resolve the key as usual and restart this same one-shot task to resume; never
create another collector. Changes to pinned collection code require review
before resume. The production pipeline hold remains in force.
