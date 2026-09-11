# History — what this project got wrong, and the traps that cost hours

Extracted verbatim from `HANDOFF.md` (§2.2, §2.3, §2.4, §9, §14) so it survives
independently of that file's live-status sections. Read this before proposing
anything that sounds like a breakthrough — most of them have been tried.

---

## Beliefs this project held, and had to withdraw

Every row here was believed by someone competent, acted on, and later
disproved. They are kept so nobody re-derives them.

| We believed | Why it was plausible | What disproved it | What stands now |
|---|---|---|---|
| `gold_left + net_spend` is the player's arrival gold | It is the arithmetic the wallet implies | Two separate discoveries: it is derived from the label (09-02), and it is not even arrival gold — it adds the spend back without removing the gap's income, ≈ **+630 g** too high (09-08) | Banned as a feature *and* as an evaluation reference |
| Exact live gold is a free win — live inference is *better* off than training | Live reads the client's real wallet; training reads a 60 s-stale frame | Feeding "exact" gold made accuracy collapse (0.63 → 0.43) | The measurement was invalid (see next row), but the assumption was never evidence in the first place |
| The live regime costs −19.4 points | Measured with a real harness | The "exact gold" arm was the inflated field above | Withdrawn |
| Then: the live regime costs −12.0 points | Better harness, better reference | The "live" arm was still a leaky estimate, scored on canonical labels, on a contaminated split | Weakened to "the direction is certain, the magnitude is unmeasured" |
| `gold_est` is label-free, therefore legitimate | It never reads the purchase | The audit found it read the frame **after** the decision to build its income residual: temporal leakage, and post-purchase income partly *caused by* the label | Replaced by `prequential-v2`, pinned by a no-future unit test |
| GOLDX gives +14 points in actual games | Two numbers, both real | They came from different metrics (a training log vs a harness), unpaired, on a leaky budget | Withdrawn. Replaced 09-09 night by a paired ablation: **+11.6 points on a 3k probe**, which is evidence for the same direction and is not the same claim |
| Prediction flickers on 2.9% of polls with identical state | Measured on 20k real snapshots | "Identical state" meant only equal inventory and level; gold, clock and board kept moving. Decomposed by cause: **0 residual flicker** | Hysteresis is delaying legitimate updates, not suppressing noise. Not removed — 300 Practice-Tool snapshots is not enough to remove it either |
| The test split measures generalisation | It was split by match, no visit leakage | Architecture, `pos_weight`, threshold, blend and graft recipe had all been chosen while looking at it | It was a validation set. Frozen temporal 70/15/15 split introduced; test untouched |
| top-1 / top-3 measure the product | They were the numbers the trainer printed | They score a canonical stand-in label (completed item, else priciest), never the three baskets the user sees | Demoted to diagnostic; `eval_policy.py` scores the displayed policy |
| Freezing the test IDs keeps a frozen split honest | The held-out games never change | Newly ingested games were being parked in **train**, and they are newer than the test cut — training on the future relative to the held-out set | New games go to `post_cutoff` and enter no artifact (see the split protocol in `HANDOFF.md`) |
| A hand-shaped `pos_weight` bump (8→24→12, save pinned low) would beat the monotone schedule | It encoded real knowledge about which classes want which weight | It landed between H24 and the simple 8:24 on every metric | The simple monotone schedule won. Recorded because the losing idea was the better *story* |
| The transformer is clearly worth its complexity | It beat a champion-frequency baseline 0.63 vs 0.13 | On the displayed-policy metric, the stale-gold model scores **20.4%** and a conditional lookup table scores **20.8%** | Unresolved and important. Only the causal-gold model (32.0%) clears the table. Until the full-corpus run, assume the architecture has not yet earned its cost |

## Engineering failures worth remembering

Cheap to hit again, so they are named with their symptom:

- **The export outgrew RAM twice** (09-03, 09-04). 2.1M rows as Python dicts is ~13 GB on a 15.8 GB machine. Symptom: a silent seven-hour hang, no traceback, machine unusable. Both `baseline.py` and `train_prefix.py` are streaming-only now.
- **Long background jobs started from an agent session get killed** (~3 min). Two overnight chains died mid-`baseline.py` with no error at all. Everything heavy now runs as a one-shot Windows scheduled task that unregisters itself.
- **The Riot dev key expires every 24 h** and killed three jobs in two days, twice mid-refresh. Regenerating in the portal invalidates the previous key, so regenerating while a job runs kills that job.
- **`$hashA + $hashB` throws in PowerShell 5.1** on duplicate keys. Crashed an overnight chain between two stages, after the expensive one had succeeded.
- **A `Start-Transcript`-based watcher died before writing its first line**, so the failure was invisible. Watchers now use `Add-Content` and ASCII only.
- **PyInstaller `--windowed` gives a process with `sys.stdout is None`**, and uvicorn's log formatter calls `.isatty()` on it. The beta exe died instantly on launch, with no console to say why.
- **Editing `daily_pull.py` mid-run changes nothing** — Python had already loaded it — while its *subprocesses* do pick up new code at launch. That asymmetry has caused one wasted night.
- **`refresh_done.txt` is purpose-agnostic.** Reusing it across two different refresh campaigns would have silently skipped 9,076 games.
- **An eval harness silently dropped one featurization flag** (`USE_GOLDX`) and would have scored a model in a regime it was never trained in. Nothing would have crashed.
- **A metric's point estimate and its confidence interval were computed over different populations** — purchases for one, purchases plus no-buys for the other.
- **Near miss**: an automatic shutdown scheduled for "when the overnight work finishes" fired while the owner was back at the machine. Aborted with seconds to spare. Never schedule a shutdown that was not asked for in that session.

## What they have in common

Not one of these announced itself. No crash, no stack trace, no red test — every
single one produced plausible output that a reasonable person acted on. The
RAM failure looked like a slow machine. The leaky gold looked like a
breakthrough. The contaminated split looked like a good score. The flicker
metric looked like a product insight.

So the operating assumption for this project is: **the failure mode is not
breakage, it is numbers that look right.** That is why the tests (`scripts/run_tests.py`) assert
properties rather than outputs (a feature cannot read the future; a point
estimate and its interval share a population; an artifact's config reaches the
featurization), and why nothing gets promoted on a number that has not been
produced by the deployed policy on a split nobody has shopped against.

---

## Traps (each cost real hours)

1. **Long background processes started from an agent session get killed
   (~3 min here).** Two overnight chains died mid-`baseline.py` with no
   traceback. Use a one-shot Windows scheduled task that unregisters itself,
   and watch its log.
2. **RAM is 15.8 GB; the export is ~4.3 GB of JSONL / 2.1 M rows.** Loading it
   as dicts costs ~13 GB and thrashes (a silent 7-hour hang on 09-03).
   `baseline.py` and `train_prefix.py` stream. Never `list(load_jsonl(...))`.
3. **The Riot dev key expires every 24 h**; 401 aborts ingest. It killed three
   jobs in two days. Verify with a `status-v4` call first. Regenerating in the
   portal invalidates the previous key, so regenerating mid-job kills the job.
   The fix is a **Personal API Key**.
4. **Windows PowerShell 5.1**: `$hashA + $hashB` throws on duplicate keys (this
   crashed an overnight chain). No `&&`/`||`, no ternary. Prefer ASCII-only
   scripts and `Add-Content` logging — a `Start-Transcript` watcher died before
   writing anything.
5. **A running Python process holds the old module.** Editing
   `daily_pull.py` mid-run changes nothing; its subprocesses, however, pick up
   new `baseline.py` / `train_prefix.py` at launch. That asymmetry is exactly
   what made one overnight run fail.
6. **Tensor cache** keys on export size/mtime plus feature knobs (`v8` now). A
   new knob that changes featurization must go in that key.
7. **`refresh_done.txt` is purpose-agnostic.** Rotate it before starting a
   different kind of refresh (the rune refresh's 9,076 ids are archived as
   `data/refresh_done_runes_0904.txt`).
8. **PyInstaller `--windowed`**: `sys.stdout` is `None` and uvicorn's log
   formatter calls `.isatty()` → instant death. `app/advisor.py` shims both
   streams and passes `log_config=None`.
9. **`PATCH_STARTS_UTC`** needs an entry on patch day (ingest warns and falls
   back to a 16-day window).
10. **Frames are 60 s apart and visits are stamped with the frame before them.**
    Root cause of the whole gold story. Any new "current state" feature
    inherits that staleness — ask which regime live will be in.

---

## What still cannot be demonstrated

Be explicit about this with the owner; it is the honest boundary of the project.

- **Live performance.** The collector is implemented but no representative
  telemetry population has been collected yet, so there is no measurement of
  the advice the overlay actually gave in real shop sessions. Every existing
  performance number remains a replay.
- **The income forecast in `gold_est`.** Validating it needs exact gold at a
  known timestamp — i.e. the same telemetry.
- **Whether the advice is good.** Imitation labels say what a Challenger did,
  never whether it worked. No human rating, no outcome-conditioned analysis.
- **Transfer to the actual users.** Testers are not Challenger and their games
  are off-distribution. Nothing measures that gap yet.
- **Manual recalls with no purchase.** Match-V5 cannot observe them, so the
  save class only covers post-death no-buys. Do not invent labels for the rest.

---
