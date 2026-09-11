---
name: run-experiment
description: Run a model or decoder comparison in build_tracker under the experiment protocol — recorded question, paired validation selection, isolated scheduled job. Use when comparing trainers, decoders or features.
---

# Run an experiment

Trigger: a model or decoder comparison, after reconstruction correctness is established. Read HANDOFF.md and the latest audit first. The 2026-09-10 start-state finding is unresolved; this procedure is not clearance to train.

Before execution record the question, primary metrics and populations, unchanged export fingerprint, candidate names, stopping rule, and resource budget. Use `first_action_exact` on all scored visits and `action_family_match` on purchases with a known next final. Select on paired validation differences. Never compare different exports as an improvement.

Run only in an isolated, durable one-shot scheduled job after the recovery supervisor is idle. Do not run this alongside its existing queue. In a fresh PowerShell process at the repository root:

```powershell
$env:PREFIX_COSINE='1'
$env:PREFIX_DMODEL='128'
$env:PREFIX_LAYERS='4'
$env:PREFIX_FF='256'
$env:PREFIX_HEADS='8'
$env:PREFIX_RUNES='1'
$env:PREFIX_POSW_SCHED='8:24'
$env:PREFIX_HISTORY='0'
$env:PREFIX_EXTRAS='1'
$env:PREFIX_TBLEND='0'
$env:PREFIX_SPLICE='0'
$env:PREFIX_POSW_CUSTOM='0'
$env:PREFIX_GOLDEST='1'
$env:PREFIX_GOLDX='1'
$env:PREFIX_EPOCHS='16'
$env:PREFIX_SAVEW='0'
$env:PREFIX_TARGETW='0'
$env:PREFIX_INIT_FROM=''
$env:PREFIX_FREEZE='0'
$env:PREFIX_OUT='audit_trunk_prequential.pt'
.venv\Scripts\python.exe scripts\train_prefix.py
# Stop on any nonzero exit before launching the graft.
$env:PREFIX_INIT_FROM='audit_trunk_prequential.pt'
$env:PREFIX_FREEZE='1'
$env:PREFIX_EPOCHS='6'
$env:PREFIX_SAVEW='0.5'
$env:PREFIX_TARGETW='0.25'
$env:PREFIX_OUT='audit_candidate_prequential.pt'
.venv\Scripts\python.exe scripts\train_prefix.py
```

Use unique output names; never overwrite existing experiments. For stale control set GOLDEST/GOLDX to 0/0 and use distinct trunk/candidate names; interpolation uses 1/0. Keep everything else identical. Compare only after both grafts complete:

```powershell
.venv\Scripts\python.exe scripts\eval_policy.py --artifact audit_candidate_prequential.pt --compare audit_candidate_stale.pt --split val --games 0
```

`--games 0` means all validation games; the evaluator default is only 300. Require zero over-budget outputs and paired intervals excluding zero on predeclared metrics. Over-budget is measured against offline estimates, not exact live wallets. Neither canonical training metrics nor the trainer's threshold calibration is a release evaluation.

Resource planning: HANDOFF's historical estimate is about 90 minutes per full trunk; this is not a new measurement. Graft/eval duration and exact disk needs must be measured for the chosen export. Inspect free disk and current JSONL/tensor sizes before launch. Keep one training process at a time on the documented 15.8 GB machine; never materialize all JSONL rows. Preserve manifests, configs, reports, and hashes. No final-test reads, automatic promotion, or architecture sweep this cycle.
