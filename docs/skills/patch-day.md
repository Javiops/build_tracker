# Patch day

Trigger: the target game patch changes. First inspect active jobs and HANDOFF.md. Do not modify a running ingestion sequence or regenerate its API key.

1. Verify patch identity, release timing and item changes against current official Riot sources. Do not infer release time from this document.
2. Inspect `app/config.py:PATCH_STARTS_UTC`, `PATCH_DATA_VERSIONS`, and `CURRENT_INGEST_PATCH`. Preserve the `16.xx` identity. If a start filter is a conservative lower bound rather than a verified regional activation time, say so; the DTO remains authoritative.
3. Pin reviewed Data Dragon versions per match patch. Never globally switch historical reconstruction to latest static data. Verify payload versions and static hashes. Unknown patches must fail closed.
4. Follow the owner's beta runbook gate of collecting 1–2 nights of the new patch before considering a new release. That waiting period alone proves no transfer or quality claim.
5. Deliberately choose the export patch via `baseline.py --patch <patch>` and whether to rebuild its frozen benchmark. Record the evaluation-version break. Export only after verified reconstruction coverage and safe publication, with no competing supervisor. The repair hold is not satisfied by checkpoint IDs alone. Daily collection accepts explicit transition patches; `--export-patch` is a separate choice.

```powershell
.venv\Scripts\python.exe scripts\test_shop_econ.py
.venv\Scripts\python.exe scripts\test_visits.py
.venv\Scripts\python.exe scripts\test_gold_causal.py
.venv\Scripts\python.exe scripts\test_decision_causal.py
.venv\Scripts\python.exe scripts\test_patch_integrity.py
```

Use `run-experiment.md` and `promote-model.md` only after the data gate passes. Failure modes: missing start-time entry; static-data prices disagreeing with historical patch; mixed patches described as one population; automatic promotion after a nightly pull. No patch change was made while authoring this procedure.
