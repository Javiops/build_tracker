# Promote a model

Trigger: the owner has selected and authorized a release candidate after reviewing displayed-policy validation. Resolve known reconstruction defects first. This document does not authorize promotion.

Reject legacy or `probe_only` artifacts, mismatched manifest/export/config provenance, test-set selection, empty reports, nonzero over-budget recommendations, and unresolved data-validity findings. A zero counter only checks the report's offline budget regime; validate exact-wallet serving separately.

After evaluation with `scripts/eval_policy.py --artifact <candidate> --split val --games 0`, verify the report's A digest against the candidate bytes. Preserve the existing served pair before replacement. Copy the selected candidate to `data/ml/prefix_model.pt` only as part of the authorized release operation; then bind its bytes:

```powershell
.venv\Scripts\python.exe scripts\promote_served_model.py --report <validation-report-name.json>
.venv\Scripts\python.exe scripts\test_live_predict.py
.venv\Scripts\python.exe scripts\test_live_api.py
.venv\Scripts\python.exe scripts\test_artifact_provenance.py
```

The report argument is relative to `data/ml`. Use `--replace` only for an authorized replacement of an existing manifest. The script writes the binding; it does not copy the candidate to the served filename. `app/deployment.py` checks digest and provenance at load; restart the serving process after changing the pair. Package only after successful checks:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_beta.ps1
```

Failure modes: copying only the model; evaluating a different digest; assuming a four-snapshot test certifies performance; using a legacy artifact as rollback without valid promotion. Current serving checks trust the local deployment manifest as the manual approval record; they do not independently rerun the report. Never hand-author that record to bypass the promotion command.
