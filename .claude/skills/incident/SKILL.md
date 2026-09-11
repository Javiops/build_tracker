---
name: incident
description: Triage a stalled job, a nonzero stage exit, an authentication error, or a data-validity finding in build_tracker. Use when output stalls, a pipeline stage fails, ingest returns 401/403, or a log looks wrong.
---

# Recovery incident

Trigger: stalled output, a nonzero stage exit, an authentication error, or a data-validity finding.

```powershell
Get-ScheduledTask -TaskName 'BuildTracker*' | Select-Object TaskName,State
Get-Content data\backfill.log -Tail 20
Get-Content data\causal_recovery.log -Tail 20
Get-Content data\daily_pull.log -Tail 20
```

Read HANDOFF.md and the actual supervisor script before acting. A historical error in a log is not proof that the latest attempt failed. Check current task state and whether the active log advances. `causal_recovery.log` may mix encodings from PowerShell append operations; decode a read-only copy if needed.

Do not start a second Riot collector or supervisor. A running Python module retains loaded code; later subprocesses can pick up edits. Editing the PowerShell file does not reliably change the running supervisor's queued instructions.

For authentication failure, confirm the job actually exited before replacing the key; never print `.env` or regenerate a key during an active sequence. Do not retry through 401/403. Preserve the checkpoint and identify its reconstruction campaign before resuming; a checkpoint filename alone does not establish its semantics.

After an authorized restart, use the existing durable scheduled task and serial sequence: full refresh, coverage verification, `daily_pull.py --hours 0 --no-train`, catch-up refresh, coverage verification. Export/training additionally require resolution of known input-validity defects; a gold-version coverage check is not an all-input causal audit.

Prefer local raw-cache replay for reconstruction repairs. Determine coverage before claiming that another API crawl is required. Never launch an expensive scan while the refetch is running merely to estimate progress. Do not schedule shutdown unless explicitly requested in the current session. Record the failing stage, exact exit, recovery action, and unresolved risks in the durable handoff; do not overwrite historical failures.
