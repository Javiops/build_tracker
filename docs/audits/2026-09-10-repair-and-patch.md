# Repair and patch boundary — 2026-09-10

## Decision and current boundary

The owner approved holding downstream work, fixing decision-start leakage, and
measuring source-backed repair before another export. The owner then required
explicit handling of patch 16.18. The running refetch was neither restarted nor
stopped. Its already-imported reconstruction code is still old. New subprocesses
load the changes below. No production database repair or model promotion has run.

`data/ml/pipeline_hold.json` blocks baseline/probe exports and transformer/RF
training before work starts. Ingestion does not consult this hold. The running
recovery supervisor may finish its serial collection/catch-up stages, then stops
at its guarded export subprocess. A refresh checkpoint is an ID list, not proof
of the new reconstruction contract.

## Implemented protections

- `decision-start-v1` snapshots frames, KDA and objectives at the first shop event.
  Later events can change purchase labels but cannot change these input snapshots.
  Gold remains separately versioned `prequential-v2`.
- The Match DTO's `gameVersion` selects reviewed static data: 16.17 → 16.17.1;
  16.18 → 16.18.1. Unknown patches fail closed. Events record patch, static-data
  version/hash, and reconstruction version. Static cache writes are atomic and
  payload versions are checked. The old cached latest-version list is preserved.
- Current CLI ingest defaults to 16.18. Daily collection explicitly accepts
  16.17 and 16.18 in one match-list pass to retain the transition tail. Wrong-patch
  DTOs are rejected before timeline fetching. The 16.18 start filter is a
  conservative 2026-09-09 UTC lower bound, not a claimed regional activation time.
- Export requires `--patch` and rejects old reconstruction/static provenance.
  Changing export patch requires deliberate split rebuilding and a new evaluation
  version. Trainers require matching manifest provenance; candidates bind static
  data. Daily collection is collection-only unless `--export-patch` is selected.
- Backfill and ladder reconstruct their requested perspectives before publishing
  game/events and perspectives in one SQLite transaction. Failed perspective
  writes roll back the game replacement. This does not retroactively repair old
  partial records or make every ingestion path atomic.

Official sources checked: [Riot patch notes](https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-18-notes/),
[Data Dragon versions](https://ddragon.leagueoflegends.com/api/versions.json), and
[Riot's Data Dragon guidance](https://support-developer.riotgames.com/hc/en-us/articles/22698698001939-League-of-Legends).

## Repair evidence and next gate

`data/ml/repair_source_audit.json` records a read-only inventory while collection
was active: 20,857 stored games, all 16.17; 11,207 had both raw files present and
9,650 lacked a pair. These are file-presence counts, not full integrity results
or final missing counts. The running collector can change coverage.

All 12 deterministically sampled covered games replayed from local files with
matching IDs/patches. Visit counts, purchase labels and gold estimates were
unchanged in those samples; several decision-context fields changed. Measured
reconstruction-only time excludes file I/O, database writes and per-player
reconstruction. Neither total repair wall time nor corpus-wide metric impact is
measured. The audit made no Riot requests and no database writes.

After the active collector and serial catch-up finish:

1. Freeze and record the source inventory. Validate every selected raw pair,
   including payload IDs, DTO patch, full lobby, and readable timeline.
2. Build a separate staging database from verified local sources with per-game
   provenance and a repair checkpoint distinct from `refresh_done.txt`. Quarantine
   missing/corrupt sources with reason codes. Do not silently omit them.
3. Validate staging integrity, lobby membership, patch/static/reconstruction
   consistency and the changed-context population before replacing any corpus.
4. Publish an explicit single-patch export as one verified generation. Existing
   export file publication is not yet a transactional generation mechanism; keep
   the hold until staging/publication and interruption checks are implemented.
5. Start a new validation-only experiment only after that gate. Keep the final
   test set sealed and promotion separate.

Do not launch another broad Riot crawl this cycle: local sources can repair a
substantial part without API time. The strongest counterargument is selection
bias from using only cached games. A clean subset must therefore be explicitly
described and its exclusions measured; it cannot stand in for the full corpus.
Reassess targeted missing-source retrieval after the existing collector finishes.

## Verification

The following assert scripts passed on CPU after the implementation: patch
integrity (including rollback in a temporary database), decision causality,
pipeline hold, shop economy, visits, causal gold, audit, live prediction, live
API, live sessions, artifact provenance, manifest digest, and probe isolation.
This validates the exercised code paths; it does not certify the existing corpus
or the legacy served model. No training or final-test evaluation was performed.
