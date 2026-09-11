# 16.18 acceptance contract

The original sample is frozen at `data/corpora/16.18_initial`. Its first audit
failed. It is evidence, not an accepted training dataset. Repairs create a new
corpus from the frozen raw responses and never overwrite this snapshot.

## Review decisions before the rebuild

- Tower counts use `TOWER_BUILDING`; inhibitors are a separate objective.
- Purchase targets are retained net gains backed by an `ITEM_PURCHASED` in the
  visit. Inventory gains from undo or automatic grants are not purchase labels.
- End-of-game item slots must not determine an earlier inventory. Support is
  seeded from the recorded role. The deterministic Atlas/Compass progression is
  retained; Bounty is an explicitly marked tier proxy when the final choice is
  unobserved. A real purchase event can establish the exact choice. Model input
  maps observed final variants to the same proxy in both offline and live rows.
  This deliberately loses variant detail; inventing historical visibility is
  unacceptable. The untouched final DTO remains available as raw evidence.
- An undo with both item IDs zero makes the inventory unreconstructable from
  that event. Preserve the raw game and mark the entire game ineligible for
  training, including every player's board. Do not guess the reversed item from
  its price. This exclusion is recorded separately from collection failures.
- `ITEM_UNDO.goldGain` records the wallet delta. Use it for prior-event gold
  accounting when supplied, including negative sale reversals. The extrapolation
  algorithm remains `prequential-v2`; its corrected transaction inputs are bound
  by the new reconstruction version and exact code hashes.
- A participant frame at the purchase's exact timestamp can already contain
  that purchase. Gold uses a strictly earlier frame; same-time rewards are not
  presumed available. The exact live wallet remains a different input regime.
- Destruction of role quest 1201 establishes the mid-quest completion timestamp;
  the pinned zero-cost boot recipe supplies its deterministic upgrade. Quest
  1202 moves boots to the role slot without discarding ownership. Later purchases
  and ingredient consumption must still behave correctly. A missing Boots tag
  on 3172 does not remove it from the boot family: its direct recipe identifies it.
- `WARD_PLACED/CONTROL_WARD` consumes one carried ward unless a matching same-time
  item-destruction event already records that consumption. Final DTO ward slots
  show presence, not stack quantities, and `roleBoundItem` is a separate slot.
- Undoing a combine restores all of the components consumed by that transaction,
  once. An `afterId` naming one of them must not duplicate it. Source `goldGain`
  and inventory reversal have different jobs and are checked separately.
- A deterministic `specialRecipe` transformation remains visible as an owned
  item. Non-purchasable transformed items are context, not new purchase targets.
  Do not use one skip flag to erase them from both contexts and labels.

## Required evidence

Before publication: focused property tests for future-frame/final-DTO mutation,
undo and grant label semantics, duplicate buys, recursive combines, source
ambiguity exclusion, and offline/live input parity. The full audit must validate
source hashes, deterministic replay, all ten perspectives, labels, inventory
algebra, causal frame bindings, board membership, combat/objective prefixes,
and explicit training exclusions. Reproducibility alone is not correctness.

The final DTO is also an independent endpoint QA anchor. It may exclude an
unreconciled game, but never supplies items to an earlier snapshot. The audit
publishes a checked-ID set and explicit quarantine reasons. Final
`qualification.json` adds whole-game exclusions and independent inventory limits;
the export must bind both reports and use only the qualification's accepted IDs.
Unknown endpoint differences
remain unresolved, even when excluded. They are not declared to be API bugs.

Viego matches are excluded in their entirety because possession can replace the
inventory outside shop transactions. The mechanic is documented in Riot's
[champion description](https://www.leagueoflegends.com/en-us/champions/viego/);
the raw trace in `EUW1_7978999931`, participant 2, demonstrates the unsupported
inventory changes in this sample. An endpoint match would not establish correct
intermediate boards. This exclusion reduces coverage and can bias the retained
population; it must be visible in the coverage report, including its effect on
tracked-pro games.

Coverage is measured after these exclusions, including role, champion, region,
tracked-pro participation and temporal splits. A passed structural/semantic gate
does not establish enough data, model quality, or exact offline wallet accuracy.

Exports publish complete, content-hashed immutable generations and switch their
pointer last. A scoped pipeline smoke run may use an accepted generation, but
cannot clear the production hold, read final-test examples, or promote a model.
Further collection and the baseline/candidate comparison remain later gates.

## Quantity semantics and inventory units

Authoritative follow-up evidence is
`data/acceptance/16.18_repaired_v2/inventory_recheck/qualification.json` and its
`coverage/` directory. The inventory recheck includes every board before a
decision and the shopper's inventory both before and after the visit. Its
accepted 1,063 games have at most six modeled regular slots; the separate count
of modeled item **copies** reaches eight. Control Ward stacks and role slots
must not be confused with regular slots. Riot documents the support slot at
[patch 26.3](https://www.leagueoflegends.com/en-gb/news/game-updates/patch-26-3-notes/).

The raw five-copy case is Tryndamere in EUW1_7979347871, participant 1, buying
Long Sword at 267579, 267746, 267913, 268815 and 269015ms. His Doran's Bow plus
those five swords occupy six regular slots. Exact traces are preserved in
`data/acceptance/16.18_repaired_v2/coverage_extremes.json`. These are valid
quantities, not a reason to deduplicate or impose an arbitrary three-copy cap.

Qualified exports encode the purchase target as `label_counts`, mapping item
IDs to positive integer counts (`item-counts-v1`). The order of map keys carries
no meaning. A tensor adapter may expand counts for legacy helpers, but all copy
counts survive; an incompatible artifact must fail instead of clipping. The
single `label_id` is retained only for named canonical diagnostics. It is not
the supervised basket. The weighted presence head counts positive visits once
per item; the count head learns the item's multiplicity. Displayed-policy
exact/recall/precision metrics use multiset equality/intersection.

`scripts/test_purchase_multiset.py` checks permutation invariance, five-copy
preservation, overflow rejection, label stripping before inference and the
actual decoder's capacity to display five legal copies. The immutable qualified
smoke used conservative regular-slot decoding for role-slot inventories; its
result cannot certify the full shopping policy or justify promotion.

The subsequent slot-legality implementation shares transaction rules between
both Predictor paths and the conditional baseline. Ward stacks occupy one
regular slot, or the support role slot; bottom boots free a regular slot only
with strictly earlier observed quest completion. New events/export rows carry
`slot_state_version=role-slots-v1`, `bot_quest_complete` and occupancy for held
items excluded from model tokens. An unordered baseline basket must have a valid
purchase sequence. Earlier displayed purchases are protected from consumption
by later combines, so the priced basket retains all displayed quantities.
The policy evaluator replays each alternative independently and rejects invalid
capacity, stack, boot-family or price claims; actual wallet violations remain
explicit in the report. Tests exercise these rules through the actual decoders.

The documented [Live Client API](https://developer.riotgames.com/docs/lol#game-client-api_live-client-data-api)
and reviewed local captures do not establish how to observe bot-quest completion
or missing role-slot inventory. Live completion remains `None`; absent bottom
boots or support wards set `role_slot_inventory_unknown`, which suppresses
those ambiguous purchases. Skipped items observed in regular slots contribute
`unmodeled_regular_slots`. No owned item or completed quest is invented. This
limitation must accompany later offline-versus-live comparisons. Existing
frozen exports remain unchanged; new slot-state evidence requires a fresh
reconstruction and its normal audit/qualification/export gates.

## Next two nights

After the initial smoke completion report exists, collect new cumulative 16.18
samples on September 12 and 13 using the same initial ladder/pro cohort. Wait
for `BuildTracker Daily Pull` and any recovery/collector lease to finish. Each
sample uses a new folder, a new observation cutoff and freshly paginated match
lists; cached raw match/timeline pairs avoid duplicate downloads. The 16
unresolved pro accounts remain a declared, accepted limitation.

For a new sample only, use:

```powershell
.venv\Scripts\python.exe scripts/collect_patch_sample.py --sample data/patch_samples/16.18_20260912 --reuse-cohorts-from data/corpora/16.18_initial/state.json
```

Resume with the same `--sample` and omit `--reuse-cohorts-from`. Never overwrite
the completed initial sample or silently replace its frozen split. The collector
now replays once and derives all ten perspectives; its code binding includes
the inventory ledger, static classifier and patch configuration. It preserves
the original cohort's observation time rather than describing it as a freshly
observed current ladder.

For each completed sample, run `freeze_patch_corpus.py`, `audit_patch_corpus.py`,
`qualify_patch_corpus.py`, and `report_corpus_coverage.py` into new, dated
directories. A fresh freeze records all event schema versions and reconstruction
code hashes, so it can be audited directly. Frozen parent generations remain
available. Auth failures stop collection and require a fresh API key; do not
turn an incomplete discovery into an accepted generation.

Only after those evidence gates should the planned baseline/candidate comparison
be prepared. Verify the new slot-state export and decoding checks, declare the new benchmark's split
version and relationship to the initial smoke split, and keep its final test
sealed. Additional games are not silently appended to any existing training
file. Choose one fixed baseline and one fixed candidate, use the same frozen
validation games and displayed-policy metrics, and retain explicit pro coverage
and quarantine counts. No model promotion is authorized by this sequence.
