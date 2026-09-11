---
description: Bind a validated candidate as the served artifact
---

Load the `promote-model` skill and follow it exactly.

This command does not authorize promotion — the owner does, for a specific
candidate, after reviewing its displayed-policy validation report. If that
authorization is not in the conversation, stop and ask.

Reject legacy or `probe_only` artifacts, mismatched manifest/export/config
provenance, test-set selection, empty reports, nonzero over-budget
recommendations, and unresolved data-validity findings.

Use the full response shape:

FINDING → EVIDENCE (with provenance) → WHAT WOULD CHANGE MY MIND → COST →
NOT DOING THIS CYCLE → RISK → NEXT
