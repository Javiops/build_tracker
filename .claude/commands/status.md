---
description: Report what is actually running in build_tracker right now
---

Run `scripts/status.py` and summarise it in a few lines: what is running, what
has uncommitted changes, which artifact is served, and whether anything looks
stuck.

Do not start or stop anything. If a job appears stalled, say so and stop —
racing or killing a running Riot sequence needs the owner's explicit go-ahead.
