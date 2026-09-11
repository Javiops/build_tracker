---
description: Run the assert-script suite and report what broke
---

Run `scripts/run_tests.py` from the repo root and report the result.

Two of these catch the expensive class of mistake, so call them out by name if
they fail:

- `test_gold_causal.py` — a feature that reads the future.
- `test_audit.py` — a harness scoring a model in a regime it was not trained
  in. Nothing crashes; the number is just wrong.

Some tests need `data/tracker.db` or a frozen export. If one fails for a
missing prerequisite rather than a real defect, say which, and do not report
the suite as passing.
