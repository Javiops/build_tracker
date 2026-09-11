"""Sanity-check the causal gold estimate (reconstruct.gold_est).

Samples stored games, re-fetches match+timeline, reconstructs, and compares two
budget candidates per shop visit:

  stale : the 60s-stale frame gold (gold_left)
  est   : the causal pre-decision estimate (gold_est, prequential-v2)

**Read the outputs carefully — this script has no ground truth.** The reference
it prints errors against is `leftover + the visit's own net spend`, which is
label-derived AND inflated (it adds back the spend without subtracting the gap's
income, ~+630g on average). So a large "|err|" is not evidence that gold_est is
wrong, and a small one is not evidence that it is right.

The two outputs that do mean something:

  * the wallet identity residual, which validates SPEND PRICING exactly and
    says nothing about the income terms;
  * afford coverage — whatever the true arrival gold was, it covered the
    visit's actual spend, so a usable budget estimate must too.

Validating the income forecast itself needs live telemetry (exact gold at a
known timestamp), which Match-V5 cannot provide.

    .venv\\Scripts\\python.exe scripts\\validate_gold_est.py [--games 250] [--seed 16]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import db
from app.ddragon import DataDragon
from app.ladder import routing_for_match_id
from app.reconstruct import reconstruct_game
from app.riot import RiotClient, RiotError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=250)
    parser.add_argument("--seed", type=int, default=16)
    return parser.parse_args()


def quantiles(values: list[float]) -> str:
    if not values:
        return "n/a"
    v = sorted(values)
    n = len(v)
    q = lambda p: v[min(n - 1, int(n * p))]
    return f"p50 {q(0.5):.0f}  p90 {q(0.9):.0f}  p99 {q(0.99):.0f}"


def main() -> None:
    args = parse_args()
    with db() as conn:
        ids = [r["match_id"] for r in conn.execute("SELECT match_id FROM games")]
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    sample = ids[: args.games]
    print(f"validating on {len(sample)} of {len(ids)} stored games", flush=True)

    client = RiotClient()
    dragon = DataDragon()
    rows: list[tuple[int, int, int, bool]] = []  # (truth, stale, est, is_save)
    identity_err: list[float] = []  # wallet identity residual per frame window (label-free, exact)
    fetched = 0
    try:
        for k, match_id in enumerate(sample, start=1):
            regional, _platform = routing_for_match_id(match_id)
            try:
                match = client.match(regional, match_id)
                timeline = client.timeline(regional, match_id) if match else None
            except RiotError as exc:
                if exc.status in (401, 403):
                    raise
                continue
            if not match or not timeline:
                continue
            fetched += 1
            debug: dict = {}
            _game, events = reconstruct_game(match, timeline, dragon, gold_debug=debug)
            # Wallet identity: f1_cur must equal f0_cur + (f1_tot - f0_tot) -
            # spends_in_window. totalGold is cumulative income (spending never
            # touches it), so any residual here is spend-pricing error —
            # measured exactly, no labels involved. It validates ONLY the spend
            # terms: it uses both frames, so it cannot say whether the income
            # split between them is right, and it is not evidence that gold_est
            # is accurate.
            for pid, series in (debug.get("frame_series") or {}).items():
                spends = [
                    (ts, a) for ts, a, kind in (debug.get("gold_events") or {}).get(pid, [])
                    if kind == "spend"
                ]
                for (t0, c0, g0), (t1, c1, g1) in zip(series, series[1:]):
                    window_spend = sum(a for ts, a in spends if t0 < ts <= t1)
                    identity_err.append(c1 - (c0 + (g1 - g0) - window_spend))
            for ev in events:
                if ev.get("type") != "shop" or ev.get("gold_est") is None:
                    continue
                gap = ev.get("gold_gap") or {}
                spend = int(ev.get("gold") or 0) - int(ev.get("gold_left") or 0)  # = net spend
                rows.append(
                    (
                        int(ev.get("gold") or 0),       # "truth": leftover + net spend
                        int(ev.get("gold_left") or 0),  # stale frame gold
                        int(ev.get("gold_est") or 0),   # forward walk
                        bool(ev.get("is_save")),
                        gap.get("residual"),
                        spend,
                    )
                )
            if k % 25 == 0:
                print(f"  …{k}/{len(sample)} games, {len(rows)} visits", flush=True)
    finally:
        client.close()

    print(f"\n{fetched} games fetched, {len(rows)} visits with estimates", flush=True)
    if identity_err:
        abs_id = [abs(x) for x in identity_err]
        bias_id = sum(identity_err) / len(identity_err)
        exact_ok = sum(1 for x in abs_id if x <= 5) / len(abs_id)
        print(
            f"wallet identity residual (n={len(identity_err)} windows): "
            f"{quantiles(abs_id)}   bias {bias_id:+.0f}   exact(<=5g) {exact_ok:.1%}",
            flush=True,
        )
    for label, subset in (
        ("all visits", rows),
        ("buy visits", [r for r in rows if not r[3]]),
        ("save visits", [r for r in rows if r[3]]),
        # residual==0 windows: no farm/assist/objective income to mistime, so
        # est IS the true arrival (identity-exact walk) — the delta against
        # the F+spend "truth" on this subset measures the truth's own error.
        ("zero-residual buys", [r for r in rows if not r[3] and r[4] is not None and r[4] <= 50]),
    ):
        if not subset:
            continue
        err_stale = [abs(s - t) for t, s, _e, _sv, _r, _sp in subset]
        err_est = [abs(e - t) for t, _s, e, _sv, _r, _sp in subset]
        bias_est = sum(e - t for t, _s, e, _sv, _r, _sp in subset) / len(subset)
        within = sum(1 for t, _s, e, _sv, _r, _sp in subset if abs(e - t) <= 150) / len(subset)
        print(f"\n{label} (n={len(subset)})", flush=True)
        print(f"  stale |err|  {quantiles(err_stale)}", flush=True)
        print(f"  est   |err|  {quantiles(err_est)}   bias(est-truth) {bias_est:+.0f}   within150g {within:.1%}", flush=True)

    # Affordability coverage: whatever the true arrival was, it afforded the
    # visit's net spend. A good budget estimate must cover the spend (+ modest
    # slack for in-visit income) on nearly every visit.
    buys = [r for r in rows if not r[3] and r[5] > 0]
    if buys:
        print(f"\nafford coverage on {len(buys)} spending visits (budget >= net spend):", flush=True)
        for name, budget_fn in (
            ("stale + 500 (today's mask)", lambda r: r[1] + 500),
            ("est + 0", lambda r: r[2]),
            ("est + 150", lambda r: r[2] + 150),
            ("est + 300", lambda r: r[2] + 300),
        ):
            cover = sum(1 for r in buys if budget_fn(r) >= r[5]) / len(buys)
            print(f"  {name:<28} {cover:.1%}", flush=True)


if __name__ == "__main__":
    main()
