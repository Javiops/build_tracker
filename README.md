# Build Tracker

An AI shop advisor for League of Legends. At an explicit shop decision, it answers one question:

> **"What did high-elo players historically buy in similar reconstructed shop states?"**

This is behavioural imitation, not counterfactual optimisation: the data records one player's action, not whether that action was best. The advisor only renders an option after the player starts a shop-decision session; it records the exact wallet and observed follow-up purchase locally for beta evaluation.

![panel](https://ddragon.leagueoflegends.com/cdn/img/perk-images/StatMods/StatModsAdaptiveForceIcon.png)

## Try it while you play (no API key needed)

Requirements: Python 3.12+, ~3GB disk for PyTorch. Works alongside the live game — it only reads the [Live Client Data API](https://developer.riotgames.com/docs/lol#game-client-api) that the League client serves locally (the same one Blitz/Porofessor use; no memory reading, ToS-safe).

**Step 0 — install Python and Git** (skip anything you already have). Open PowerShell and run:

```powershell
winget install Python.Python.3.12
winget install Git.Git
winget install Microsoft.VCRedist.2015+.x64
```

(The last one is required by PyTorch — without it the server crashes with `Error loading c10.dll` / `WinError 1114`. If winget hangs on it, install it directly from Microsoft instead: https://aka.ms/vs/17/release/vc_redist.x64.exe)

then close and reopen PowerShell. (No winget? Get Python from [python.org/downloads](https://www.python.org/downloads/) — in the installer, tick **"Add python.exe to PATH"** — and Git from [git-scm.com](https://git-scm.com/download/win).)

**Step 1 — get the app** (the `cd ~` first line matters: it makes sure the download lands in your user folder, so the next commands find it):

```powershell
cd ~
git clone https://github.com/Javiops/build_tracker.git
cd build_tracker
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Step 2 —** install a *promoted pair* from the beta release: both
`prefix_model.pt` and its matching `deployment_manifest.json` belong in
`data\ml\`. The app refuses an orphaned or legacy model on purpose: a served
artifact must be bound to a frozen-validation, displayed-policy report and an
exact-budget pass.

**Step 3 — run it:**

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app
```

Open **http://127.0.0.1:8000/live** (second monitor or alt-tab), start any game, then click **I’m deciding in shop** only when the shop is open and you are choosing. A purchase ends that 90-second session; a timeout records a no-buy. No recommendation is generated while you are playing normally. Note: custom 1v1s and Practice Tool report no positions and few players, so matchup-aware suggestions only shine in real games with full lobbies.

**In-game overlay**: run `start_overlay.bat` for a draggable always-on-top widget over the game itself. It asks the player to open an explicit shop-decision session, then shows a first line and legal alternatives. The hold card is experimental: historic no-buy labels are limited to post-death cases, while prospective sessions collect the missing manual no-buys. Requires League's display mode set to **Borderless** (or Windowed); exclusive Fullscreen cannot be drawn over.

## Collecting data / training your own (API key required)

Put a Riot API key in `.env` (`RIOT_API_KEY=RGAPI-...`, see `.env.example`):

```powershell
.venv\Scripts\python.exe -m app.ingest --ladder --region both --daily  # pull Challenger/GM games
.venv\Scripts\python.exe scripts\baseline.py                           # export training data
.venv\Scripts\python.exe scripts\train_prefix.py                       # train (GPU recommended)
```

`scripts\daily_pull.py` chains pull → export → retrain; a scheduled task runs it nightly.

## How it works

Riot match timelines are replayed into **shop visits** (inventory before/after, gold, all ten players' builds, objectives). A small transformer over the 10-player board — the shopper as its own token, the lane opponent specially marked — predicts the visit's full basket under multi-label loss, with affordability masking from recipe-aware gold math. Historic `SAVE` labels are specifically respawn-with-gold-and-bought-nothing states; they do not observe every manual recall no-buy. The decoder is budget- and slot-constrained. Plan arrows come from the directly supervised next-final head at the observed state, never a fabricated high-gold state.

Canonical-item top-k and champion-frequency comparisons are training diagnostics, not live-policy performance. Candidate selection uses a frozen validation split, the displayed-policy evaluator, a train-only conditional multiset baseline, and an explicit serving-promotion manifest. There is no validated online-performance or causal-outcome claim.
