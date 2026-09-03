# Build Tracker

An AI shop advisor for League of Legends. While you play, it answers one question in real time:

> **"Given that a KR Challenger — a really good player — visited the shop (including basing and buying nothing), what is the optimal action?"**

It shows **Optimal buys** (always affordable with your current gold, each with an arrow to the full item it builds toward), or a **SAVE** row when holding your gold is the Challenger play. Behind it: a transformer trained on ~1.3M shop decisions from Challenger/GM games (KR + EUW), retrained nightly.

![panel](https://ddragon.leagueoflegends.com/cdn/img/perk-images/StatMods/StatModsAdaptiveForceIcon.png)

## Try it while you play (no API key needed)

Requirements: Python 3.12+, ~3GB disk for PyTorch. Works alongside the live game — it only reads the [Live Client Data API](https://developer.riotgames.com/docs/lol#game-client-api) that the League client serves locally (the same one Blitz/Porofessor use; no memory reading, ToS-safe).

**Step 0 — install Python and Git** (skip anything you already have). Open PowerShell and run:

```powershell
winget install Python.Python.3.12
winget install Git.Git
```

then close and reopen PowerShell. (No winget? Get Python from [python.org/downloads](https://www.python.org/downloads/) — in the installer, tick **"Add python.exe to PATH"** — and Git from [git-scm.com](https://git-scm.com/download/win).)

**Step 1 — get the app:**

```powershell
git clone https://github.com/Javiops/build_tracker.git
cd build_tracker
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Step 2 —** download `prefix_model.pt` from the [latest release](https://github.com/Javiops/build_tracker/releases) and put it in the repo's `data\ml\` folder (create it if needed).

**Step 3 — run it:**

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app
```

Open **http://127.0.0.1:8000/live** (second monitor or alt-tab), start any game — including Practice Tool — and recommendations appear a few seconds after loading in. Note: custom 1v1s and Practice Tool report no positions and few players, so matchup-aware suggestions only shine in real games with full lobbies.

## Collecting data / training your own (API key required)

Put a Riot API key in `.env` (`RIOT_API_KEY=RGAPI-...`, see `.env.example`):

```powershell
.venv\Scripts\python.exe -m app.ingest --ladder --region both --daily  # pull Challenger/GM games
.venv\Scripts\python.exe scripts\baseline.py                           # export training data
.venv\Scripts\python.exe scripts\train_prefix.py                       # train (GPU recommended)
```

`scripts\daily_pull.py` chains pull → export → retrain; a scheduled task runs it nightly.

## How it works

Riot match timelines are replayed into **shop visits** (inventory before/after, gold, all ten players' builds, objectives). A small transformer over the 10-player board — the shopper as its own token, the lane opponent specially marked — predicts the visit's full basket under multi-label loss, with affordability masking from recipe-aware gold math. "Save" is a first-class action learned from respawn-with-gold-and-bought-nothing states. The decoder is budget-, slot-, and build-plan-constrained, with thresholds self-calibrated at train time.

Current honest metrics (match-held-out test, vs a champion-frequency baseline at 0.13/0.30): **top-1 0.58, top-3 0.83**; save decisions 0.39/0.75.
