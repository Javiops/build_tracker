#!/usr/bin/env bash
# Idempotent Cloud Agent install: prepares a Python virtualenv with the
# project dependencies. Safe to run repeatedly.
set -euo pipefail

cd "$(dirname "$0")/.."

# Ubuntu ships the venv module in a separate package; install it once if missing.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-pip
fi

# Create the virtualenv once, then reuse it on later runs.
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# Warm the Data Dragon static-data cache (items/champions) so the app has data
# to render even before a Riot sync. Non-fatal if the network is unavailable.
python - <<'PY'
try:
    from app.ddragon import DataDragon

    DataDragon()
    print("Data Dragon cache warmed")
except Exception as exc:  # pragma: no cover - best effort
    print(f"Data Dragon warm skipped: {exc}")
PY

echo "cloud-install: done"
