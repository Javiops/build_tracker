# Build the tester bundle: dist\BuildAdvisor\ + dist\BuildAdvisor-beta.zip
#
# Uses a separate build venv (.venv-beta) with CPU-only torch — the model is
# 4MB and scores in milliseconds on CPU, and CUDA torch would balloon the
# bundle by ~2GB. Testers need no Riot API key and no Python: the bundle talks
# only to the League client's local API and Data Dragon's public CDN.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_beta.ps1 [-Console]
#
# -Console keeps a console window on the exe (debug builds).

param([switch]$Console)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venv = Join-Path $root ".venv-beta"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Creating build venv (.venv-beta)..."
    py -3 -m venv $venv
    & $py -m pip install --upgrade pip
}

Write-Host "Installing build deps (CPU torch)..."
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install fastapi uvicorn httpx pywebview numpy python-dotenv pyinstaller

Write-Host "Freezing BuildAdvisor..."
$windowed = "--windowed"
if ($Console) { $windowed = "--console" }
& $py -m PyInstaller --noconfirm --clean --name BuildAdvisor $windowed `
    --distpath dist --workpath build --specpath build `
    --paths $root `
    --hidden-import app.main `
    --collect-submodules uvicorn `
    --collect-all webview `
    (Join-Path $root "app\advisor.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$dist = Join-Path $root "dist\BuildAdvisor"
$model = Join-Path $root "data\ml\prefix_model.pt"
$deployment = Join-Path $root "data\ml\deployment_manifest.json"
if (-not (Test-Path $model) -or -not (Test-Path $deployment)) {
    throw "Refusing to build a beta with an unpromoted model. Run validation policy evaluation and scripts\promote_served_model.py first."
}
$approved = Get-Content -Raw $deployment | ConvertFrom-Json
$actual = (Get-FileHash -Algorithm SHA256 $model).Hash.ToLowerInvariant()
if ($approved.artifact_sha256 -ne $actual) {
    throw "deployment_manifest.json does not bind the exact prefix_model.pt bytes being bundled."
}
Write-Host "Copying assets into $dist..."
Copy-Item -Recurse -Force (Join-Path $root "web") (Join-Path $dist "web")
New-Item -ItemType Directory -Force (Join-Path $dist "scripts") | Out-Null
Copy-Item -Force (Join-Path $root "scripts\train_prefix.py") (Join-Path $dist "scripts\train_prefix.py")
New-Item -ItemType Directory -Force (Join-Path $dist "data\ml") | Out-Null
Copy-Item -Force $model (Join-Path $dist "data\ml\prefix_model.pt")
Copy-Item -Force $deployment (Join-Path $dist "data\ml\deployment_manifest.json")
New-Item -ItemType Directory -Force (Join-Path $dist "data\cache") | Out-Null
Copy-Item -Force (Join-Path $root "data\cache\*.json") (Join-Path $dist "data\cache\")

$zip = Join-Path $root "dist\BuildAdvisor-beta.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $dist -DestinationPath $zip
Write-Host "Done: $zip"
