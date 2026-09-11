param([string]$TaskName='BuildTracker 16.18 Qualified Smoke')
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = '1'
try {
    & (Join-Path $root '.venv/Scripts/python.exe') scripts/run_qualified_smoke.py --coverage data/acceptance/16.18_repaired_v2/inventory_recheck/coverage --name 16.18-initial-smoke-v2
    $smokeExit = $LASTEXITCODE
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $smokeExit
