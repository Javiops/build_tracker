param([string]$TaskName='BuildTracker 16.18 Rebuild V2')
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$ErrorActionPreference = 'Stop'
$folder = Join-Path $root 'data/acceptance/16.18_repaired_v2'
[System.IO.Directory]::CreateDirectory($folder) | Out-Null
$log = Join-Path $folder 'chain.log'
$env:PYTHONUTF8 = '1'
$python = Join-Path $root '.venv/Scripts/python.exe'
function Set-Stage([string]$stage, [string]$state, [int]$code=0) {
    $record = @{ stage=$stage; state=$state; exit_code=$code; time_utc=[DateTime]::UtcNow.ToString('o') }
    $temporary = Join-Path $folder 'chain_status.pending.json'
    $record | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination (Join-Path $folder 'chain_status.json') -Force
    Add-Content -LiteralPath $log -Value ($record | ConvertTo-Json -Compress) -Encoding UTF8
}
$result = 1
try {
    Set-Stage 'rebuild' 'running'
    $ErrorActionPreference = 'Continue'
    & $python scripts/rebuild_frozen_corpus.py --source data/corpora/16.18_initial --out data/corpora/16.18_repaired_v2 2>&1 |
        ForEach-Object { Add-Content -LiteralPath $log -Value ([string]$_) -Encoding UTF8 }
    $result = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($result -ne 0) { Set-Stage 'rebuild' 'failed' $result; exit $result }
    Set-Stage 'audit' 'running'
    $ErrorActionPreference = 'Continue'
    & $python scripts/audit_patch_corpus.py --corpus data/corpora/16.18_repaired_v2 --out data/acceptance/16.18_repaired_v2/semantic_audit.json 2>&1 |
        ForEach-Object { Add-Content -LiteralPath $log -Value ([string]$_) -Encoding UTF8 }
    $result = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($result -eq 0) { Set-Stage 'audit' 'complete' } else { Set-Stage 'audit' 'failed' $result }
} catch {
    Add-Content -LiteralPath $log -Value ([string]$_) -Encoding UTF8
    Set-Stage 'supervisor' 'failed' 1
    $result = 1
} finally {
    # All durable evidence lives in the log/status/report, not the scheduler.
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $result
