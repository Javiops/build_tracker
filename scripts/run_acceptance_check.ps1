param([string]$TaskName='BuildTracker 16.18 Audit Recheck')
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$folder = Join-Path $root 'data/acceptance/16.18_repaired_v2/recheck'
[System.IO.Directory]::CreateDirectory($folder) | Out-Null
$log = Join-Path $folder 'chain.log'
$env:PYTHONUTF8 = '1'
$python = Join-Path $root '.venv/Scripts/python.exe'
$stages = @(
    @{ Name='audit'; Arguments=@('scripts/audit_patch_corpus.py','--corpus','data/corpora/16.18_repaired_v2','--out','data/acceptance/16.18_repaired_v2/recheck/semantic_audit.json') },
    @{ Name='qualification'; Arguments=@('scripts/qualify_patch_corpus.py','--corpus','data/corpora/16.18_repaired_v2','--audit','data/acceptance/16.18_repaired_v2/recheck/semantic_audit.json','--out','data/acceptance/16.18_repaired_v2/qualification.json') }
)
function Write-Status([string]$stage,[string]$state,[int]$exitCode=0) {
    $record = @{ stage=$stage; state=$state; exit_code=$exitCode; time_utc=[DateTime]::UtcNow.ToString('o') }
    $record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'chain_status.json') -Encoding UTF8
    Add-Content -LiteralPath $log -Value ($record | ConvertTo-Json -Compress) -Encoding UTF8
}
$result = 1
try {
    foreach ($stage in $stages) {
        Write-Status $stage.Name 'running'
        $stageArguments = $stage.Arguments
        $ErrorActionPreference = 'Continue'
        & $python @stageArguments 2>&1 | ForEach-Object { Add-Content -LiteralPath $log -Value ([string]$_) -Encoding UTF8 }
        $result = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($result -ne 0) { Write-Status $stage.Name 'failed' $result; break }
        Write-Status $stage.Name 'complete'
    }
} catch {
    Add-Content -LiteralPath $log -Value ([string]$_) -Encoding UTF8
    Write-Status 'supervisor' 'failed' 1
    $result = 1
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
exit $result
