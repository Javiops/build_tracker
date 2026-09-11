# One-shot sample queued behind the existing recovery, without interrupting it.
param([Parameter(Mandatory=$true)][string]$Sample, [string]$AcceptCodeSha='')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $Sample 'job.log'
function Say([string]$message) {
    Add-Content -LiteralPath $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $message) -Encoding utf8
}
try {
    Say 'Queued: waiting for recovery and daily collection to finish. No Riot requests while waiting.'
    $deadline = (Get-Date).AddHours(48)
    do {
        $active = @(Get-ScheduledTask -TaskName 'BuildTracker Causal Recovery','BuildTracker Daily Pull' | Where-Object State -eq 'Running')
        if ($active.Count -eq 0) { break }
        if ((Get-Date) -gt $deadline) { throw 'Timed out waiting for collection slot; sample not fetched' }
        Start-Sleep -Seconds 60
    } while ($true)
    # Allow the previous rate-limit window to expire before a new client starts.
    Start-Sleep -Seconds 125
    Say 'Starting frozen 16.18 sample: ladder plus tracked pros; isolated staging DB.'
    $priorPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $sampleArgs = @((Join-Path $root 'scripts\collect_patch_sample.py'), '--sample', $Sample)
    if ($AcceptCodeSha) { $sampleArgs += @('--accept-code-sha', $AcceptCodeSha) }
    & (Join-Path $root '.venv\Scripts\python.exe') @sampleArgs 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $priorPreference
    if ($code -ne 0) { throw "Sample exited $code; inspect checkpoint/log and resume after resolving the cause" }
    Say 'Sample collection and structural validation finished. Read report.json for completeness and exclusions. No export or training ran.'
} catch {
    Say ("SAMPLE BLOCKED: " + $_.Exception.Message)
    exit 1
}
