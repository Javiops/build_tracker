# One-shot, serial supervisor for the audit recovery.  Run via Task Scheduler:
# a long foreground process launched from an agent session is not durable here.
# It intentionally does not promote a model or read the final test split.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "data\causal_recovery.log"

function Say([string]$message) {
    $line = "[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] " + $message
    Write-Output $line
    Add-Content -LiteralPath $log -Value $line -Encoding utf8
}

function Invoke-Stage([string]$name, [string[]]$arguments) {
    Say "--- $name start ---"
    # Python/PyTorch warnings arrive on stderr. Do not mistake them for a
    # failed stage; the native process exit code remains authoritative.
    $priorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $py @arguments 2>&1 | Tee-Object -FilePath $log -Append
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $priorPreference
    }
    if ($exitCode -ne 0) {
        throw "$name failed with exit code $exitCode"
    }
    Say "--- $name complete ---"
}

function Last-LineNumber([string]$path, [string]$needle) {
    $match = Select-String -LiteralPath $path -Pattern $needle -SimpleMatch | Select-Object -Last 1
    if ($null -eq $match) { return 0 }
    return $match.LineNumber
}

try {
    $dailyLog = Join-Path $root "data\daily_pull.log"
    Say "supervisor started; waiting for the active daily exporter/trainer to finish"
    for ($i = 0; $i -lt 720; $i++) {  # up to 12 hours; then leave an explicit failure log
        $exportLine = Last-LineNumber $dailyLog "--- export start ---"
        $finishedLine = Last-LineNumber $dailyLog "=== daily pipeline finished"
        if ($exportLine -gt 0 -and $finishedLine -gt $exportLine) { break }
        Start-Sleep -Seconds 60
    }
    $exportLine = Last-LineNumber $dailyLog "--- export start ---"
    $finishedLine = Last-LineNumber $dailyLog "=== daily pipeline finished"
    if ($exportLine -eq 0 -or $finishedLine -le $exportLine) {
        throw "daily pipeline did not finish within the supervisor wait window"
    }

    # refresh_done.txt was deliberately archived before this supervisor ran:
    # its 15,142 entries were from untagged/leaky reconstruction. This first
    # pass therefore re-fetches every stored full-lobby game, not just a tail.
    Invoke-Stage "causal full refresh" @("-m", "app.ingest", "--backfill", "--refresh")
    Invoke-Stage "verify full refresh" @("scripts/verify_causal_refresh.py")

    # The active daily run already covered its rolling lookback window.  Do not
    # duplicate it with a fixed 40 h request after a long refresh: --hours 0
    # derives the gap from the start of the last completed daily run and adds
    # the pipeline's small safety overlap.  This stays serial with the refresh.
    Invoke-Stage "incremental catch-up pull (gap only)" @("scripts/daily_pull.py", "--hours", "0", "--no-train")
    Invoke-Stage "causal refresh of catch-up games" @("-m", "app.ingest", "--backfill", "--refresh")
    Invoke-Stage "verify final refresh coverage" @("scripts/verify_causal_refresh.py")

    Invoke-Stage "frozen temporal export" @("scripts/baseline.py", "--rebuild-split")

    $stamp = Get-Date -Format "yyyyMMdd_HHmm"
    $common = @{
        PREFIX_COSINE = "1"; PREFIX_DMODEL = "128"; PREFIX_LAYERS = "4"
        PREFIX_FF = "256"; PREFIX_HEADS = "8"; PREFIX_RUNES = "1"
        PREFIX_POSW_SCHED = "8:24"
    }
    foreach ($key in $common.Keys) { Set-Item -Path ("Env:" + $key) -Value $common[$key] }

    $candidates = @(
        @{ name = "stale"; goldest = "0"; goldx = "0" },
        @{ name = "interpolated"; goldest = "1"; goldx = "0" },
        @{ name = "prequential"; goldest = "1"; goldx = "1" }
    )
    foreach ($candidate in $candidates) {
        $name = $candidate.name
        $env:PREFIX_GOLDEST = $candidate.goldest
        $env:PREFIX_GOLDX = $candidate.goldx
        $env:PREFIX_EPOCHS = "16"; $env:PREFIX_SAVEW = "0"; $env:PREFIX_TARGETW = "0"
        $env:PREFIX_INIT_FROM = ""; $env:PREFIX_FREEZE = "0"
        $env:PREFIX_OUT = "prefix_trunk_${name}_${stamp}.pt"
        Invoke-Stage "train trunk $name" @("scripts/train_prefix.py")

        $env:PREFIX_EPOCHS = "6"; $env:PREFIX_SAVEW = "0.5"; $env:PREFIX_TARGETW = "0.25"
        $env:PREFIX_INIT_FROM = "prefix_trunk_${name}_${stamp}.pt"; $env:PREFIX_FREEZE = "1"
        $env:PREFIX_OUT = "prefix_candidate_${name}_${stamp}.pt"
        Invoke-Stage "graft candidate $name" @("scripts/train_prefix.py")
    }

    Invoke-Stage "policy eval prequential vs stale" @(
        "scripts/eval_policy.py", "--artifact", "prefix_candidate_prequential_${stamp}.pt",
        "--compare", "prefix_candidate_stale_${stamp}.pt", "--split", "val"
    )
    Invoke-Stage "policy eval prequential vs interpolated" @(
        "scripts/eval_policy.py", "--artifact", "prefix_candidate_prequential_${stamp}.pt",
        "--compare", "prefix_candidate_interpolated_${stamp}.pt", "--split", "val"
    )
    Invoke-Stage "conditional policy baseline" @("scripts/eval_conditional_policy.py", "--split", "val")
    Say "RECOVERY COMPLETE: candidates and validation reports are ready for a manual promotion decision"
} catch {
    Say ("RECOVERY BLOCKED: " + $_.Exception.Message)
    exit 1
}
