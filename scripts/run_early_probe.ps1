# One-shot isolated causal diagnostic. It intentionally does not touch data/ml,
# read the final test split, make API requests, or promote an artifact.
param(
    [ValidateRange(100, 20000)]
    [int]$Games = 3000,
    [switch]$PrequentialOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$suffix = "causal_v2_newest_$Games"
$logName = if ($Games -eq 3000 -and -not $PrequentialOnly) { "early_causal_probe.log" } else { "causal_probe_${Games}.log" }
$log = Join-Path $root ("data\" + $logName)
$out = Join-Path $root ("data\ml\probes\" + $suffix)

function Say([string]$message) {
    $line = "[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] " + $message
    Write-Output $line
    Add-Content -LiteralPath $log -Value $line -Encoding utf8
}

function Invoke-Stage([string]$name, [string[]]$arguments) {
    Say "--- $name start ---"
    # Python/PyTorch legitimately writes warnings to stderr. PowerShell turns
    # redirected native stderr into ErrorRecords, so Stop would abort a healthy
    # process before we can inspect its real exit code.
    $priorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $py @arguments 2>&1 | Tee-Object -FilePath $log -Append
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $priorPreference
    }
    if ($exitCode -ne 0) { throw "$name failed with exit code $exitCode" }
    Say "--- $name complete ---"
}

try {
    Say "waiting for $Games newest-first causal checkpoint entries"
    for ($i = 0; $i -lt 480; $i++) {
        if ((Get-Content (Join-Path $root "data\refresh_done.txt")).Count -ge $Games) { break }
        Start-Sleep -Seconds 60
    }
    $count = (Get-Content (Join-Path $root "data\refresh_done.txt")).Count
    if ($count -lt $Games) { throw "causal refresh did not reach $Games entries within 8 hours" }
    $manifest = Join-Path $out "split_manifest.json"
    $train = Join-Path $out "visits_train.jsonl"
    $val = Join-Path $out "visits_val.jsonl"
    if (Test-Path $out) {
        if (!(Test-Path $manifest) -or !(Test-Path $train) -or !(Test-Path $val)) {
            throw "probe output is partial and cannot be resumed safely: $out"
        }
        Say "reusing completed isolated probe export"
    } else {
        Invoke-Stage "export isolated causal probe" @("scripts/export_clean_probe.py", "--games", "$Games", "--out-dir", $out)
    }
    $env:PREFIX_ML_DIR = $out
    $common = @{ PREFIX_COSINE = "1"; PREFIX_DMODEL = "128"; PREFIX_LAYERS = "4"; PREFIX_FF = "256"; PREFIX_HEADS = "8"; PREFIX_RUNES = "1"; PREFIX_POSW_SCHED = "8:24" }
    foreach ($key in $common.Keys) { Set-Item -Path ("Env:" + $key) -Value $common[$key] }

    $candidates = if ($PrequentialOnly) {
        @(@{ name = "prequential"; goldest = "1"; goldx = "1" })
    } else {
        @(
            @{ name = "stale"; goldest = "0"; goldx = "0" },
            @{ name = "interpolated"; goldest = "1"; goldx = "0" },
            @{ name = "prequential"; goldest = "1"; goldx = "1" }
        )
    }
    foreach ($candidate in $candidates) {
        $name = $candidate.name
        $existing = Join-Path $out "probe_candidate_${name}.pt"
        if (Test-Path $existing) {
            Say "reusing completed probe candidate $name"
            continue
        }
        $env:PREFIX_GOLDEST = $candidate.goldest; $env:PREFIX_GOLDX = $candidate.goldx
        $env:PREFIX_EPOCHS = "4"; $env:PREFIX_SAVEW = "0"; $env:PREFIX_TARGETW = "0"; $env:PREFIX_INIT_FROM = ""; $env:PREFIX_FREEZE = "0"
        $env:PREFIX_OUT = "probe_trunk_${name}.pt"
        Invoke-Stage "probe trunk $name" @("scripts/train_prefix.py")
        $env:PREFIX_EPOCHS = "2"; $env:PREFIX_SAVEW = "0.5"; $env:PREFIX_TARGETW = "0.25"; $env:PREFIX_INIT_FROM = "probe_trunk_${name}.pt"; $env:PREFIX_FREEZE = "1"
        $env:PREFIX_OUT = "probe_candidate_${name}.pt"
        Invoke-Stage "probe graft $name" @("scripts/train_prefix.py")
    }
    if ($PrequentialOnly) {
        Invoke-Stage "probe policy prequential" @("scripts/eval_policy.py", "--artifact", "probe_candidate_prequential.pt", "--split", "val", "--games", "0")
    } else {
        Invoke-Stage "probe policy prequential vs stale" @("scripts/eval_policy.py", "--artifact", "probe_candidate_prequential.pt", "--compare", "probe_candidate_stale.pt", "--split", "val", "--games", "0", "--bootstrap", "200")
        Invoke-Stage "probe policy prequential vs interpolation" @("scripts/eval_policy.py", "--artifact", "probe_candidate_prequential.pt", "--compare", "probe_candidate_interpolated.pt", "--split", "val", "--games", "0", "--bootstrap", "200")
    }
    Invoke-Stage "probe conditional policy baseline" @("scripts/eval_conditional_policy.py", "--split", "val")
    Say "PROBE COMPLETE: diagnostic-only artifacts; promotion is blocked by provenance"
} catch {
    Say ("PROBE BLOCKED: " + $_.Exception.Message)
    exit 1
}
