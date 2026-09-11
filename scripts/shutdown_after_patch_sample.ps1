# One-shot watcher: power off only after a successful, structurally valid sample.
param(
    [string]$SampleTask = 'BuildTracker Patch 16.18 Sample',
    [string]$Report = 'C:\Users\Javier\build_tracker\data\patch_samples\16.18_20260911\report.json'
)
$ErrorActionPreference = 'Stop'
$log = Join-Path (Split-Path -Parent $Report) 'shutdown_watcher.log'
function Say([string]$message) {
    Add-Content -LiteralPath $log -Value ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $message) -Encoding utf8
}
try {
    Say "Waiting for $SampleTask."
    $deadline = (Get-Date).AddHours(48)
    do {
        $task = Get-ScheduledTask -TaskName $SampleTask -ErrorAction Stop
        if ($task.State -ne 'Running') { break }
        if ((Get-Date) -gt $deadline) { throw 'Timed out; shutdown cancelled' }
        Start-Sleep -Seconds 30
    } while ($true)

    $info = Get-ScheduledTaskInfo -TaskName $SampleTask -ErrorAction Stop
    if ($info.LastTaskResult -ne 0) {
        throw "Sample task failed with result $($info.LastTaskResult); shutdown cancelled"
    }
    if (!(Test-Path -LiteralPath $Report)) { throw 'Sample report missing; shutdown cancelled' }
    $result = Get-Content -LiteralPath $Report -Raw -Encoding utf8 | ConvertFrom-Json
    if ($result.structural_checks_passed -ne $true) {
        throw 'Sample structural validation did not pass; shutdown cancelled'
    }
    Say 'Sample completed and structural validation passed. Shutdown scheduled in 60 seconds.'
    & shutdown.exe /s /t 60 /c 'BuildTracker 16.18 sample completed and passed structural validation.'
    if ($LASTEXITCODE -ne 0) { throw "shutdown.exe failed with exit code $LASTEXITCODE" }
} catch {
    Say ("PC LEFT ON: " + $_.Exception.Message)
    exit 1
}
