param([string]$Corpus='data/corpora/16.18_initial', [string]$Report='data/acceptance/16.18_initial/semantic_audit.json', [string]$TaskName='')
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$ErrorActionPreference = 'Stop'
$reportPath = Join-Path $root $Report
[System.IO.Directory]::CreateDirectory((Split-Path -Parent $reportPath)) | Out-Null
$log = [System.IO.Path]::ChangeExtension($reportPath, '.log')
$env:PYTHONUTF8 = '1'
$ErrorActionPreference = 'Continue'
& (Join-Path $root '.venv/Scripts/python.exe') (Join-Path $root 'scripts/audit_patch_corpus.py') --corpus $Corpus --out $Report 2>&1 | ForEach-Object {
    Add-Content -LiteralPath $log -Value ([string]$_) -Encoding utf8
}
$result = $LASTEXITCODE
if ($TaskName) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue }
exit $result
