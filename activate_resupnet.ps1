$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeRoot = "E:\ResUpNet"

$env:RESUPNET_RUNTIME_ROOT = $RuntimeRoot
$env:RESUPNET_CACHE_ROOT = Join-Path $RuntimeRoot "cache"
$env:RESUPNET_RUNS_ROOT = Join-Path $RuntimeRoot "runs"
$env:TMP = Join-Path $RuntimeRoot "tmp"
$env:TEMP = Join-Path $RuntimeRoot "tmp"
$env:TMPDIR = Join-Path $RuntimeRoot "tmp"
$env:PIP_CACHE_DIR = Join-Path $RuntimeRoot "cache\pip"
$env:MPLCONFIGDIR = Join-Path $RuntimeRoot "cache\matplotlib"
$env:JOBLIB_TEMP_FOLDER = Join-Path $RuntimeRoot "tmp\joblib"
$env:KAGGLE_CONFIG_DIR = Join-Path $RepoRoot "data\.kaggle"

New-Item -ItemType Directory -Force `
    $env:RESUPNET_CACHE_ROOT, `
    $env:RESUPNET_RUNS_ROOT, `
    $env:TMP, `
    $env:PIP_CACHE_DIR, `
    $env:MPLCONFIGDIR, `
    $env:JOBLIB_TEMP_FOLDER | Out-Null

& (Join-Path $RepoRoot ".venv\Scripts\Activate.ps1")

Write-Host "ResUpNet venv active"
Write-Host "Runtime root: $env:RESUPNET_RUNTIME_ROOT"
Write-Host "Pip cache:    $env:PIP_CACHE_DIR"
Write-Host "Temp:         $env:TEMP"
