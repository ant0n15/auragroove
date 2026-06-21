# Builds a clean, portable copy of this project ready to zip / move to another PC.
# Excludes the non-portable venv and generated/cache folders. Keeps the bundled
# models (checkpoints), the engine source, uv.exe, and the launcher/installer.
$ErrorActionPreference = "Stop"
$src = $PSScriptRoot
$dst = Join-Path (Split-Path $src -Parent) "auragroove_release"

Write-Host "Staging a clean copy to:" -ForegroundColor Cyan
Write-Host "  $dst"
Write-Host "Excluding: acestep_engine\.venv, __pycache__, auragroove_outputs, .git"
Write-Host "(The 11.6 GB model weights ARE included.)"
Write-Host ""

$venv = Join-Path $src "acestep_engine\.venv"
$outs = Join-Path $src "auragroove_outputs"
$git  = Join-Path $src ".git"

# robocopy returns 0-7 on success; treat >=8 as a real error.
robocopy $src $dst /E /XD $venv $outs $git "__pycache__" /XF "*.pyc" `
    /MT:16 /NFL /NDL /NJH /NP /R:1 /W:1 | Out-Null
if ($LASTEXITCODE -ge 8) { Write-Error "robocopy failed (code $LASTEXITCODE)"; exit 1 }

$gb = [math]::Round((Get-ChildItem $dst -Recurse -File | Measure-Object Length -Sum).Sum/1GB, 2)
Write-Host ""
Write-Host "Done. Clean copy is ~$gb GB at:" -ForegroundColor Green
Write-Host "  $dst"
Write-Host ""
Write-Host "Next:"
Write-Host "  1. Zip that folder (right-click -> Send to -> Compressed folder, or 7-Zip),"
Write-Host "     OR just copy the folder to the other PC."
Write-Host "  2. On the other PC: run install.bat (or just run_auragroove.bat - it auto-installs)."
