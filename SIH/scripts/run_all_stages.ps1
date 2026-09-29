param(
    [Parameter(Mandatory=$true)]
    [string]$Video
)

$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Video)) {
    Write-Error "Video not found: $Video"
    exit 1
}

Write-Host "Running pipeline for: $Video" -ForegroundColor Cyan

& $py main.py preprocess-video --video $Video
& $py main.py extract-metadata --video $Video
& $py main.py reconstruct-sparse --video $Video
& $py main.py reconstruct-dense --video $Video
& $py main.py track-vehicles --video $Video
& $py main.py mask-reconstruct --video $Video
& $py main.py localize-3d --video $Video
& $py main.py manage-assets
& $py main.py match-assets
& $py main.py fuse-scene
& $py main.py texture-clean-mesh --quality all

Write-Host "Pipeline complete." -ForegroundColor Green
