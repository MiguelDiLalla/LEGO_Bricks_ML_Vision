# PowerShell Script to Ensure Necessary .gitkeep Files Exist
# This script will create .gitkeep files in required empty directories

# List of directories that must exist with .gitkeep files
$directories = @(
    "cache/datasets",
    "cache/models",
    "logs",
    "results/TrainingSessions",
    "data/processed"
)

# Loop through each directory
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    
    $gitkeepPath = Join-Path $dir ".gitkeep"
    if (!(Test-Path $gitkeepPath)) {
        New-Item -ItemType File -Path $gitkeepPath -Force | Out-Null
    }
}

Write-Host "✅ All necessary .gitkeep files have been ensured."
