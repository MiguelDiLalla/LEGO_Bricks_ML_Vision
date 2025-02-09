$executionPath = Get-Location

$folders = @(
    "$executionPath/utils",
    "$executionPath/configs",
    "$executionPath/cache",
    "$executionPath/models",
    "$executionPath/data",
    "$executionPath/logs"
)

$files = @(
    "$executionPath/cli.py",
    "$executionPath/train.py",
    "$executionPath/utils/data_utils.py",
    "$executionPath/utils/model_utils.py",
    "$executionPath/utils/augmentation.py",
    "$executionPath/configs/dataset.yaml",
    "$executionPath/configs/training.json",
    "$executionPath/README.md",
    "$executionPath/requirements.txt"
)

foreach ($folder in $folders) {
    if (!(Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
    }
}

foreach ($file in $files) {
    if (!(Test-Path $file)) {
        New-Item -ItemType File -Path $file | Out-Null
    }
}

Write-Output "Folder structure and placeholder files created successfully in $executionPath."
