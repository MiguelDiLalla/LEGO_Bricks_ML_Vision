# PowerShell Script to Reorganize LEGO_Bricks_ML_Vision Project

# Define base paths
$ProjectRoot = "$PSScriptRoot"  # Automatically gets the script location
$DockerTrain = "$ProjectRoot/docker/train"
$DockerUtils = "$ProjectRoot/docker/utils"
$Scripts = "$ProjectRoot/scripts"
$LegacyScripts = "$Scripts/Legacy_scripts"

# Ensure directories exist
$Directories = @(
    $DockerTrain, $DockerUtils, "$ProjectRoot/models", "$ProjectRoot/notebooks",
    "$ProjectRoot/presentation", "$ProjectRoot/results", "$ProjectRoot/tests",
    "$ProjectRoot/docs", "$ProjectRoot/data/annotations", "$ProjectRoot/data/processed",
    "$ProjectRoot/data/raw"
)
$Directories | ForEach-Object { if (!(Test-Path $_)) { New-Item -ItemType Directory -Path $_ -Force } }

# Move Training Docker Files
Move-Item "$Scripts/Dockerfile.train" "$DockerTrain/Dockerfile" -Force -ErrorAction SilentlyContinue
Move-Item "$Scripts/requirements-train.txt" "$DockerTrain/requirements-train.txt" -Force -ErrorAction SilentlyContinue
Move-Item "$Scripts/train_cli.py" "$DockerTrain/train_cli.py" -Force -ErrorAction SilentlyContinue
Move-Item "$Scripts/pipeline_train_docker.py" "$DockerTrain/pipeline_train_docker.py" -Force -ErrorAction SilentlyContinue

# Move Utility Docker Files
Move-Item "$Scripts/pipeline_utils.py" "$DockerUtils/pipeline_utils.py" -Force -ErrorAction SilentlyContinue
Move-Item "$Scripts/utils_cli.py" "$DockerUtils/utils_cli.py" -Force -ErrorAction SilentlyContinue
Move-Item "$Scripts/Dockerfile.utils" "$DockerUtils/Dockerfile" -Force -ErrorAction SilentlyContinue
Move-Item "$Scripts/requirements-utils.txt" "$DockerUtils/requirements-utils.txt" -Force -ErrorAction SilentlyContinue

# Move Legacy Scripts
Move-Item "$Scripts/pipeline_setup.py" "$LegacyScripts/pipeline_setup.py" -Force -ErrorAction SilentlyContinue
Move-Item "$Scripts/pipeline_train.py" "$LegacyScripts/pipeline_train.py" -Force -ErrorAction SilentlyContinue
Move-Item "$Scripts/pipeline.py" "$LegacyScripts/pipeline.py" -Force -ErrorAction SilentlyContinue
Move-Item "$Scripts/PyPl_publish.py" "$LegacyScripts/PyPl_publish.py" -Force -ErrorAction SilentlyContinue
Move-Item "$Scripts/pyproject.toml" "$LegacyScripts/pyproject.toml" -Force -ErrorAction SilentlyContinue
Move-Item "$Scripts/setup.py" "$LegacyScripts/setup.py" -Force -ErrorAction SilentlyContinue

# Create placeholder files if missing
$PlaceholderFiles = @(
    "$DockerUtils/Dockerfile", "$DockerUtils/requirements-utils.txt", "$DockerUtils/utils_cli.py"
)
$PlaceholderFiles | ForEach-Object { if (!(Test-Path $_)) { New-Item -ItemType File -Path $_ -Force } }

Write-Host "[INFO] Project reorganization completed successfully!" -ForegroundColor Green
