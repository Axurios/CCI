# 1. Error Handling: Stop the script if any command fails
$ErrorActionPreference = "Stop"

$env:EE_PROJECT="alexcloud-489214"
# 2. Install uv using standard pip install uv
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..." -ForegroundColor Cyan
    pip install uv
}
 
uv pip install numpy matplotlib scikit-learn earthengine-api opencv-python tdqm wandb
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
Write-Host "bulk packages installed..." -ForegroundColor Cyan

# # 4. Clone the geotessera repository
$moduleInstalled = python -c "import geotessera" 2>$null
$lastExit = $LASTEXITCODE
if ($lastExit -ne 0) {
    Write-Host "geotessera not found. Installing..." -ForegroundColor Cyan
    
    # Check if we need to clone it first
    if (-not (Test-Path "geotessera")) {
        git clone https://github.com/ucam-eo/geotessera
    }

    pushd geotessera
    uv pip install -e .
    popd
} else {
    Write-Host "geotessera is already installed." -ForegroundColor Green
}



# Check if Earth Engine credentials
Write-Host "Checking Earth Engine credentials..." -ForegroundColor Cyan

$pythonCode = @"
import ee
import os
try:
    project = os.getenv('EE_PROJECT', 'alexcloud-489214')
    ee.Initialize(project=project)
    print('SUCCESS')
except Exception as e:
    print(f'FAIL: {e}')
"@

$checkResult = uv run python -c "$pythonCode"
if ($checkResult -like "*SUCCESS*") {
    Write-Host "Earth Engine already authenticated." -ForegroundColor Green
} else {
    Write-Host "Authentication required. Opening browser..." -ForegroundColor Yellow
    uv run python -c "import ee; ee.Authenticate()"
}

# Building the dataset
Write-Host "Starting training..."  -ForegroundColor Cyan
$env:PYTHONPATH = "."
uv run python -m src.train
#uv run python -m src.dataset.build_dataset










# """
# import ee, os
# try:
#     # Try to initialize with your specific project
#     # project = os.getenv("EE_PROJECT")
#     project = "alexcloud-489214"
#     if project:
#         print("got project")
#         ee.Initialize(project=project)
#     else:
#         print("env var not set, using default project")
#         ee.Initialize(project="alexcloud-489214")  # fallback
#     print("SUCCESS")

# except Exception:
#     print("FAIL")
# """