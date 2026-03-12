# 1. Error Handling: Stop the script if any command fails
$ErrorActionPreference = "Stop"

# 2. Install uv using standard pip
pip install uv
 
uv pip install numpy matplotlib scikit-learn earthengine-api

# 4. Clone the repository
if (-not (Test-Path "geotessera")) {
    git clone https://github.com/ucam-eo/geotessera
}

pushd geotessera
uv pip install -e .
popd

# Check if Earth Engine credentials
Write-Host "Checking Earth Engine credentials..." -ForegroundColor Cyan

$pythonCode = @"
import ee
try:
    # Try to initialize with your specific project
    ee.Initialize(project='alexcloud-489214')
    print('SUCCESS')
except Exception:
    print('FAIL')
"@
# uv run python -c $pythonCode | Out-Null

$checkResult = uv run python -c $pythonCode
if ($checkResult -like "*SUCCESS*") {
    Write-Host "Earth Engine already authenticated." -ForegroundColor Green
} else {
    Write-Host "Authentication required. Opening browser..." -ForegroundColor Yellow
    uv run python -c "import ee; ee.Authenticate()"
}

# Building the dataset
Write-Host "Building the dataset..."  -ForegroundColor Cyan
uv run python build_dataset.py
