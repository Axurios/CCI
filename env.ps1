# This script temporarily adds Mamba to the PATH for the current terminal session.

# 1. Define Paths
$mambaBase = "C:\Users\Alexa\miniforge3"
$mambaBin = "$mambaBase\Library\bin"
$envName = "esa_env"
$pythonVersion = "3.11"
$envPath = "$mambaBase\envs\$envName"

# 2. Add to PATH for this session
if (Test-Path $mambaBin) {
    $env:PATH += ";$mambaBin"
    Write-Host "Mamba path linked." -ForegroundColor Cyan
} else {
    Write-Error "Mamba not found at $mambaBin. Please check your installation."
    return
}

# 3. Initialize Shell Hook
mamba shell hook -s powershell | Out-String | Invoke-Expression

# 4. Check if the PHYSICAL folder exists
if (Test-Path $envPath) {
    Write-Host "Environment '$envName' already exists at $envPath. Activating..." -ForegroundColor Green
} else {
    Write-Host "Environment '$envName' not found. Creating with Python $pythonVersion..." -ForegroundColor Yellow
    # This command creates the env AND the physical directory
    mamba create -n $envName python=$pythonVersion -y
}

# 5. Activate
mamba activate $envName

Write-Host "Active Environment: $envName (Python $pythonVersion)" -ForegroundColor Magenta
