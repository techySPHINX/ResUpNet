# ============================================================================
# ResUpNet BraTS Virtual Environment Setup Script
# ============================================================================
# This script creates a clean virtual environment and installs all dependencies
# Run this script in PowerShell: .\setup_venv.ps1
# ============================================================================

Write-Host "`n" -NoNewline
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "  ResUpNet BraTS - Virtual Environment Setup" -ForegroundColor Yellow
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python version
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Green
$pythonVersion = python --version 2>&1
Write-Host "      Found: $pythonVersion" -ForegroundColor Gray

# Step 2: Remove old virtual environment if exists
if (Test-Path ".venv") {
    Write-Host "[2/6] Removing old virtual environment..." -ForegroundColor Green
    Remove-Item -Recurse -Force .venv
    Write-Host "      ✅ Old .venv removed" -ForegroundColor Gray
} else {
    Write-Host "[2/6] No existing virtual environment found" -ForegroundColor Green
}

# Step 3: Create new virtual environment
Write-Host "[3/6] Creating new virtual environment..." -ForegroundColor Green
python -m venv .venv
if ($LASTEXITCODE -eq 0) {
    Write-Host "      ✅ Virtual environment created at .venv\" -ForegroundColor Gray
} else {
    Write-Host "      ❌ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Step 4: Activate virtual environment
Write-Host "[4/6] Activating virtual environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1
Write-Host "      ✅ Virtual environment activated" -ForegroundColor Gray

# Step 5: Upgrade pip
Write-Host "[5/6] Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip --quiet
Write-Host "      ✅ pip upgraded" -ForegroundColor Gray

# Step 6: Install all requirements
Write-Host "[6/6] Installing project dependencies..." -ForegroundColor Green
Write-Host "      This may take 2-3 minutes..." -ForegroundColor Yellow

# Install core dependencies first
python -m pip install --upgrade numpy==1.24.3 --quiet
python -m pip install --upgrade tensorflow==2.20.0 --quiet

# Install from requirements_brats.txt
if (Test-Path "requirements_brats.txt") {
    python -m pip install -r requirements_brats.txt --quiet
    Write-Host "      ✅ Requirements from requirements_brats.txt installed" -ForegroundColor Gray
}

# Install Jupyter packages
python -m pip install jupyter ipykernel notebook jupyterlab --quiet
Write-Host "      ✅ Jupyter packages installed" -ForegroundColor Gray

# Register Jupyter kernel
Write-Host ""
Write-Host "[*] Registering Jupyter kernel..." -ForegroundColor Green
python -m ipykernel install --user --name=resupnet-brats --display-name="ResUpNet BraTS (venv)"
Write-Host "    ✅ Kernel 'ResUpNet BraTS (venv)' registered" -ForegroundColor Gray

# Verify installation
Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "  Verifying Installation" -ForegroundColor Yellow
Write-Host "="*80 -ForegroundColor Cyan

python -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__}')"
python -c "import numpy as np; print(f'✅ NumPy {np.__version__}')"
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__}')"
python -c "import ipykernel; print(f'✅ IPyKernel {ipykernel.__version__}')"

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "  Setup Complete! 🎉" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Close and reopen VS Code" -ForegroundColor White
Write-Host "  2. Open resunet_brats_medical_2.ipynb" -ForegroundColor White
Write-Host "  3. Click 'Select Kernel' (top-right)" -ForegroundColor White
Write-Host "  4. Choose: 'ResUpNet BraTS (venv)'" -ForegroundColor White
Write-Host "  5. Run cells 3→4→5→7→12→16→29→30→31→33→34→37" -ForegroundColor White
Write-Host ""
Write-Host "Training Configuration:" -ForegroundColor Yellow
Write-Host "  ✅ 30 epochs (optimized)" -ForegroundColor Gray
Write-Host "  ✅ Higher learning rate (3e-4)" -ForegroundColor Gray
Write-Host "  ✅ Adaptive callbacks" -ForegroundColor Gray
Write-Host "  ✅ Data augmentation enabled" -ForegroundColor Gray
Write-Host ""
Write-Host "Expected training time: ~45-60 minutes on CPU" -ForegroundColor Cyan
Write-Host ""
