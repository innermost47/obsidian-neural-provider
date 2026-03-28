@echo off
setlocal enabledelayedexpansion

:: ================================================================
::  OBSIDIAN Neural Provider — Install Script (Windows)
:: ================================================================

echo.
echo ==================================================
echo   OBSIDIAN Neural Provider -- Installation
echo ==================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYTHON_VERSION!") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if !PYTHON_MAJOR! LSS 3 (
    echo [ERROR] Python !PYTHON_VERSION! detected. Python 3.10+ is required.
    pause
    exit /b 1
)
if !PYTHON_MAJOR! EQU 3 if !PYTHON_MINOR! LSS 10 (
    echo [ERROR] Python !PYTHON_VERSION! detected. Python 3.10+ is required.
    pause
    exit /b 1
)

echo [OK] Python !PYTHON_VERSION! detected

set HAS_CUDA=false
set CUDA_VERSION=
set GPU_NAME=

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set HAS_CUDA=true
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do (
        set GPU_NAME=%%g
        goto :gpu_found
    )
    :gpu_found
    echo [OK] NVIDIA GPU detected: !GPU_NAME!

    for /f "tokens=*" %%c in ('nvidia-smi ^| findstr "CUDA Version"') do (
        for /f "tokens=3" %%v in ("%%c") do set CUDA_VERSION=%%v
    )
    if not "!CUDA_VERSION!"=="" (
        echo     CUDA Version: !CUDA_VERSION!
    )
) else (
    echo [WARN] No NVIDIA GPU detected -- will run on CPU (slow)
)

echo.
echo [..] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
echo [OK] Virtual environment created

echo.
echo [..] Installing PyTorch...

if "!HAS_CUDA!"=="true" (
    set CUDA_MAJOR=0
    if not "!CUDA_VERSION!"=="" (
        for /f "tokens=1 delims=." %%m in ("!CUDA_VERSION!") do set CUDA_MAJOR=%%m
    )

    if !CUDA_MAJOR! GEQ 12 (
        echo     CUDA 12.x detected, installing cu121...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    ) else (
        echo     CUDA 11.x detected, installing cu118...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
    )
) else (
    echo     No GPU -- installing CPU version...
    pip install torch torchvision torchaudio --quiet
)

if errorlevel 1 (
    echo [ERROR] PyTorch installation failed
    pause
    exit /b 1
)
echo [OK] PyTorch installed

echo.
echo [..] Installing provider dependencies...
pip install -r requirements_provider.txt --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation failed
    pause
    exit /b 1
)
echo [OK] Dependencies installed

echo.
python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'[OK] CUDA available: {name}')
    print(f'     VRAM: {vram:.1f} GB')
    if vram < 4:
        print('[WARN] Less than 4GB VRAM -- even small model may not fit')
    elif vram < 8:
        print('[INFO] 4-8GB VRAM -- use --model stable-audio-open-small')
    else:
        print('[INFO] 8GB+ VRAM -- both models supported')
else:
    print('[WARN] No GPU acceleration -- CPU only (very slow)')
"

echo.
echo ==================================================
echo   Installation complete!
echo ==================================================
echo.
echo   Start the provider server:
echo.
echo   venv\Scripts\activate.bat
echo.
echo   Full model (RTX 3070+, 8GB VRAM):
echo   python provider.py --key YOUR_API_KEY
echo.
echo   Small model (RTX 3060+, 4GB VRAM):
echo   python provider.py --key YOUR_API_KEY --model stable-audio-open-small
echo.
echo   Your API key is provided by the OBSIDIAN Neural admin.
echo ==================================================
echo.
pause