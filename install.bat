@echo off
setlocal enabledelayedexpansion

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

nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] No NVIDIA GPU detected.
    echo         CPU mode is not allowed in the OBSIDIAN Neural provider network.
    echo         Minimum requirement: NVIDIA RTX 3070 ^(8GB VRAM^)
    echo                           or NVIDIA RTX 3060 ^(4GB VRAM^) for the small model.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do set GPU_NAME=%%g
set CUDA_VERSION=
for /f "tokens=9" %%v in ('nvidia-smi ^| findstr /C:"CUDA Version"') do set CUDA_VERSION=%%v

echo [OK] NVIDIA GPU detected: !GPU_NAME!
if not "!CUDA_VERSION!"=="" echo     CUDA Version: !CUDA_VERSION!

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
echo [..] Installing PyTorch with CUDA support...

set CUDA_MAJOR=11
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

if errorlevel 1 (
    echo [ERROR] PyTorch installation failed
    pause
    exit /b 1
)
echo [OK] PyTorch installed

echo.
echo [..] Installing provider dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation failed
    pause
    exit /b 1
)
echo [OK] Dependencies installed

echo.
echo [..] Checking CUDA availability...
(
echo import torch
echo if torch.cuda.is_available^(^):
echo     name = torch.cuda.get_device_name^(0^)
echo     vram = torch.cuda.get_device_properties^(0^).total_memory / 1024**3
echo     print^('[OK] CUDA available: ' + name^)
echo     print^('[OK] VRAM: ' + str^(round^(vram, 1^)^) + ' GB'^)
echo     if vram ^< 4:
echo         print^('[ERROR] Less than 4GB VRAM -- GPU not supported'^)
echo     elif vram ^< 8:
echo         print^('[INFO] 4-8GB VRAM -- set MODEL=stable-audio-open-small in .env'^)
echo     else:
echo         print^('[INFO] 8GB+ VRAM -- both models supported'^)
echo else:
echo     print^('[ERROR] CUDA not available -- check your drivers'^)
) > _check_gpu.py
python _check_gpu.py
del _check_gpu.py

echo.
echo ==================================================
echo   Installation complete!
echo ==================================================
echo.
echo   1. Copy .env.example to .env and fill in your API key:
echo      copy .env.example .env
echo.
echo   2. Start the provider server:
echo      venv\Scripts\activate.bat
echo      python provider.py
echo.
echo   Your API key is provided by the OBSIDIAN Neural admin.
echo ==================================================
echo.
pause