#!/bin/bash
# ================================================================
#  OBSIDIAN Neural Provider — Install Script (Linux / macOS)
# ================================================================

set -e

PYTHON_MIN="3.10"
VENV_DIR="venv"

echo ""
echo "=================================================="
echo "  OBSIDIAN Neural Provider — Installation"
echo "=================================================="
echo ""

if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo "❌ Python $PYTHON_VERSION detected. Python 3.10+ is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

HAS_CUDA=false
HAS_ROCM=false
HAS_METAL=false
CUDA_VERSION=""

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" 2>/dev/null || echo "")
    HAS_CUDA=true
    echo "✅ NVIDIA GPU detected: $GPU_NAME"
    if [ -n "$CUDA_VERSION" ]; then
        echo "   CUDA Version: $CUDA_VERSION"
    fi
elif command -v rocm-smi &>/dev/null; then
    HAS_ROCM=true
    echo "✅ AMD GPU (ROCm) detected"
elif [[ "$(uname)" == "Darwin" ]]; then
    CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
    if [[ "$CHIP" == *"Apple"* ]]; then
        HAS_METAL=true
        echo "✅ Apple Silicon detected: $CHIP"
    fi
fi

if [ "$HAS_CUDA" = false ] && [ "$HAS_ROCM" = false ] && [ "$HAS_METAL" = false ]; then
    echo "⚠️  No GPU detected — will run on CPU (slow)"
fi

echo ""
echo "📦 Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

echo ""
echo "🔥 Installing PyTorch..."

if [ "$HAS_CUDA" = true ]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)

    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "   → CUDA 12.x detected, installing cu121..."
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu121 --quiet
    else
        echo "   → CUDA 11.x detected, installing cu118..."
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu118 --quiet
    fi

elif [ "$HAS_ROCM" = true ]; then
    echo "   → ROCm detected, installing rocm6.0..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.0 --quiet

elif [ "$HAS_METAL" = true ]; then
    echo "   → Apple Silicon, installing standard PyTorch (MPS)..."
    pip install torch torchvision torchaudio --quiet

else
    echo "   → CPU only..."
    pip install torch torchvision torchaudio --quiet
fi

echo "✅ PyTorch installed"

echo ""
echo "📦 Installing provider dependencies..."
pip install -r requirements_provider.txt --quiet
echo "✅ Dependencies installed"

echo ""
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'   VRAM: {vram:.1f} GB')
    if vram < 4:
        print('⚠️  Less than 4GB VRAM — even small model may not fit')
    elif vram < 8:
        print('ℹ️  4-8GB VRAM — use --model stable-audio-open-small')
    else:
        print('ℹ️  8GB+ VRAM — both models supported')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ Apple MPS available')
else:
    print('⚠️  No GPU acceleration — CPU only (very slow)')
"

echo ""
echo "=================================================="
echo "  Installation complete!"
echo "=================================================="
echo ""
echo "  Start the provider server:"
echo ""
echo "  source venv/bin/activate"
echo ""
echo "  # Full model (RTX 3070+, 8GB VRAM):"
echo "  python provider.py --key YOUR_API_KEY"
echo ""
echo "  # Small model (RTX 3060+, 4GB VRAM):"
echo "  python provider.py --key YOUR_API_KEY --model stable-audio-open-small"
echo ""
echo "  Your API key is provided by the OBSIDIAN Neural admin."
echo "=================================================="
echo ""