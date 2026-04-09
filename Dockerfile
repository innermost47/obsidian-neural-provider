FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

ARG HF_TOKEN
RUN python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" && \
    python3 -c "from diffusers import StableAudioPipeline; StableAudioPipeline.from_pretrained('stabilityai/stable-audio-open-1.0')"

COPY provider.py .

CMD ["python3", "provider.py"]