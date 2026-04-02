FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

ENV HF_HUB_ENABLE_HF_TRANSFER=1

ARG HF_TOKEN

RUN python3 -c "from diffusers import StableAudioPipeline; \
    StableAudioPipeline.from_pretrained('stabilityai/stable-audio-open-1.0', \
    cache_dir='/app/models', \
    use_auth_token='${HF_TOKEN}', \
    force_download=True)"

COPY provider.py .

ENV PROVIDER_API_KEY=""
ENV CENTRAL_SERVER_URL=""
ENV SERVER_TO_PROVIDER_KEY=""
ENV HOST="0.0.0.0"
ENV PORT=8000
ENV MODEL="stable-audio-open-1.0"
ENV HF_HOME="/app/models"

EXPOSE 8000

CMD ["python3", "provider.py"]