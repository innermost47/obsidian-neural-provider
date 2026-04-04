# OBSIDIAN Neural Provider

### Related Repositories

| Repository                                                                                             | Description                                  |
| ------------------------------------------------------------------------------------------------------ | -------------------------------------------- |
| [obsidian-neural-central](https://github.com/innermost47/obsidian-neural-central)                      | Central inference server                     |
| **[obsidian-neural-provider](https://github.com/innermost47/obsidian-neural-provider)** ← you are here | Provider kit — run a GPU node on the network |
| [obsidian-neural-frontend](https://github.com/innermost47/obsidian-neural-frontend)                    | Storefront & dashboard                       |
| [obsidian-neural-controller](https://github.com/innermost47/obsidian-neural-controller)                | Mobile MIDI controller app                   |
| [ai-dj](https://github.com/innermost47/ai-dj)                                                          | VST3/AU plugin (client)                      |

---

## Overview

OBSIDIAN Neural is an open source VST3/AU plugin for real-time AI music generation directly in your DAW. This repository contains the **provider kit**: a containerized FastAPI inference server you run on your machine to contribute your GPU to the distributed generation network.

## How it works

```

Musician in their DAW
↓ types a prompt
OBSIDIAN Neural central server
↓ finds an available GPU in the pool
Your machine (provider)
↓ generates audio with Stable Audio
↓ returns validated WAV
Musician receives the sound in real time

```

Subscription revenue is redistributed **equally** among all eligible providers each month via Stripe Connect, after deduction of a 15% platform fee covering infrastructure costs (fal.ai, hosting, maintenance). This fee is published publicly each month.

---

## Requirements

| Component  | `stable-audio-open-1.0`              |
| ---------- | ------------------------------------ |
| NVIDIA GPU | RTX 3070+ (8 GB VRAM)                |
| RAM        | 16 GB                                |
| OS         | Windows / Linux                      |
| CUDA       | 11.8+                                |
| Docker     | 20.10+ with NVIDIA Container Toolkit |

---

## Quick start

### 1 — Install Docker + NVIDIA Container Toolkit

**Windows:**

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend
- NVIDIA drivers ≥ 525 (NVIDIA Container Toolkit is bundled with Docker Desktop)

**Linux:**

```bash
# Docker
curl -fsSL https://get.docker.com | sh

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2 — Join the network

1. Open a **GitHub Discussion** on [innermost47/ai-dj](https://github.com/innermost47/ai-dj/discussions) with your GPU model
2. The admin creates your provider account and sends you your **activation token** (`OBSIDIAN_TOKEN`)
3. The pool is limited to **10 providers** in phase 1

### 3 — Run the provider

```bash
docker run -d \
  --name obsidian-provider \
  -e OBSIDIAN_TOKEN=your_activation_token \
  -e CENTRAL_SERVER_URL=https:///ai-harmony.duckdns.org/obsidian \
  --gpus all \
  -p 8000:8000 \
  -v obsidian_data:/data \
  --restart unless-stopped \
  innermost47/obsidian-neural-provider:latest
```

That's it. The container:

- Activates itself automatically with your token on first start
- Downloads credentials and saves them locally for subsequent restarts
- Connects to the central server and starts accepting jobs

### 4 — Verify it's running

```bash
docker logs -f <container_id>
```

You should see:

```
🔑 Activating provider with token...
✅ Activated as: your-provider-name
🔌 Attempting to connect to the central registry...
✅ Connected to the central server (Active presence)
```

---

## Image verification

The image is signed with [Cosign](https://github.com/sigstore/cosign). Verify before running:

```bash
cosign verify --key cosign.pub innermost47/obsidian-neural-provider:latest
```

`cosign.pub` is available at the root of this repository.

---

## Models

| Model                   | VRAM    | Quality | Speed (RTX 3070) | Size  |
| ----------------------- | ------- | ------- | ---------------- | ----- |
| `stable-audio-open-1.0` | ~7-8 GB | ⭐⭐⭐  | ~10-15s          | ~5 GB |

The model is bundled in the Docker image — no download required at runtime.

---

## Heartbeat

The provider automatically sends a heartbeat to the central server every 5 minutes. This lets the central server know your machine is online, independently of random verification pings.

---

## Monthly redistribution

```
Monthly revenue
    - 15% platform fee (fal.ai + hosting + maintenance)
    = Distributable amount
        ÷ nb eligible providers
        = Equal share per provider
```

Example with 180€ and 6 providers:

```
180€ - 27€ = 153€ → 25.50€ per provider
```

**Eligibility:**

1. **Presence** — worked ≥ 8h on at least 80% of your active days that month, AND accumulated ≥ 80% of your total expected hours. Providers who joined mid-month are evaluated proportionally from their join date.
2. **Activity** — processed at least 1 real job during the month (not a fal.ai fallback)

Pings and canary tests are sent at **random times** to prevent any form of scheduled cheating. The platform fee and redistributed amounts are published publicly each month on the central server.

---

## Security

- Your server only receives the optimized prompt — never user personal data
- The central server validates every WAV via FFmpeg before sending it to the musician
- An invalid WAV results in an **immediate ban**
- Your credentials are stored locally in a Docker volume (`/data/credentials.json`) and never transmitted after activation
- The image is signed — verify it before running (see above)

---

## Benchmarking

Before joining the network, run the local benchmark to verify your GPU meets the minimum speed requirements:

```bash
docker run --rm --gpus all innermost47/obsidian-neural-provider:latest python benchmark.py
```

| Model                   | Max average time for 10s audio |
| ----------------------- | ------------------------------ |
| `stable-audio-open-1.0` | 60s                            |

---

## Building from source

The `Dockerfile` is provided for transparency and self-hosting:

```bash
git clone https://github.com/innermost47/obsidian-neural-provider.git
cd obsidian-neural-provider
docker build --build-arg HF_TOKEN=your_hf_token -t obsidian-neural-provider .
```

A HuggingFace account and acceptance of the [Stable Audio Open 1.0 license](https://huggingface.co/stabilityai/stable-audio-open-1.0) are required to build the image.

---

## Project structure

```
obsidian-neural-provider/
├── provider.py
├── benchmark.py
├── requirements.txt
├── Dockerfile
├── cosign.pub
├── .env.example
└── README.md
```

---

## Public Data Dashboard

All network data — active subscribers, monthly redistribution history, and proof-of-generation logs — is published live at:

**[obsidian-neural.com/public.html](https://obsidian-neural.com/public.html)**

No authentication required. No data is ever deleted.

---

## Contributing

The code is open source — each provider has an equal voice in project decisions. Submit improvements via Pull Request.

---

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE)

---

_Made with 🎵 in France — [obsidian-neural.com](https://obsidian-neural.com)_
