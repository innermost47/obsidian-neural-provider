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

OBSIDIAN Neural is an open source VST3/AU workstation for real-time AI music generation. This repository contains the **provider kit**: a containerized FastAPI inference server that now supports **8 specialized AI models** simultaneously. By running this kit, you contribute your GPU to the distributed network and earn a share of the platform's revenue.

## 🧠 The Multi-Model Engine

Each provider node is now a versatile workstation capable of switching between 8 specialized "brains" in real-time:

1.  **Stable Audio Open 1.0** — General purpose foundation for full-mix textures.
2.  **Foundation-1** — Surgical tag-based control for melodic and harmonic phrasing.
3.  **Audialab EDM Elements** — High-energy EDM leads, supersaws, and plucks.
4.  **RC Infinite Pianos** — High-fidelity grand and electric piano performances.
5.  **RC Vocal Textures** — Choral and operatic vocal chord progressions.
6.  **SAO Instrumental** — Modern indie, rock, and lofi stems.
7.  **StableBeaT** — Advanced trap drum machine and 808 grooves.
8.  **Gluten-V1** — Specialized loop engine for trap and wavy melodic motifs.

**Auto-Config Technology:** The provider script automatically extracts optimal parameters (`Steps`, `CFG Scale`, `Conditioning Duration`) directly from each model's internal `model_config.json` to guarantee the best possible audio quality.

---

## How it works

```
Musician in their DAW
↓ types a prompt or draws on the canvas
OBSIDIAN Neural central server
↓ finds an available Multi-Model GPU Provider
Your machine (provider)
├── LLM inference (Gemma 4 via Ollama)
│       ↓ optimizes the prompt / analyzes the drawing
│       ↓ returns structured JSON response
└── Audio generation (Dynamic Model Stack)
        ↓ Loads weights (.safetensors) for the requested model
        ↓ Generates audio using lab-tested settings (from model config)
        ↓ returns validated WAV
Musician receives the sound in real time
```

Subscription revenue is redistributed **equally** among all eligible providers each month via Stripe Connect, after deduction of a 15% platform fee covering infrastructure costs (fal.ai fallback, hosting, maintenance).

---

## Requirements

| Component  | Specification                                 |
| ---------- | --------------------------------------------- |
| NVIDIA GPU | RTX 3070+ (8 GB VRAM min, 12 GB+ recommended) |
| RAM        | 16 GB                                         |
| Storage    | ~40 GB (required for the full 8-model suite)  |
| OS         | Windows / Linux                               |
| CUDA       | 11.8+                                         |
| Docker     | 20.10+ with NVIDIA Container Toolkit          |

---

## What your provider runs

Each provider runs two inference stacks:

| Stack     | Model / Capabilities                                       |
| --------- | ---------------------------------------------------------- |
| **Audio** | **8 Specialized Models** (On-demand loading)               |
| **LLM**   | `gemma4:e2b` via Ollama for prompt optimization and vision |

**Jobs are mutually exclusive** — your provider processes one request at a time (LLM or Audio) to ensure maximum VRAM availability and stability.

---

## Quick start

### 1 — Benchmark your GPU

Verify your GPU can handle the high-quality multi-model generation:

```bash
docker run --rm --gpus all \
  -e HF_HOME=/root/.cache/huggingface \
  innermost47/obsidian-neural-provider:latest \
  python benchmark.py
```

### 2 — Join the network

1. Send an email to **contact@obsidian-neural.com** with your GPU model and public URL.
2. Once approved, the admin will send you your **activation token** (`OBSIDIAN_TOKEN`).

---

### 3 — Network setup

Your machine must be reachable over **HTTPS with WebSocket support**.

#### 3a — Get a free domain (DDNS)

If you don't have a static IP, use [DuckDNS](https://www.duckdns.org):

**Linux Setup:**

```bash
mkdir -p ~/duckdns
cat > ~/duckdns/duck.sh << 'EOF'
echo url="https://www.duckdns.org/update?domains=YOUR_DOMAIN&token=YOUR_TOKEN&ip=" | curl -k -o ~/duckdns/duck.log -K -
EOF
chmod +x ~/duckdns/duck.sh
(crontab -l 2>/dev/null; echo "*/5 * * * * ~/duckdns/duck.sh >/dev/null 2>&1") | crontab -
```

#### 3b — Port forwarding

On your router admin:

- Forward **port 443** → **your machine's local IP:443**
- Forward **port 80** → **your machine's local IP:80** (for SSL)

#### 3c — Install nginx + SSL

**Linux Example:**

```bash
sudo apt install nginx certbot python3-certbot-nginx -y
sudo certbot --nginx -d myprovider.duckdns.org
```

Edit your nginx config to enable WebSocket proxying to port 8000:

```nginx
location / {
    proxy_pass http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 300s;
}
```

---

### 4 — Run the provider

```bash
docker run -d \
  --name obsidian-provider \
  -e OBSIDIAN_TOKEN=your_activation_token \
  -e CENTRAL_SERVER_URL=https://central.server.url.com \
  --gpus all \
  -p 8000:8000 \
  -v obsidian_data:/data \
  --restart unless-stopped \
  innermost47/obsidian-neural-provider:latest
```

The container will:

- Activate itself and download weights for the **8 specialized models**.
- Start Ollama for real-time prompt analysis.
- Connect to the central registry via WebSocket.

---

## Security & verification

The central server ensures network integrity via:

**Audio proof-of-work** — periodic spectrogram comparisons against reference banks for all 8 models.
**LLM conversation echo** — verifies LLM responses haven't been tampered with.
**Canary tests** — random invalid requests to verify provider validation logic.

---

## Monthly redistribution

**Redistribution Formula:**
`Monthly revenue - 15% platform fee = Distributable amount ÷ nb eligible providers`

**Eligibility:**

1. **Presence** — Online ≥ 8h on at least 80% of active days.
2. **Activity** — Processed at least 1 real job (non-fallback) during the month.

---

## Building from source

```bash
git clone https://github.com/innermost47/obsidian-neural-provider.git
cd obsidian-neural-provider
docker build --build-arg HF_TOKEN=your_hf_token -t obsidian-neural-provider .
```

---

## Public Data Dashboard

Track active subscribers and monthly redistribution history at:
**[obsidian-neural.com/public.html](https://obsidian-neural.com/public.html)**

---

## License

GNU Affero General Public License v3.0

---

_Made with 🎵 for the future of sampling — [obsidian-neural.com](https://obsidian-neural.com)_
