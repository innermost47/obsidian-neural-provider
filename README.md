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

OBSIDIAN Neural is an open source VST3/AU plugin for real-time AI music generation directly in your DAW. This repository contains the **provider kit**: a FastAPI inference server you run on your machine to contribute your GPU to the distributed generation network.

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

Subscription revenue is redistributed **equally** among all active providers each month via Stripe Connect, after deduction of a 15% platform fee covering infrastructure costs (fal.ai, hosting, maintenance). This fee is published publicly each month.

---

## Requirements

| Component  | `stable-audio-open-1.0` | `stable-audio-open-small` |
| ---------- | ----------------------- | ------------------------- |
| NVIDIA GPU | RTX 3070+ (8 GB VRAM)   | RTX 3060+ (4 GB VRAM)     |
| RAM        | 16 GB                   | 8 GB                      |
| OS         | Windows / Linux / macOS | Windows / Linux / macOS   |
| Python     | 3.10+                   | 3.10+                     |
| CUDA       | 11.8+                   | 11.8+                     |

> Apple Silicon (M1/M2/M3) supported via Metal.

---

## Installation

### Linux / macOS

```bash
git clone https://github.com/innermost47/obsidian-neural-provider.git
cd obsidian-neural-provider
chmod +x install.sh
./install.sh
```

### Windows

```bat
git clone https://github.com/innermost47/obsidian-neural-provider.git
cd obsidian-neural-provider
install.bat
```

The scripts automatically detect your GPU and install the correct PyTorch version (CUDA 11.8, CUDA 12.1, ROCm, or CPU).

---

## Configuration

Copy `.env.example` to `.env` and fill it in:

```bash
cp .env.example .env
```

```env
PROVIDER_API_KEY=op_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CENTRAL_SERVER_URL=https://api.obsidian-neural.com
MODEL=stable-audio-open-1.0
HOST=0.0.0.0
PORT=8000
HEARTBEAT_INTERVAL=300
```

| Variable             | Description                                                      |
| -------------------- | ---------------------------------------------------------------- |
| `PROVIDER_API_KEY`   | Key provided by the OBSIDIAN Neural admin — **never share this** |
| `CENTRAL_SERVER_URL` | Central server URL                                               |
| `MODEL`              | `stable-audio-open-1.0` or `stable-audio-open-small`             |
| `HOST`               | Listening interface (0.0.0.0 = all)                              |
| `PORT`               | Server port (default: 8000)                                      |
| `HEARTBEAT_INTERVAL` | Heartbeat frequency in seconds (default: 300)                    |

---

## Starting the server

```bash
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate.bat  # Windows

python provider.py
```

CLI arguments override `.env` if needed:

```bash
python provider.py --key op_xxx --model stable-audio-open-small --port 8002 --server https://api.obsidian-neural.com
```

---

## Models

| Model                     | VRAM    | Quality | Speed (RTX 3070) | Download size |
| ------------------------- | ------- | ------- | ---------------- | ------------- |
| `stable-audio-open-1.0`   | ~7-8 GB | ⭐⭐⭐  | ~30-120s         | ~5 GB         |
| `stable-audio-open-small` | ~3-4 GB | ⭐⭐    | ~5-15s           | ~1.5 GB       |

First launch downloads the model from HuggingFace. Subsequent launches are instant (cached model).

---

## Endpoints

| Method | Route       | Auth | Description                      |
| ------ | ----------- | ---- | -------------------------------- |
| `GET`  | `/status`   | ✅   | Availability + model info + VRAM |
| `POST` | `/generate` | ✅   | Audio generation → WAV bytes     |
| `GET`  | `/health`   | ❌   | Server health                    |

Authentication via `X-API-Key` header.

---

## Heartbeat

The provider automatically sends a heartbeat to the central server every 5 minutes (configurable via `HEARTBEAT_INTERVAL`). This lets the central server know your machine is online, independently of random pings.

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

1. Uptime ≥ 60% of random pings that month
2. At least 1 real job processed during the month

Pings are sent at **random times** to prevent cron-based cheating. The platform fee and redistributed amounts are published publicly each month on the central server.

---

## Security

- Your server only receives the optimized prompt — never user personal data
- The central server validates every WAV via FFmpeg before sending it to the musician
- An invalid WAV results in an **immediate ban**
- Never share your `PROVIDER_API_KEY`

---

## Project structure

```
obsidian-neural-provider/
├── provider.py
├── requirements.txt
├── .env.example
├── install.sh
├── install.bat
└── README.md
```

---

## Public Statistics

The central server exposes the number of active subscribers in real time:

```
GET https://api.obsidian-neural.com/api/v1/public/stats
```

Response:

```json
{
  "paying_users": 42,
  "updated_at": "2026-03-28T14:32:00Z"
}
```

This endpoint is public and requires no authentication. It allows every provider to verify the platform's growth and estimate their monthly share.

Combined with `public/finances`, you get a complete and transparent view of the network's financial health:

| Source                     | Data                                  |
| -------------------------- | ------------------------------------- |
| `/api/v1/public/stats`     | Active subscribers count in real time |
| `/api/v1/public/finances`  | Monthly redistribution history        |
| `/api/v1/public/ownership` | Proof of generation ownership         |

---

## Joining the network

1. Check that you have the minimum required GPU
2. Open a **GitHub Discussion** on [innermost47/ai-dj](https://github.com/innermost47/ai-dj/discussions) with your GPU model and preferred inference model
3. The admin sends you your API key
4. Copy `.env.example` → `.env`, fill in your key, start the provider

The pool is limited to **10 providers** in phase 1.

---

## Contributing

The code is open source — each provider has an equal voice in project decisions. Submit improvements via Pull Request.

---

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE)

---

_Made with 🎵 in France — [obsidian-neural.com](https://obsidian-neural.com)_
