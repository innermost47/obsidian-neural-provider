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

### 1 — Benchmark your GPU

Before anything, verify your GPU is fast enough. The model is bundled in the image — no download required:

```bash
docker run --rm --gpus all \
  -e HF_HOME=/root/.cache/huggingface \
  innermost47/obsidian-neural-provider:latest \
  python benchmark.py
```

| Model                   | Max average time for 10s audio |
| ----------------------- | ------------------------------ |
| `stable-audio-open-1.0` | 60s                            |

If your GPU is eligible, proceed to the next step.

---

### 2 — Join the network

1. Send an email to **contact@obsidian-neural.com** with your GPU model and your public URL (see Network setup below)
2. The admin creates your provider account and sends you your **activation token** (`OBSIDIAN_TOKEN`)
3. The pool is limited to **10 providers** in phase 1

---

### 3 — Network setup

Your machine must be reachable from the internet over **HTTPS with WebSocket support**. This requires:

- A domain name or free DDNS hostname pointing to your public IP
- Port forwarding on your router (443 → your machine)
- nginx as a reverse proxy with a valid SSL certificate
- Ports 80 and 443 open on your firewall

#### 3a — Get a free domain (DDNS)

If you don't have a static IP or domain name, use [DuckDNS](https://www.duckdns.org):

1. Create a free account and register a subdomain (e.g. `myprovider.duckdns.org`)
2. Keep your IP up to date:

**Linux:**

```bash
mkdir -p ~/duckdns
cat > ~/duckdns/duck.sh << 'EOF'
echo url="https://www.duckdns.org/update?domains=YOUR_DOMAIN&token=YOUR_TOKEN&ip=" | curl -k -o ~/duckdns/duck.log -K -
EOF
chmod +x ~/duckdns/duck.sh
(crontab -l 2>/dev/null; echo "*/5 * * * * ~/duckdns/duck.sh >/dev/null 2>&1") | crontab -
```

**Windows:**
Download and install the [DuckDNS Windows client](https://www.duckdns.org/install.jsp?tab=windows).

#### 3b — Port forwarding

On your router admin panel (usually `192.168.1.1`):

- Forward **external port 443** → **your machine's local IP:443**
- Forward **external port 80** → **your machine's local IP:80** (required for Let's Encrypt)

Assign a **static local IP** to your machine in your router's DHCP settings.

#### 3c — Install nginx + SSL certificate

**Linux:**

```bash
sudo apt install nginx certbot python3-certbot-nginx -y
sudo certbot --nginx -d myprovider.duckdns.org
sudo nano /etc/nginx/sites-available/obsidian-provider
```

```nginx
server {
    listen 80;
    server_name myprovider.duckdns.org;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name myprovider.duckdns.org;

    ssl_certificate /etc/letsencrypt/live/myprovider.duckdns.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/myprovider.duckdns.org/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/obsidian-provider /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

**Windows:**

1. Download [nginx for Windows](https://nginx.org/en/docs/windows.html)
2. Download [win-acme](https://www.win-acme.com/) for Let's Encrypt certificates
3. Run win-acme to obtain a certificate for your domain
4. Edit `nginx/conf/nginx.conf`:

```nginx
events {}

http {
    server {
        listen 80;
        server_name myprovider.duckdns.org;
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl;
        server_name myprovider.duckdns.org;

        ssl_certificate C:/path/to/fullchain.pem;
        ssl_certificate_key C:/path/to/privkey.pem;

        location / {
            proxy_pass http://localhost:8000;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
    }
}
```

```bat
nginx.exe
```

#### 3d — Open firewall ports

**Linux:**

```bash
sudo ufw allow 80
sudo ufw allow 443
sudo ufw reload
```

**Windows:**

```powershell
netsh advfirewall firewall add rule name="OBSIDIAN HTTP" protocol=TCP dir=in localport=80 action=allow
netsh advfirewall firewall add rule name="OBSIDIAN HTTPS" protocol=TCP dir=in localport=443 action=allow
```

---

### 4 — Install Docker + NVIDIA Container Toolkit

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

---

### 5 — Run the provider

```bash
docker run -d \
  --name obsidian-provider \
  -e OBSIDIAN_TOKEN=your_activation_token \
  -e CENTRAL_SERVER_URL=https://api.obsidian-neural.com \
  --gpus all \
  -p 8000:8000 \
  -v obsidian_data:/data \
  --restart unless-stopped \
  innermost47/obsidian-neural-provider:latest
```

The container:

- Activates itself automatically with your token on first start
- Saves credentials locally for subsequent restarts — no re-activation needed
- Connects to the central server via WebSocket and starts accepting jobs

---

### 6 — Verify it's running

```bash
docker logs -f obsidian-provider
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
- All communication uses HTTPS + WSS — never plain HTTP
- The image is signed with Cosign — verify it before running (see above)

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
