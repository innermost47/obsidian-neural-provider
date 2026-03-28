# OBSIDIAN Neural Provider

<p align="center">
  <img src="https://obsidian-neural.com/logo.png" alt="OBSIDIAN Neural" width="80"/>
</p>

<p align="center">
  <strong>GPU inference server for the OBSIDIAN Neural distributed network</strong><br/>
  Contribute your GPU. Generate music. Share the revenue.
</p>

<p align="center">
  <a href="https://github.com/innermost47/ai-dj"><img src="https://img.shields.io/badge/Main%20Project-OBSIDIAN%20Neural-b8605c" alt="Main Project"/></a>
  <a href="https://obsidian-neural.com"><img src="https://img.shields.io/badge/Website-obsidian--neural.com-1a1a1a" alt="Website"/></a>
  <img src="https://img.shields.io/badge/License-AGPL--3.0-b8605c" alt="License"/>
  <img src="https://img.shields.io/badge/Python-3.10+-1a1a1a" alt="Python"/>
</p>

---

> 🇫🇷 [Français](#français) · 🇬🇧 [English](#english)

---

## Français

### Qu'est-ce que c'est ?

OBSIDIAN Neural est un plugin VST3/AU open source qui génère de l'audio par IA en temps réel, directement dans votre DAW. Ce dépôt contient le **kit provider** : un serveur d'inférence FastAPI que vous faites tourner sur votre machine pour contribuer votre GPU au réseau de génération distribué.

### Comment ça marche ?

```
Musicien dans son DAW
        ↓ tape un prompt
Serveur central OBSIDIAN Neural
        ↓ cherche un GPU disponible dans le pool
Votre machine (provider)
        ↓ génère le son avec Stable Audio
        ↓ renvoie le WAV validé
Musicien reçoit le son en temps réel
```

Les revenus des abonnements sont redistribués **à parts égales** entre tous les providers actifs chaque mois via Stripe Connect, après déduction d'une commission plateforme de 15% couvrant les coûts d'infrastructure (fal.ai, hébergement, maintenance). Cette commission est publiée publiquement chaque mois.

---

### Prérequis

| Composant  | `stable-audio-open-1.0` | `stable-audio-open-small` |
| ---------- | ----------------------- | ------------------------- |
| GPU NVIDIA | RTX 3070+ (8 GB VRAM)   | RTX 3060+ (4 GB VRAM)     |
| RAM        | 16 GB                   | 8 GB                      |
| OS         | Windows / Linux / macOS | Windows / Linux / macOS   |
| Python     | 3.10+                   | 3.10+                     |
| CUDA       | 11.8+                   | 11.8+                     |

> Apple Silicon (M1/M2/M3) supporté via Metal.

---

### Installation

#### Linux / macOS

```bash
git clone https://github.com/innermost47/obsidian-neural-provider.git
cd obsidian-neural-provider
chmod +x install.sh
./install.sh
```

#### Windows

```bat
git clone https://github.com/innermost47/obsidian-neural-provider.git
cd obsidian-neural-provider
install.bat
```

Les scripts détectent automatiquement votre GPU et installent la bonne version de PyTorch (CUDA 11.8, CUDA 12.1, ROCm ou CPU).

---

### Configuration

Copiez `.env.example` en `.env` et remplissez-le :

```bash
cp .env.example .env
```

```env
PROVIDER_API_KEY=op_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CENTRAL_SERVER_URL=https://api.obsidian-neural.com
MODEL=stable-audio-open-1.0
HOST=0.0.0.0
PORT=8001
HEARTBEAT_INTERVAL=300
```

| Variable             | Description                                                      |
| -------------------- | ---------------------------------------------------------------- |
| `PROVIDER_API_KEY`   | Clé fournie par l'admin OBSIDIAN Neural — **ne jamais partager** |
| `CENTRAL_SERVER_URL` | URL du serveur central                                           |
| `MODEL`              | `stable-audio-open-1.0` ou `stable-audio-open-small`             |
| `HOST`               | Interface d'écoute (0.0.0.0 = toutes)                            |
| `PORT`               | Port du serveur (défaut : 8001)                                  |
| `HEARTBEAT_INTERVAL` | Fréquence des heartbeats en secondes (défaut : 300)              |

---

### Démarrage

```bash
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate.bat  # Windows

python provider.py
```

Les arguments CLI surchargent le `.env` si besoin :

```bash
python provider.py --key op_xxx --model stable-audio-open-small --port 8002 --server https://api.obsidian-neural.com
```

---

### Modèles

| Modèle                    | VRAM    | Qualité | Vitesse (RTX 3070) | Téléchargement |
| ------------------------- | ------- | ------- | ------------------ | -------------- |
| `stable-audio-open-1.0`   | ~7-8 GB | ⭐⭐⭐  | ~30-120s           | ~5 GB          |
| `stable-audio-open-small` | ~3-4 GB | ⭐⭐    | ~5-15s             | ~1.5 GB        |

Le premier démarrage télécharge le modèle depuis HuggingFace. Les démarrages suivants sont instantanés (modèle en cache).

---

### Endpoints

| Méthode | Route       | Auth | Description                         |
| ------- | ----------- | ---- | ----------------------------------- |
| `GET`   | `/status`   | ✅   | Disponibilité + infos modèle + VRAM |
| `POST`  | `/generate` | ✅   | Génération audio → WAV bytes        |
| `GET`   | `/health`   | ❌   | Santé du serveur                    |

L'authentification se fait via le header `X-API-Key`.

---

### Heartbeat

Le provider envoie automatiquement un heartbeat au serveur central toutes les 5 minutes (configurable via `HEARTBEAT_INTERVAL`). Cela permet au serveur central de savoir que votre machine est en ligne indépendamment des pings aléatoires.

---

### Redistribution mensuelle

```
Revenus mensuels
    - 15% commission plateforme (fal.ai + hébergement + maintenance)
    = Montant distribuable
        ÷ nb providers éligibles
        = Part égale par provider
```

Exemple avec 180€ et 6 providers :

```
180€ - 27€ = 153€ → 25.50€ par provider
```

**Éligibilité :**

1. Uptime ≥ 60% des pings aléatoires du mois
2. Au moins 1 job réel traité dans le mois

Les pings sont envoyés à des **heures aléatoires** pour éviter la triche au cron. La commission et les montants redistribués sont publiés publiquement chaque mois sur le serveur central.

---

### Sécurité

- Votre serveur ne reçoit que le prompt optimisé — jamais de données personnelles utilisateur
- Le serveur central valide chaque WAV via FFmpeg avant de l'envoyer au musicien
- Un WAV invalide entraîne un **ban immédiat**
- Ne partagez jamais votre `PROVIDER_API_KEY`

---

### Structure du projet

```
obsidian-neural-provider/
├── provider.py
├── requirements_provider.txt
├── .env.example
├── install.sh
├── install.bat
└── README.md
```

---

### Rejoindre le réseau

1. Vérifiez que vous avez le GPU minimum requis
2. Ouvrez une **GitHub Discussion** sur [innermost47/ai-dj](https://github.com/innermost47/ai-dj/discussions) en précisant votre GPU et le modèle préféré
3. L'admin vous envoie votre clé API
4. Copiez `.env.example` → `.env`, renseignez votre clé, lancez le provider

Le pool est limité à **10 providers** en phase 1.

---

### Contribuer

Le code est open source — chaque provider a autant de voix que les autres dans les décisions du projet. Proposez vos améliorations via Pull Request.

---

### Licence

GNU Affero General Public License v3.0 — voir [LICENSE](LICENSE)

---

---

## English

### What is this?

OBSIDIAN Neural is an open source VST3/AU plugin for real-time AI music generation directly in your DAW. This repository contains the **provider kit**: a FastAPI inference server you run on your machine to contribute your GPU to the distributed generation network.

### How it works

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

### Requirements

| Component  | `stable-audio-open-1.0` | `stable-audio-open-small` |
| ---------- | ----------------------- | ------------------------- |
| NVIDIA GPU | RTX 3070+ (8 GB VRAM)   | RTX 3060+ (4 GB VRAM)     |
| RAM        | 16 GB                   | 8 GB                      |
| OS         | Windows / Linux / macOS | Windows / Linux / macOS   |
| Python     | 3.10+                   | 3.10+                     |
| CUDA       | 11.8+                   | 11.8+                     |

> Apple Silicon (M1/M2/M3) supported via Metal.

---

### Installation

#### Linux / macOS

```bash
git clone https://github.com/innermost47/obsidian-neural-provider.git
cd obsidian-neural-provider
chmod +x install.sh
./install.sh
```

#### Windows

```bat
git clone https://github.com/innermost47/obsidian-neural-provider.git
cd obsidian-neural-provider
install.bat
```

The scripts automatically detect your GPU and install the correct PyTorch version (CUDA 11.8, CUDA 12.1, ROCm, or CPU).

---

### Configuration

Copy `.env.example` to `.env` and fill it in:

```bash
cp .env.example .env
```

```env
PROVIDER_API_KEY=op_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CENTRAL_SERVER_URL=https://api.obsidian-neural.com
MODEL=stable-audio-open-1.0
HOST=0.0.0.0
PORT=8001
HEARTBEAT_INTERVAL=300
```

| Variable             | Description                                                      |
| -------------------- | ---------------------------------------------------------------- |
| `PROVIDER_API_KEY`   | Key provided by the OBSIDIAN Neural admin — **never share this** |
| `CENTRAL_SERVER_URL` | Central server URL                                               |
| `MODEL`              | `stable-audio-open-1.0` or `stable-audio-open-small`             |
| `HOST`               | Listening interface (0.0.0.0 = all)                              |
| `PORT`               | Server port (default: 8001)                                      |
| `HEARTBEAT_INTERVAL` | Heartbeat frequency in seconds (default: 300)                    |

---

### Starting the server

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

### Models

| Model                     | VRAM    | Quality | Speed (RTX 3070) | Download size |
| ------------------------- | ------- | ------- | ---------------- | ------------- |
| `stable-audio-open-1.0`   | ~7-8 GB | ⭐⭐⭐  | ~30-120s         | ~5 GB         |
| `stable-audio-open-small` | ~3-4 GB | ⭐⭐    | ~5-15s           | ~1.5 GB       |

First launch downloads the model from HuggingFace. Subsequent launches are instant (cached model).

---

### Endpoints

| Method | Route       | Auth | Description                      |
| ------ | ----------- | ---- | -------------------------------- |
| `GET`  | `/status`   | ✅   | Availability + model info + VRAM |
| `POST` | `/generate` | ✅   | Audio generation → WAV bytes     |
| `GET`  | `/health`   | ❌   | Server health                    |

Authentication via `X-API-Key` header.

---

### Heartbeat

The provider automatically sends a heartbeat to the central server every 5 minutes (configurable via `HEARTBEAT_INTERVAL`). This lets the central server know your machine is online, independently of random pings.

---

### Monthly redistribution

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

### Security

- Your server only receives the optimized prompt — never user personal data
- The central server validates every WAV via FFmpeg before sending it to the musician
- An invalid WAV results in an **immediate ban**
- Never share your `PROVIDER_API_KEY`

---

### Project structure

```
obsidian-neural-provider/
├── provider.py
├── requirements_provider.txt
├── .env.example
├── install.sh
├── install.bat
└── README.md
```

---

### Joining the network

1. Check that you have the minimum required GPU
2. Open a **GitHub Discussion** on [innermost47/ai-dj](https://github.com/innermost47/ai-dj/discussions) with your GPU model and preferred inference model
3. The admin sends you your API key
4. Copy `.env.example` → `.env`, fill in your key, start the provider

The pool is limited to **10 providers** in phase 1.

---

### Contributing

The code is open source — each provider has an equal voice in project decisions. Submit improvements via Pull Request.

---

### License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE)

---

_Made with 🎵 in France — [obsidian-neural.com](https://obsidian-neural.com)_
