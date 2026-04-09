import argparse
import asyncio
import gc
import re
import io
import os
import time
import threading
from contextlib import asynccontextmanager
from typing import Optional
import json
import httpx
import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends, status as fastapi_status
from fastapi.responses import Response, PlainTextResponse, JSONResponse
from pydantic import BaseModel, field_validator
import websockets
import sys
import hashlib


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

PROVIDER_API_KEY: str = os.getenv("PROVIDER_API_KEY", "")
CENTRAL_SERVER_URL: str = os.getenv("CENTRAL_SERVER_URL", "")
HEARTBEAT_INTERVAL: int = int(os.getenv("HEARTBEAT_INTERVAL", "300"))
SHARED_SECRET = os.getenv("SERVER_TO_PROVIDER_KEY")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
MODEL_KEY: str = os.getenv("MODEL", "RoyalCities/Foundation-1")
CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE", "/data/credentials.json")
SUPPORTED_MODELS = {
    "RoyalCities/Foundation-1": "RoyalCities/Foundation-1",
}

MAX_DURATION = 30
MIN_DURATION = 2
TARGET_SAMPLE_RATE = 44100
LOAD_MODEL_ON_THE_FLY = False

generator: Optional["AudioGenerator"] = None


class ProcessRequest(BaseModel):
    action: str
    prompt: Optional[str] = None
    duration: Optional[int] = 10
    seed: Optional[int] = None

    class Config:
        extra = "forbid"

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        if v not in ("health", "status", "generate"):
            raise ValueError("action must be 'health', 'status', or 'generate'")
        return v

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v, info):
        if info.data.get("action") == "generate" and v is not None:
            if not (MIN_DURATION <= v <= MAX_DURATION):
                raise ValueError(
                    f"duration must be between {MIN_DURATION} and {MAX_DURATION}"
                )
        return v

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v, info):
        if info.data.get("action") == "generate":
            if not v or not v.strip():
                raise ValueError("prompt is required for generate action")
        return v

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v, info):
        if v is not None:
            if not (0 <= v <= 2**31 - 1):
                raise ValueError("seed must be between 0 and 2147483647")
        return v


async def verify_server_identity(x_api_key: str = Header(None)):
    if not SHARED_SECRET or x_api_key != SHARED_SECRET:
        raise HTTPException(
            status_code=fastapi_status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
    return x_api_key


async def activate_with_token(token: str, central_url: str) -> dict:
    if os.path.exists(CREDENTIALS_FILE):
        print("🔑 Loading saved credentials...")
        with open(CREDENTIALS_FILE, "r") as f:
            return json.load(f)

    print("🔑 Activating provider with token...")
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{central_url.rstrip('/')}/api/v1/providers/activate",
                json={"token": token},
            )
            if response.status_code != 200:
                print(f"❌ Activation failed: {response.text}")
                sys.exit(1)
            data = response.json()
            os.makedirs(os.path.dirname(CREDENTIALS_FILE), exist_ok=True)
            with open(CREDENTIALS_FILE, "w") as f:
                json.dump(data, f)
            print(f"✅ Activated as: {data['provider_name']}")
            return data
        except Exception as e:
            print(f"❌ Cannot reach central server: {e}")
            sys.exit(1)


def clean_code_for_hashing(text: str) -> bytes:
    cleaned = re.sub(r"\s+", "", text)
    return cleaned.encode("utf-8")


def _compute_self_hash() -> str:
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            local_content = f.read()

        content_bytes = clean_code_for_hashing(local_content)

        api_key_hashed = hashlib.sha256(PROVIDER_API_KEY.encode()).hexdigest()
        identity = f"{api_key_hashed}:{SHARED_SECRET}".encode()

        return hashlib.sha256(content_bytes + identity).hexdigest()
    except Exception as e:
        print(f"❌ Hash calculation error: {e}")
        return "unknown"


def connect_to_central_registry():

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    attempts = 0

    while True:
        if not CENTRAL_SERVER_URL or not PROVIDER_API_KEY:
            print("⚠️ WebSocket: credentials not ready, retrying in 10s...")
            time.sleep(10)
            continue
        try:
            ws_url = (
                CENTRAL_SERVER_URL.replace("http://", "ws://")
                .replace("https://", "wss://")
                .rstrip("/")
            )
            uri = f"{ws_url}/api/v1/providers/connect"

            headers = {
                "X-Provider-Key": PROVIDER_API_KEY,
                "X-Model": MODEL_KEY,
                "X-Provider-Hash": SELF_HASH,
            }
            print(f"🔌 Attempting to connect to the central registry: {uri}...")
            websocket = loop.run_until_complete(
                websockets.connect(
                    uri,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=60,
                )
            )
            print("✅ Connected to the central server (Active presence)")

            while True:
                loop.run_until_complete(websocket.recv())

        except Exception as e:
            attempts += 1
            if attempts > 3:
                print(
                    f"\nCritical Error: Failed to connect to registry after {attempts} attempts."
                )
                print(f"Details: {e}")
                print("Exiting process...")
                exit(1)
            print(f"❌ Register disconnection (Error: {e})")
            print("🔄 Attempting to reconnect in 10 seconds...")
            time.sleep(10)


def send_heartbeat_sync():
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        if not CENTRAL_SERVER_URL or not PROVIDER_API_KEY:
            continue
        try:
            with httpx.Client(timeout=10.0) as client:
                client.post(
                    f"{CENTRAL_SERVER_URL.rstrip('/')}/api/v1/providers/heartbeat",
                    headers={
                        "X-API-Key": PROVIDER_API_KEY,
                        "X-Provider-Hash": SELF_HASH,
                    },
                    json=True,
                )
            print(f"💓 Heartbeat sent")
        except Exception as e:
            print(f"⚠️  Heartbeat failed: {e}")


class AudioGenerator:
    def __init__(self, model_key: str = "stable-audio-open-1.0"):
        self.model_key = model_key
        self.model_id = SUPPORTED_MODELS[model_key]
        self.pipeline = None
        self.sample_rate = TARGET_SAMPLE_RATE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()
        self._generating = False

    def load(self):
        from diffusers import StableAudioPipeline

        print(f"⚡ Loading {self.model_id} on {self.device}...")

        if self.device == "cuda":
            from diffusers import (
                BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
                StableAudioDiTModel,
            )
            from transformers import (
                BitsAndBytesConfig as TransformersBitsAndBytesConfig,
                T5EncoderModel,
            )

            text_encoder = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                quantization_config=TransformersBitsAndBytesConfig(load_in_8bit=True),
                torch_dtype=torch.float16,
            )
            transformer = StableAudioDiTModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
                torch_dtype=torch.float16,
            )
            self.pipeline = StableAudioPipeline.from_pretrained(
                self.model_id,
                text_encoder=text_encoder,
                transformer=transformer,
                torch_dtype=torch.float16,
                device_map="balanced",
            )
        else:
            raise RuntimeError("No CUDA GPU available. CPU mode is not allowed.")

        self.sample_rate = self.pipeline.vae.sampling_rate
        print(f"✅ Model loaded (sample rate: {self.sample_rate}Hz)")

    def unload(self):
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def generate_with_seed(self, prompt: str, duration: int, seed: int) -> bytes:
        with self._lock:
            self._generating = True
            try:
                return self._generate_with_seed(prompt, duration, seed)
            finally:
                self._generating = False

    def _generate_with_seed(self, prompt: str, duration: int, seed: int) -> bytes:
        try:
            if LOAD_MODEL_ON_THE_FLY:
                self.load()
            duration = max(MIN_DURATION, min(10, duration))
            num_inference_steps = 50
            cfg_scale = 7.0

            gen = torch.Generator(device=self.device).manual_seed(seed)

            t0 = time.time()

            result = self.pipeline(
                prompt,
                negative_prompt="Low quality, distorted, noise",
                num_inference_steps=num_inference_steps,
                audio_end_in_s=duration,
                num_waveforms_per_prompt=1,
                generator=gen,
                guidance_scale=cfg_scale,
            )

            print(f"✅ Verify done in {time.time() - t0:.1f}s")

            audio = result.audios[0].float().cpu().numpy()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.sample_rate != TARGET_SAMPLE_RATE:
                if len(audio.shape) > 1 and audio.shape[0] == 2:
                    audio = np.array(
                        [
                            librosa.resample(
                                audio[0],
                                orig_sr=self.sample_rate,
                                target_sr=TARGET_SAMPLE_RATE,
                            ),
                            librosa.resample(
                                audio[1],
                                orig_sr=self.sample_rate,
                                target_sr=TARGET_SAMPLE_RATE,
                            ),
                        ]
                    )
                else:
                    audio = librosa.resample(
                        audio, orig_sr=self.sample_rate, target_sr=TARGET_SAMPLE_RATE
                    )

            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9

            if len(audio.shape) > 1 and audio.shape[0] == 2:
                audio = audio.T

            buf = io.BytesIO()
            sf.write(buf, audio, TARGET_SAMPLE_RATE, format="WAV")
            buf.seek(0)
            return buf.read()
        finally:
            if LOAD_MODEL_ON_THE_FLY:
                self.unload()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PROVIDER_API_KEY, SHARED_SECRET, SELF_HASH

    obsidian_token = os.getenv("OBSIDIAN_TOKEN", "")
    if obsidian_token:
        if not CENTRAL_SERVER_URL:
            print("❌ CENTRAL_SERVER_URL is required with OBSIDIAN_TOKEN")
            sys.exit(1)
        creds = await activate_with_token(obsidian_token, CENTRAL_SERVER_URL)
        PROVIDER_API_KEY = creds["api_key"]
        SHARED_SECRET = creds["server_to_provider_key"]
    else:
        if not PROVIDER_API_KEY or not SHARED_SECRET:
            print("❌ PROVIDER_API_KEY and SERVER_TO_PROVIDER_KEY required in .env")
            sys.exit(1)

    SELF_HASH = _compute_self_hash()

    ws_thread = threading.Thread(target=connect_to_central_registry, daemon=True)
    ws_thread.start()

    hb_thread = threading.Thread(target=send_heartbeat_sync, daemon=True)
    hb_thread.start()

    yield


app = FastAPI(
    title="OBSIDIAN Neural Provider",
    description="GPU inference server for the OBSIDIAN Neural distributed network",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/process", dependencies=[Depends(verify_server_identity)])
async def process(request: ProcessRequest):

    if request.action == "health":
        return JSONResponse(
            content={
                "status": "ok",
                "model": generator.model_key,
                "model_id": generator.model_id,
            },
            headers={"X-Provider-Hash": SELF_HASH},
        )

    elif request.action == "status":
        is_available = generator is not None and not generator._generating
        vram_info = {}
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vram_used = torch.cuda.memory_allocated(0) / 1024**3
            vram_info = {
                "vram_total_gb": round(vram_total, 1),
                "vram_used_gb": round(vram_used, 1),
            }
        return JSONResponse(
            content={
                "available": is_available,
                "api_key": PROVIDER_API_KEY,
                "model": generator.model_key,
                "model_id": generator.model_id,
                "device": generator.device,
                "generating": generator._generating if generator else False,
                **vram_info,
            },
            headers={"X-Provider-Hash": SELF_HASH},
        )

    elif request.action == "generate":
        if generator is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        if generator._generating:
            raise HTTPException(
                status_code=503, detail="Already generating — try again later"
            )

        duration = max(MIN_DURATION, min(MAX_DURATION, request.duration))

        try:
            loop = asyncio.get_event_loop()
            wav_bytes = await loop.run_in_executor(
                None,
                generator.generate_with_seed,
                request.prompt,
                duration,
                request.seed,
            )
            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "X-Provider-Key": PROVIDER_API_KEY,
                    "X-Model": generator.model_key,
                    "X-Duration": str(duration),
                    "X-Sample-Rate": str(TARGET_SAMPLE_RATE),
                    "X-Seed": str(request.seed),
                    "X-Provider-Hash": SELF_HASH,
                },
            )
        except Exception as e:
            print(f"❌ Generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Service OK"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBSIDIAN Neural GPU Provider Server")
    parser.add_argument("--key", default="", help="API key (overrides .env)")
    parser.add_argument("--port", type=int, default=0, help="Port (overrides .env)")
    parser.add_argument("--host", default="", help="Host (overrides .env)")
    parser.add_argument(
        "--server", default="", help="Central server URL (overrides .env)"
    )
    parser.add_argument(
        "--on-the-fly",
        action="store_true",
        help="Load Stable Diffusion at each generation process",
    )

    args = parser.parse_args()

    if args.key:
        PROVIDER_API_KEY = args.key
    if args.port:
        PORT = args.port
    if args.host:
        HOST = args.host
    if args.server:
        CENTRAL_SERVER_URL = args.server

    env_val = os.getenv("LOAD_ON_THE_FLY", "false").lower()
    LOAD_MODEL_ON_THE_FLY = args.on_the_fly or env_val == "true"

    if MODEL_KEY not in SUPPORTED_MODELS:
        print(
            f"❌ Unknown model: {MODEL_KEY}. Choose from: {list(SUPPORTED_MODELS.keys())}"
        )
        exit(1)

    if not torch.cuda.is_available():
        print(
            "❌ No CUDA GPU detected. CPU mode is not allowed in the provider network."
        )
        print(
            "   Minimum requirement: NVIDIA RTX 3070 (8GB VRAM) or RTX 3060 (4GB VRAM) for the small model."
        )
        exit(1)

    generator = AudioGenerator(model_key=MODEL_KEY)
    if not LOAD_MODEL_ON_THE_FLY:
        generator.load()

    print(f"\n{'='*55}")
    print(f"  OBSIDIAN Neural Provider")
    print(f"  Model  : {generator.model_id}")
    print(f"  Device : {generator.device}")
    print(f"  Host   : {HOST}:{PORT}")
    print(f"  Server : {CENTRAL_SERVER_URL or 'not configured'}")
    print(f"{'='*55}\n")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info", backlog=2048)
