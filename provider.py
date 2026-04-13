import argparse
import asyncio
import gc
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
from pydantic import BaseModel, field_validator, ConfigDict
import websockets
import sys
import ollama


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

PROVIDER_API_KEY: str = os.getenv("PROVIDER_API_KEY", "")
CENTRAL_SERVER_URL: str = os.getenv("CENTRAL_SERVER_URL", "")
HEARTBEAT_INTERVAL: int = int(os.getenv("HEARTBEAT_INTERVAL", "300"))
SHARED_SECRET = os.getenv("SERVER_TO_PROVIDER_KEY")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
MODEL_KEY: str = os.getenv("MODEL", "stable-audio-open-1.0")
CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE", "/data/credentials.json")
SUPPORTED_MODELS = {
    "stable-audio-open-1.0": "stabilityai/stable-audio-open-1.0",
}
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = "gemma4:e2b"
FOUNDATION1_HF_REPO = "RoyalCities/Foundation-1"
FOUNDATION1_SUPPORTED_BPM = [100, 110, 120, 128, 130, 140, 150]
MAX_DURATION = 30
MIN_DURATION = 2
TARGET_SAMPLE_RATE = 44100


_llm_generating: bool = False

generator: Optional["AudioGenerator"] = None
foundation1_generator: Optional["Foundation1Generator"] = None


class AudioProcessRequest(BaseModel):
    action: str
    prompt: Optional[str] = None
    duration: Optional[int] = 10
    seed: Optional[int] = None
    model: Optional[str] = "stable-audio-open-1.0"
    bpm: Optional[int] = None
    bars: Optional[int] = 8
    key: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        if v not in ("health", "status", "generate"):
            raise ValueError("action must be 'health', 'status', or 'generate'")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        allowed = {"stable-audio-open-1.0", "foundation-1"}
        if v not in allowed:
            raise ValueError(f"model must be one of {allowed}")
        return v

    @field_validator("bars")
    @classmethod
    def validate_bars(cls, v):
        if v is not None and v not in (4, 8):
            raise ValueError("bars must be 4 or 8")
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
    def validate_seed(cls, v):
        if v is not None and not (0 <= v <= 2**31 - 1):
            raise ValueError("seed must be between 0 and 2147483647")
        return v


class ConversationMessage(BaseModel):
    role: str
    content: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ("system", "user", "assistant"):
            raise ValueError("role must be 'system', 'user', or 'assistant'")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("content cannot be empty")
        if len(v) > 32000:
            raise ValueError("content exceeds 32000 characters")
        return v


class LLMInferRequest(BaseModel):
    action: str
    system_prompt: str
    history: list[ConversationMessage] = []
    user_message: str
    image_base64: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("system_prompt cannot be empty")
        if len(v) > 32000:
            raise ValueError("system_prompt exceeds 32000 characters")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v):
        if not v or not v.strip():
            raise ValueError("user_message cannot be empty")
        if len(v) > 8000:
            raise ValueError("user_message exceeds 8000 characters")
        return v

    @field_validator("image_base64")
    @classmethod
    def validate_image_base64(cls, v):
        if v is not None:
            import base64

            try:
                base64.b64decode(v, validate=True)
            except Exception:
                raise ValueError("image_base64 is not valid base64")
            if len(v) > 13_600_000:
                raise ValueError("image_base64 exceeds 10MB")
        return v


class LLMInferResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system_prompt: str
    history: list[ConversationMessage]
    user_message: str
    response: str
    model: str
    provider_key: str


def sanitize_header(value: str) -> str:
    return value.encode("latin-1", errors="replace").decode("latin-1")


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


async def ollama_infer(
    system_prompt: str,
    history: list[ConversationMessage],
    user_message: str,
    image_base64: Optional[str] = None,
) -> str:
    try:
        client = ollama.AsyncClient()

        messages = [{"role": "system", "content": system_prompt}]

        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        if image_base64:
            messages.append(
                {
                    "role": "user",
                    "content": user_message,
                    "images": [image_base64],
                }
            )
        else:
            messages.append({"role": "user", "content": user_message})

        response = await client.chat(
            model=LLM_MODEL,
            messages=messages,
        )
        return response.message.content
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


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
            attempts = 0
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
                    },
                    json=True,
                )
            print(f"💓 Heartbeat sent")
        except Exception as e:
            print(f"⚠️  Heartbeat failed: {e}")


def _nearest_supported_bpm(bpm: int) -> int:
    return min(FOUNDATION1_SUPPORTED_BPM, key=lambda x: abs(x - bpm))


class Foundation1Generator:
    def __init__(self):
        self._lock = threading.Lock()
        self._generating = False

    def generate(
        self, prompt: str, bpm: int, bars: int, key: Optional[str], seed: int
    ) -> bytes:
        with self._lock:
            self._generating = True
            try:
                return self._generate(prompt, bpm, bars, key, seed)
            finally:
                self._generating = False

    def _generate(
        self, prompt: str, bpm: int, bars: int, key: Optional[str], seed: int
    ) -> bytes:
        import math
        import json as _json
        from einops import rearrange
        from huggingface_hub import hf_hub_download
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = (
            torch.bfloat16
            if (device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )

        ckpt_path = hf_hub_download(
            repo_id=FOUNDATION1_HF_REPO, filename="Foundation_1.safetensors"
        )
        config_path = hf_hub_download(
            repo_id=FOUNDATION1_HF_REPO, filename="model_config.json"
        )

        with open(config_path) as f:
            model_config = _json.load(f)

        model = create_model_from_config(model_config)
        state_dict = load_ckpt_state_dict(ckpt_path)
        model.load_state_dict(state_dict)
        model.to(device).to(dtype).eval().requires_grad_(False)

        try:
            sample_rate = model_config["sample_rate"]
            snapped_bpm = _nearest_supported_bpm(bpm)
            if snapped_bpm != bpm:
                print(
                    f"⚠️  BPM {bpm} → snapped à {snapped_bpm} (stretch géré par le serveur central)"
                )

            clip_seconds = (60.0 / snapped_bpm) * 4 * bars
            seconds_int = int(math.ceil(clip_seconds))
            min_input = getattr(model, "min_input_length", None)
            target_samples = int(seconds_int * sample_rate)
            if (
                isinstance(min_input, int)
                and min_input > 0
                and target_samples % min_input != 0
            ):
                target_samples += min_input - (target_samples % min_input)

            parts = [prompt]
            if key:
                parts.append(key)
            parts.append(f"{bars} Bars")
            parts.append(f"{snapped_bpm} BPM")
            full_prompt = ", ".join(parts)
            print(f"📝 Foundation-1 prompt: {full_prompt}")

            conditioning = [
                {
                    "prompt": full_prompt,
                    "seconds_start": 0.0,
                    "seconds_total": float(seconds_int),
                }
            ]

            t0 = time.time()
            audio = generate_diffusion_cond(
                model,
                conditioning=conditioning,
                negative_conditioning=None,
                steps=75,
                cfg_scale=7.0,
                batch_size=1,
                sample_size=target_samples,
                sample_rate=sample_rate,
                seed=seed,
                device=device,
                sampler_type="dpmpp-3m-sde",
                sigma_min=0.03,
                sigma_max=500,
                scale_phi=0.0,
            )
            print(f"✅ Foundation-1 done in {time.time() - t0:.2f}s")

            audio = rearrange(audio, "b d n -> d (b n)")
            audio = audio.to(torch.float32).clamp(-1, 1)

            clip_samples = int(round(clip_seconds * sample_rate))
            audio = audio[:, : max(1, min(audio.shape[-1], clip_samples))].contiguous()

            fade_len = min(int(round(0.015 * sample_rate)), audio.shape[-1])
            if fade_len > 1:
                ramp = torch.linspace(
                    1.0, 0.0, steps=fade_len, device=audio.device, dtype=audio.dtype
                )
                audio[:, -fade_len:] *= ramp

            audio_np = audio.cpu().numpy()
            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                audio_np = audio_np / max_val * 0.9

            buf = io.BytesIO()
            sf.write(buf, audio_np, sample_rate, format="WAV")
            buf.seek(0)
            return buf.read(), snapped_bpm

        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


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

    def generate_with_seed(
        self,
        prompt: str,
        duration: int,
        seed: int,
        bpm: Optional[int] = None,
        key: Optional[str] = None,
    ) -> bytes:
        with self._lock:
            self._generating = True
            try:
                return self._generate_with_seed(prompt, duration, seed, bpm, key)
            finally:
                self._generating = False

    def _generate_with_seed(
        self,
        prompt: str,
        duration: int,
        seed: int,
        bpm: Optional[int] = None,
        key: Optional[str] = None,
    ) -> bytes:
        try:
            self.load()
            duration = max(MIN_DURATION, min(10, duration))
            num_inference_steps = 50
            cfg_scale = 7.0

            final_prompt = prompt.strip()
            if key:
                final_prompt += f", {key}"
            if bpm:
                final_prompt += f", {bpm} BPM"

            gen = torch.Generator(device=self.device).manual_seed(seed)

            t0 = time.time()

            result = self.pipeline(
                final_prompt,
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
            self.unload()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PROVIDER_API_KEY, SHARED_SECRET, generator, foundation1_generator

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

    ws_thread = threading.Thread(target=connect_to_central_registry, daemon=True)
    ws_thread.start()

    hb_thread = threading.Thread(target=send_heartbeat_sync, daemon=True)
    hb_thread.start()

    if generator is None:
        generator = AudioGenerator(model_key=MODEL_KEY)

    foundation1_generator = Foundation1Generator()

    print(f"  Model        : {generator.model_id}")
    print(f"  Device       : {generator.device}")
    print(f"  Foundation-1 : {FOUNDATION1_HF_REPO} (cache HF)")

    yield


app = FastAPI(
    title="OBSIDIAN Neural Provider",
    description="GPU inference server for the OBSIDIAN Neural distributed network",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/process", dependencies=[Depends(verify_server_identity)])
async def process(raw: dict):
    action = raw.get("action")

    if action == "llm_infer":
        global _llm_generating
        if _llm_generating or (generator and generator._generating):
            raise HTTPException(status_code=503, detail="Provider busy")
        try:
            request = LLMInferRequest(**raw)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

        t0 = time.time()
        _llm_generating = True
        try:
            llm_response = await ollama_infer(
                request.system_prompt,
                request.history,
                request.user_message,
                request.image_base64,
            )
            print(f"✅ LLM infer done in {time.time() - t0:.1f}s")
            return LLMInferResponse(
                system_prompt=request.system_prompt,
                history=request.history,
                user_message=request.user_message,
                response=llm_response,
                model=LLM_MODEL,
                provider_key=PROVIDER_API_KEY,
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503, detail="Ollama not reachable on provider"
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502, detail=f"Ollama error: {e.response.text}"
            )
        except Exception as e:
            print(f"❌ LLM infer error: {e}")
            raise HTTPException(status_code=500, detail=f"LLM infer failed: {str(e)}")
        finally:
            _llm_generating = False

    if action in ("health", "status", "generate"):
        try:
            request = AudioProcessRequest(**raw)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

        if request.action == "health":
            return JSONResponse(
                content={
                    "status": "ok",
                    "model": generator.model_key,
                    "model_id": generator.model_id,
                    "foundation1": FOUNDATION1_HF_REPO,
                }
            )

        elif request.action == "status":
            is_available = (
                generator is not None
                and not generator._generating
                and not (foundation1_generator and foundation1_generator._generating)
                and not _llm_generating
            )
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
                        "generating": (
                            (generator._generating if generator else False)
                            or (
                                foundation1_generator._generating
                                if foundation1_generator
                                else False
                            )
                            or _llm_generating
                        ),
                        "generating_llm": _llm_generating,
                        **vram_info,
                    }
                )

        elif request.action == "generate":
            use_foundation = request.model == "foundation-1"

            if use_foundation:
                if foundation1_generator is None:
                    raise HTTPException(
                        status_code=503, detail="Foundation-1 not available"
                    )
                if foundation1_generator._generating:
                    raise HTTPException(
                        status_code=503, detail="Already generating — try again later"
                    )
                if not request.bpm:
                    raise HTTPException(
                        status_code=422, detail="bpm is required for foundation-1 model"
                    )

                try:
                    loop = asyncio.get_event_loop()
                    wav_bytes, snapped_bpm = await loop.run_in_executor(
                        None,
                        foundation1_generator.generate,
                        request.prompt,
                        request.bpm,
                        request.bars or 8,
                        request.key,
                        request.seed,
                    )
                    return Response(
                        content=wav_bytes,
                        media_type="audio/wav",
                        headers={
                            "X-Provider-Key": PROVIDER_API_KEY,
                            "X-Model": "foundation-1",
                            "X-BPM": str(request.bpm),
                            "X-Snapped-BPM": str(snapped_bpm),
                            "X-Bars": str(request.bars or 8),
                            "X-Key": sanitize_header(str(request.key or "")),
                            "X-Seed": str(request.seed),
                        },
                    )
                except Exception as e:
                    print(f"❌ Foundation-1 error: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Foundation-1 generation failed: {str(e)}",
                    )

            else:
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
                        request.bpm,
                        request.key,
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
                        },
                    )
                except Exception as e:
                    print(f"❌ Generation error: {e}")
                    raise HTTPException(
                        status_code=500, detail=f"Generation failed: {str(e)}"
                    )

    raise HTTPException(
        status_code=422,
        detail=f"Unknown action '{action}'. Valid: health, status, generate, llm_infer",
    )


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

    args = parser.parse_args()

    if args.key:
        PROVIDER_API_KEY = args.key
    if args.port:
        PORT = args.port
    if args.host:
        HOST = args.host
    if args.server:
        CENTRAL_SERVER_URL = args.server

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

    print(f"\n{'='*55}")
    print(f"  OBSIDIAN Neural Provider")
    print(f"  Host   : {HOST}:{PORT}")
    print(f"  Server : {CENTRAL_SERVER_URL or 'not configured'}")
    print(f"{'='*55}\n")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info", backlog=2048)
