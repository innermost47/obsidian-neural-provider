import os
import sys


def _bootstrap_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="OBSIDIAN Neural GPU Provider Server")
    parser.add_argument("--key", default="", help="Provider API key (overrides .env)")
    parser.add_argument("--secret", default="", help="Server-to-provider shared key")
    parser.add_argument("--port", type=int, default=0, help="Port (overrides .env)")
    parser.add_argument("--host", default="", help="Host (overrides .env)")
    parser.add_argument("--server", default="", help="Central server URL")
    args, _ = parser.parse_known_args()

    if args.key:
        os.environ["PROVIDER_API_KEY"] = args.key
    if args.secret:
        os.environ["SERVER_TO_PROVIDER_KEY"] = args.secret
    if args.port:
        os.environ["PORT"] = str(args.port)
    if args.host:
        os.environ["HOST"] = args.host
    if args.server:
        os.environ["CENTRAL_SERVER_URL"] = args.server


if __name__ == "__main__":
    _bootstrap_cli()

import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from stable_audio_tools import get_pretrained_model

import credentials
from settings import (
    CENTRAL_SERVER_URL,
    CREDENTIALS_FILE,
    HOST,
    MAX_DURATION,
    MIN_DURATION,
    MODEL_KEY,
    PORT,
    SA3_KEEP_IN_RAM,
    STABLE_AUDIO_3_MODELS,
    STABLE_AUDIO_MODELS,
    SUPPORTED_MODELS,
    TARGET_SAMPLE_RATE,
    DEFAULT_SA3_DURATION,
    DEFAULT_BARS,
)
from models import AudioProcessRequest
from server_utils import (
    activate_with_token,
    connect_to_central_registry,
    sanitize_header,
    send_heartbeat_sync,
    verify_server_identity,
)
from audio_generator import AudioGenerator
from sa3_generator import StableAudio3Generator
from sa_generator import StableAudioGenerator

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


_vram_lock = asyncio.Lock()

generator: Optional["AudioGenerator"] = None
stable_audio_generators: dict[str, "StableAudioGenerator"] = {}
stable_audio_3_generators: dict[str, "StableAudio3Generator"] = {}


def _mask(value: str) -> str:
    if not value:
        return "<empty>"
    return f"{value[:6]}…{value[-4:]} (len={len(value)})"


def _load_models() -> None:
    global generator

    if generator is None:
        generator = AudioGenerator(model_key=MODEL_KEY)

    for model_key, (repo_id, ckpt, config) in STABLE_AUDIO_MODELS.items():
        stable_audio_generators[model_key] = StableAudioGenerator(
            repo_id, ckpt, config, model_key=model_key
        )
        print(f"  {model_key} : {repo_id}/{ckpt}")

    for model_key, repo_id in STABLE_AUDIO_3_MODELS.items():
        stable_audio_3_generators[model_key] = StableAudio3Generator(
            repo_id, model_key=model_key
        )
        print(f"  {model_key} : {repo_id} (SA3)")

    if SA3_KEEP_IN_RAM:
        print("🔥 Pre-loading SA3 into RAM (SA3_KEEP_IN_RAM=true)...")
        for sa3 in stable_audio_3_generators.values():
            sa3._cached_model, sa3._cached_config = get_pretrained_model(sa3.repo_id)
        print("✅ SA3 ready in RAM")

    print(f"  Model  : {generator.model_id}")
    print(f"  Device : {generator.device}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if credentials.load_from_file():
        print(f"🔑 Credentials loaded from {CREDENTIALS_FILE}")
    elif os.getenv("OBSIDIAN_TOKEN", ""):
        if not CENTRAL_SERVER_URL:
            print("❌ CENTRAL_SERVER_URL is required when using OBSIDIAN_TOKEN")
            sys.exit(1)
        creds = await activate_with_token(
            os.getenv("OBSIDIAN_TOKEN"), CENTRAL_SERVER_URL
        )
        credentials.set_credentials(creds["api_key"], creds["server_to_provider_key"])
        print(f"🔑 Activation successful, credentials saved to {CREDENTIALS_FILE}")
    elif credentials.load_from_env():
        print("🔑 Credentials loaded from environment (manual mode)")
        print("   Note: no credentials file — this key must already exist server-side.")
    else:
        print("❌ No credentials available.")
        print("   Provide OBSIDIAN_TOKEN, or set PROVIDER_API_KEY and")
        print("   SERVER_TO_PROVIDER_KEY, or mount an existing credentials file.")
        sys.exit(1)

    print(f"   api_key     : {_mask(credentials.get_api_key())}")
    print(f"   shared_key  : {_mask(credentials.get_shared_secret())}")

    threading.Thread(target=connect_to_central_registry, daemon=True).start()
    threading.Thread(target=send_heartbeat_sync, daemon=True).start()

    _load_models()

    yield


app = FastAPI(
    title="OBSIDIAN Neural Provider",
    description="GPU inference server for the OBSIDIAN Neural distributed network",
    version="1.0.0",
    lifespan=lifespan,
)


def _audio_response(wav_bytes: bytes, extra_headers: dict) -> Response:
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Provider-Key": credentials.get_api_key(),
            **extra_headers,
        },
    )


def _require_seed(request: AudioProcessRequest) -> int:
    if request.seed is None:
        raise HTTPException(status_code=422, detail="seed is required for generation")
    return int(request.seed)


@app.post("/process", dependencies=[Depends(verify_server_identity)])
async def process(raw: dict):
    action = raw.get("action")

    if action not in ("health", "status", "generate"):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown action '{action}'. Valid: health, status, generate",
        )

    try:
        request = AudioProcessRequest(**raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    if generator is None:
        raise HTTPException(status_code=503, detail="Provider is still starting up")

    if request.action == "health":
        return JSONResponse(
            content={
                "status": "ok",
                "model": generator.model_key,
                "model_id": generator.model_id,
                "available_models": (
                    list(STABLE_AUDIO_MODELS.keys())
                    + list(STABLE_AUDIO_3_MODELS.keys())
                    + ["stable-audio-open-1.0"]
                ),
            }
        )

    if request.action == "status":
        any_sa3_generating = any(
            s._generating for s in stable_audio_3_generators.values()
        )
        any_sag_generating = any(
            s._generating for s in stable_audio_generators.values()
        )
        is_generating = (
            generator._generating or any_sa3_generating or any_sag_generating
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
                "available": not is_generating,
                "api_key": credentials.get_api_key(),
                "model": generator.model_key,
                "model_id": generator.model_id,
                "device": generator.device,
                "generating": is_generating,
                "generating_llm": False,
                **vram_info,
            }
        )

    seed = _require_seed(request)
    loop = asyncio.get_running_loop()

    if request.model in STABLE_AUDIO_3_MODELS:
        sa3 = stable_audio_3_generators.get(request.model)
        if sa3 is None:
            raise HTTPException(
                status_code=503, detail=f"Model {request.model} not available"
            )
        if sa3._generating:
            raise HTTPException(
                status_code=503, detail="Already generating — try again later"
            )

        duration = max(
            MIN_DURATION,
            min(MAX_DURATION, request.duration or DEFAULT_SA3_DURATION),
        )
        try:
            async with _vram_lock:
                wav_bytes = await loop.run_in_executor(
                    None,
                    sa3.generate,
                    request.prompt,
                    duration,
                    seed,
                    request.bpm,
                    request.key,
                )
        except Exception as e:
            print(f"❌ SA3 error: {e}")
            raise HTTPException(status_code=500, detail=f"SA3 generation failed: {e}")

        return _audio_response(
            wav_bytes,
            {
                "X-Model": request.model,
                "X-Duration": str(duration),
                "X-Sample-Rate": str(TARGET_SAMPLE_RATE),
                "X-Seed": str(seed),
            },
        )

    if request.model in STABLE_AUDIO_MODELS:
        sag = stable_audio_generators.get(request.model)
        if sag is None:
            raise HTTPException(
                status_code=503, detail=f"Model {request.model} not available"
            )
        if sag._generating:
            raise HTTPException(
                status_code=503, detail="Already generating — try again later"
            )
        if not request.bpm:
            raise HTTPException(
                status_code=422, detail="bpm is required for this model"
            )

        bars = request.bars or DEFAULT_BARS
        try:
            async with _vram_lock:
                wav_bytes, snapped_bpm = await loop.run_in_executor(
                    None,
                    sag.generate,
                    request.prompt,
                    request.bpm,
                    bars,
                    request.key,
                    seed,
                )
        except Exception as e:
            print(f"❌ {request.model} error: {e}")
            raise HTTPException(
                status_code=500, detail=f"{request.model} generation failed: {e}"
            )

        return _audio_response(
            wav_bytes,
            {
                "X-Model": request.model,
                "X-BPM": str(request.bpm),
                "X-Snapped-BPM": str(snapped_bpm),
                "X-Bars": str(bars),
                "X-Key": sanitize_header(str(request.key or "")),
                "X-Seed": str(seed),
            },
        )

    if generator._generating:
        raise HTTPException(
            status_code=503, detail="Already generating — try again later"
        )

    duration = max(
        MIN_DURATION, min(MAX_DURATION, request.duration or DEFAULT_SA3_DURATION)
    )
    try:
        async with _vram_lock:
            wav_bytes = await loop.run_in_executor(
                None,
                generator.generate_with_seed,
                request.prompt,
                duration,
                seed,
                request.bpm,
                request.key,
            )
    except Exception as e:
        print(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return _audio_response(
        wav_bytes,
        {
            "X-Model": generator.model_key,
            "X-Duration": str(duration),
            "X-Sample-Rate": str(TARGET_SAMPLE_RATE),
            "X-Seed": str(seed),
        },
    )


@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Service OK"


if __name__ == "__main__":
    if MODEL_KEY not in SUPPORTED_MODELS:
        print(f"❌ Unknown model: {MODEL_KEY}")
        print(f"   Choose from: {list(SUPPORTED_MODELS.keys())}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print(
            "❌ No CUDA GPU detected. CPU mode is not allowed in the provider network."
        )
        print("   Minimum requirement: NVIDIA RTX 3070 (8GB VRAM), or")
        print("   RTX 3060 (4GB VRAM) for the small model.")
        sys.exit(1)

    print(f"\n{'=' * 55}")
    print("  OBSIDIAN Neural Provider")
    print(f"  Host   : {HOST}:{PORT}")
    print(f"  Server : {CENTRAL_SERVER_URL or 'not configured'}")
    print(f"{'=' * 55}\n")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info", backlog=2048)
