import argparse
import asyncio
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import Response, PlainTextResponse, JSONResponse
import sys
from stable_audio_tools import get_pretrained_model
from settings import (
    PROVIDER_API_KEY,
    SHARED_SECRET,
    CENTRAL_SERVER_URL,
    MODEL_KEY,
    STABLE_AUDIO_3_MODELS,
    STABLE_AUDIO_MODELS,
    SA3_KEEP_IN_RAM,
    MIN_DURATION,
    MAX_DURATION,
    TARGET_SAMPLE_RATE,
    SUPPORTED_MODELS,
)
from models import AudioProcessRequest
from server_utils import (
    activate_with_token,
    connect_to_central_registry,
    sanitize_header,
    send_heartbeat_sync,
    verify_server_identity,
)
from sa3_generator import StableAudio3Generator
from sa_generator import StableAudioGenerator
from audio_generator import AudioGenerator

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_vram_lock = asyncio.Lock()

generator: Optional["AudioGenerator"] = None
stable_audio_generators: dict[str, "StableAudioGenerator"] = {}
stable_audio_3_generators: dict[str, "StableAudio3Generator"] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PROVIDER_API_KEY, SHARED_SECRET, generator, stable_audio_generators

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
                    "available_models": (
                        list(STABLE_AUDIO_MODELS.keys())
                        + list(STABLE_AUDIO_3_MODELS.keys())
                        + ["stable-audio-open-1.0"]
                    ),
                }
            )

        elif request.action == "status":
            any_sa3_generating = any(
                s._generating for s in stable_audio_3_generators.values()
            )
            any_sag_generating = any(
                s._generating for s in stable_audio_generators.values()
            )
            is_available = (
                generator is not None
                and not generator._generating
                and not any_sag_generating
                and not any_sa3_generating
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
                        "generating": generator._generating
                        or any_sa3_generating
                        or any_sag_generating,
                        "generating_llm": False,
                        **vram_info,
                    }
                )

        elif request.action == "generate":
            use_sa3_model = request.model in STABLE_AUDIO_3_MODELS
            use_stable_audio_model = request.model in STABLE_AUDIO_MODELS

            if use_sa3_model:
                sa3 = stable_audio_3_generators.get(request.model)
                if sa3 is None:
                    raise HTTPException(
                        status_code=503, detail=f"Model {request.model} not available"
                    )
                if sa3._generating:
                    raise HTTPException(
                        status_code=503, detail="Already generating — try again later"
                    )

                duration = max(MIN_DURATION, min(MAX_DURATION, request.duration or 30))
                try:
                    loop = asyncio.get_event_loop()
                    async with _vram_lock:
                        wav_bytes = await loop.run_in_executor(
                            None,
                            sa3.generate,
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
                            "X-Model": request.model,
                            "X-Duration": str(duration),
                            "X-Sample-Rate": str(TARGET_SAMPLE_RATE),
                            "X-Seed": str(request.seed),
                        },
                    )
                except Exception as e:
                    print(f"❌ SA3 error: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"SA3 generation failed: {str(e)}",
                    )

            elif use_stable_audio_model:
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

                try:
                    loop = asyncio.get_event_loop()
                    async with _vram_lock:
                        wav_bytes, snapped_bpm = await loop.run_in_executor(
                            None,
                            sag.generate,
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
                            "X-Model": request.model,
                            "X-BPM": str(request.bpm),
                            "X-Snapped-BPM": str(snapped_bpm),
                            "X-Bars": str(request.bars or 8),
                            "X-Key": sanitize_header(str(request.key or "")),
                            "X-Seed": str(request.seed),
                        },
                    )
                except Exception as e:
                    print(f"❌ {request.model} error: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"{request.model} generation failed: {str(e)}",
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
                    async with _vram_lock:
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
        detail=f"Unknown action '{action}'. Valid: health, status, generate",
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
