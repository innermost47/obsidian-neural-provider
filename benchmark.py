import argparse
import gc
import io
import random
import statistics
import time
from typing import Optional

import numpy as np
import soundfile as sf
import torch

SUPPORTED_MODELS = {
    "stable-audio-open-1.0": "stabilityai/stable-audio-open-1.0",
    "stable-audio-open-small": "stabilityai/stable-audio-open-small",
}

THRESHOLDS = {
    "stable-audio-open-1.0": 60,
    "stable-audio-open-small": 20,
}

TEST_PROMPT = "Dark ambient drone, slow evolving texture, deep bass, 140 BPM"
TEST_DURATION = 10
TARGET_SAMPLE_RATE = 44100
WARMUP_RUNS = 1
BENCH_RUNS = 3


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("❌ No CUDA GPU detected. CPU mode is not supported.")


def print_vram(label: str):
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM [{label}]: {used:.1f} / {total:.1f} GB")


def load_model(model_key: str, device: str):
    from diffusers import StableAudioPipeline

    model_id = SUPPORTED_MODELS[model_key]
    is_small = "small" in model_key

    print(f"\n⚡ Loading {model_id} on {device}...")
    t0 = time.time()

    if device == "cuda" and not is_small:
        from diffusers import (
            BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
            StableAudioDiTModel,
        )
        from transformers import (
            BitsAndBytesConfig as TransformersBitsAndBytesConfig,
            T5EncoderModel,
        )

        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            quantization_config=TransformersBitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=torch.float16,
        )
        transformer = StableAudioDiTModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=torch.float16,
        )
        pipeline = StableAudioPipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=torch.float16,
            device_map="balanced",
        )
    else:
        pipeline = StableAudioPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        pipeline = pipeline.to(device)

    sample_rate = pipeline.vae.sampling_rate
    elapsed = time.time() - t0
    print(f"✅ Loaded in {elapsed:.1f}s (sample rate: {sample_rate}Hz)")
    print_vram("after load")

    return pipeline, sample_rate


def unload_model(pipeline):
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print_vram("after unload")


def run_generation(pipeline, sample_rate: int, device: str, is_small: bool) -> float:
    num_inference_steps = 8 if is_small else 50
    cfg_scale = 1.0 if is_small else 7.0
    seed = random.randint(0, 2**31 - 1)
    gen = torch.Generator(device=device).manual_seed(seed)

    t0 = time.time()

    result = pipeline(
        TEST_PROMPT,
        negative_prompt="Low quality, distorted, noise",
        num_inference_steps=num_inference_steps,
        audio_end_in_s=TEST_DURATION,
        num_waveforms_per_prompt=1,
        generator=gen,
        guidance_scale=cfg_scale,
    )

    elapsed = time.time() - t0

    audio = result.audios[0].float().cpu().numpy()
    if sample_rate != TARGET_SAMPLE_RATE:
        import librosa

        if len(audio.shape) > 1 and audio.shape[0] == 2:
            audio = np.array(
                [
                    librosa.resample(
                        audio[0], orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
                    ),
                    librosa.resample(
                        audio[1], orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
                    ),
                ]
            )
        else:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
            )

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9
    if len(audio.shape) > 1 and audio.shape[0] == 2:
        audio = audio.T

    buf = io.BytesIO()
    sf.write(buf, audio, TARGET_SAMPLE_RATE, format="WAV")
    wav_size = buf.tell() / 1024

    print(f"    ✅ {elapsed:.1f}s — {wav_size:.0f} KB")
    return elapsed


def benchmark_model(model_key: str, runs: int, no_warmup: bool) -> Optional[dict]:
    device = get_device()
    is_small = "small" in model_key

    print(f"\n{'='*55}")
    print(f"  Benchmarking: {model_key}")
    print(f"{'='*55}")

    try:
        pipeline, sample_rate = load_model(model_key, device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    times = []

    if not no_warmup:
        print(f"\n🔥 Warmup (excluded from stats)...")
        try:
            run_generation(pipeline, sample_rate, device, is_small)
        except Exception as e:
            print(f"  ⚠️  Warmup failed: {e}")

    print(f"\n⏱️  Benchmark ({runs} runs)...")
    for i in range(runs):
        print(f"  Run {i+1}/{runs}...")
        try:
            t = run_generation(pipeline, sample_rate, device, is_small)
            times.append(t)
        except Exception as e:
            print(f"    ❌ Failed: {e}")

    print(f"\n🧹 Unloading model...")
    unload_model(pipeline)

    if not times:
        print("  ❌ No successful generation.")
        return None

    avg = statistics.mean(times)
    median = statistics.median(times)
    best = min(times)
    worst = max(times)
    threshold = THRESHOLDS.get(model_key, 60)
    eligible = avg <= threshold
    rtf = TEST_DURATION / avg

    return {
        "model_key": model_key,
        "runs": len(times),
        "avg": avg,
        "median": median,
        "best": best,
        "worst": worst,
        "rtf": rtf,
        "threshold": threshold,
        "eligible": eligible,
    }


def print_result(r: dict):
    print(f"\n  {'─'*45}")
    print(f"  Model     : {r['model_key']}")
    print(f"  Runs      : {r['runs']}")
    print(f"  Average   : {r['avg']:.1f}s")
    print(f"  Median    : {r['median']:.1f}s")
    print(f"  Best      : {r['best']:.1f}s")
    print(f"  Worst     : {r['worst']:.1f}s")
    print(f"  RTF       : x{r['rtf']:.2f}  (>x1.0 = faster than real-time)")
    print(f"  Threshold : {r['threshold']}s for {TEST_DURATION}s audio")
    print(f"  Eligible  : {'✅ YES' if r['eligible'] else '❌ NO — too slow'}")
    print(f"  {'─'*45}")


def main():
    parser = argparse.ArgumentParser(description="OBSIDIAN Neural Local GPU Benchmark")
    parser.add_argument(
        "--runs",
        type=int,
        default=BENCH_RUNS,
        help="Benchmark runs per model (default: 3)",
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup run")
    parser.add_argument(
        "--model",
        choices=list(SUPPORTED_MODELS.keys()) + ["all"],
        default="all",
        help="Model to benchmark (default: all)",
    )
    args = parser.parse_args()

    models = list(SUPPORTED_MODELS.keys()) if args.model == "all" else [args.model]

    print(f"\n{'='*55}")
    print(f"  OBSIDIAN Neural — Local GPU Benchmark")
    print(f"  Prompt   : {TEST_PROMPT[:50]}...")
    print(f"  Duration : {TEST_DURATION}s audio")
    print(f"  Runs     : {args.runs} per model (+{WARMUP_RUNS} warmup)")
    print(f"  Models   : {', '.join(models)}")
    print(f"{'='*55}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\n  GPU    : {props.name}")
        print(f"  VRAM   : {props.total_memory / 1024**3:.1f} GB")

    results = []
    for model_key in models:
        r = benchmark_model(model_key, args.runs, args.no_warmup)
        if r:
            results.append(r)

    # Summary
    print(f"\n\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")

    for r in results:
        print_result(r)

    print(f"\n{'='*55}")
    print(f"  VERDICT")
    print(f"{'='*55}")
    for r in results:
        status = "🟢 ELIGIBLE" if r["eligible"] else "🔴 NOT ELIGIBLE"
        print(f"  {r['model_key']:<35} {status}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
