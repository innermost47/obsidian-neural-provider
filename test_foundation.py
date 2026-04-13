import json
import os
import time
import math
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_CONFIG_PATH = os.environ.get("FOUNDATION_1_CONFIG_FILE")
MODEL_CKPT_PATH = os.environ.get("FOUNDATION_1_CKPT_PATH")
OUTPUT_DIR = "./benchmark_results"
STEPS = 75
CFG_SCALE = 7.0
SEED = 42
BEATS_PER_BAR = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


PROMPTS = [
    {
        "name": "bass_acid_techno",
        "prompt": "Bass, FM Bass, Acid, Gritty, Dark, Thick, Sub Bass, Bassline, Pitch Bend, 303, Medium Distortion, Medium Reverb",
        "bars": 8,
        "bpm": 140,
        "key": "A minor",
        "negative": "melodic, bright, clean, pads",
    },
    {
        "name": "synth_lead_arp",
        "prompt": "Synth Lead, High Saw, Supersaw, Warm, Wide, Bright, Crisp, Arp, Fast Speed, Complex, Medium Reverb, Low Delay",
        "bars": 8,
        "bpm": 128,
        "key": "F minor",
        "negative": "bass, dark, slow, pad",
    },
    {
        "name": "pad_atmospheric",
        "prompt": "Synth, Pad, Atmosphere, Dreamy, Wide, Airy, Soft, Warm, Chord Progression, Rising, Sustained, High Reverb, Stereo Delay",
        "bars": 8,
        "bpm": 110,
        "key": "D major",
        "negative": "distortion, gritty, harsh, percussion",
    },
    {
        "name": "violin_epic",
        "prompt": "Violin, Bowed Strings, Rich, Full, Warm, Intimate, Epic, Rising, Complex, Arp, Fast Speed, Medium Reverb",
        "bars": 8,
        "bpm": 130,
        "key": "D minor",
        "negative": "synthetic, digital, electronic",
    },
    {
        "name": "reese_bass_dnb",
        "prompt": "Bass, Reese Bass, Dark, Gritty, Thick, Deep, Buzzy, Growl, Rolling, Bassline, Low Distortion, Medium Reverb",
        "bars": 4,
        "bpm": 150,
        "key": "G# minor",
        "negative": "melody, bright, clean, pad",
    },
    {
        "name": "rhodes_jazz",
        "prompt": "Keys, Rhodes Piano, Warm, Soft, Analog, Vintage, Round, Chord Progression, Sustained, Low Reverb, Mono Delay",
        "bars": 8,
        "bpm": 120,
        "key": "C major",
        "negative": "distortion, harsh, digital, gritty",
    },
    {
        "name": "pluck_melodic",
        "prompt": "Plucked Strings, Harp, Pluck, Sparkly, Bright, Airy, Glassy, Melody, Top Melody, Catchy, Fast Speed, Medium Reverb",
        "bars": 4,
        "bpm": 120,
        "key": "B minor",
        "negative": "bass, dark, distortion",
    },
    {
        "name": "chiptune_lead",
        "prompt": "Chiptune, Synth Lead, Pulse Wave, Bitcrushed, Retro, Bright, Simple, Melody, Catchy, Medium Reverb",
        "bars": 8,
        "bpm": 128,
        "key": "D minor",
        "negative": "warm, analog, acoustic, soft",
    },
    {
        "name": "trumpet_brass",
        "prompt": "Trumpet, Brass, Warm, Smooth, Silky, Full, Rich, Complex Arp, Melody, Rising, High Reverb, Low Distortion",
        "bars": 8,
        "bpm": 130,
        "key": "C minor",
        "negative": "electronic, synthetic, digital",
    },
    {
        "name": "kalimba_ambient",
        "prompt": "Kalimba, Mallet, Metallic, Sparkly, Wide, Bright, Airy, Thick, Alternating, Chord Progression, Atmosphere, Spacey, Medium Reverb, Overdriven",
        "bars": 8,
        "bpm": 120,
        "key": "B minor",
        "negative": "bass, dark, distortion, gritty",
    },
]


def pick_preferred_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_model(model_config_path, model_ckpt_path, device, preferred_dtype):
    print(f"\n🔧 Loading the model on {device} as {preferred_dtype}...")
    t0 = time.time()

    with open(model_config_path) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)
    state_dict = load_ckpt_state_dict(model_ckpt_path)
    model.load_state_dict(state_dict)
    model.to(device).eval().requires_grad_(False)

    if device.type in ("cuda", "mps"):
        model.to(preferred_dtype)

    elapsed = time.time() - t0
    print(f"✅ Model loaded in {elapsed:.2f}s")
    return model, model_config


def clip_samples_from_bars_bpm(bars, bpm, sample_rate, beats_per_bar=BEATS_PER_BAR):
    clip_seconds = (60.0 / float(bpm)) * float(beats_per_bar) * float(bars)
    return int(round(clip_seconds * sample_rate)), clip_seconds


def target_samples(clip_samples, sample_rate, min_input_length):
    seconds_int = int(math.ceil(clip_samples / sample_rate))
    target = int(seconds_int * sample_rate)
    if isinstance(min_input_length, int) and min_input_length > 0:
        if target % min_input_length != 0:
            target += min_input_length - (target % min_input_length)
    return seconds_int, target


def nearest_supported_bpm(bpm: float) -> int:
    supported = [100, 110, 120, 128, 130, 140, 150]
    return min(supported, key=lambda x: abs(x - bpm))


def run_generation(model, model_config, entry, device):
    sample_rate = model_config["sample_rate"]
    bpm = nearest_supported_bpm(entry["bpm"])
    bars = entry["bars"]
    key = entry.get("key", "")
    negative = entry.get("negative", "")

    clip_samples, _ = clip_samples_from_bars_bpm(bars, bpm, sample_rate)
    min_input = getattr(model, "min_input_length", None)
    seconds_int, input_sample_size = target_samples(
        clip_samples, sample_rate, min_input
    )

    full_prompt = f"{entry['prompt']}, {key}, {bars} Bars, {bpm} BPM"
    print(f"   📝 Prompt : {full_prompt}")

    conditioning = [
        {
            "prompt": full_prompt,
            "seconds_start": 0.0,
            "seconds_total": float(seconds_int),
        }
    ]
    negative_conditioning = (
        [
            {
                "prompt": negative,
                "seconds_start": 0.0,
                "seconds_total": float(seconds_int),
            }
        ]
        if negative
        else None
    )

    t0 = time.time()

    audio = generate_diffusion_cond(
        model,
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        steps=STEPS,
        cfg_scale=CFG_SCALE,
        batch_size=1,
        sample_size=int(input_sample_size),
        sample_rate=sample_rate,
        seed=SEED,
        device=device,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=500,
        scale_phi=0.0,
    )

    elapsed = time.time() - t0

    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).clamp(-1, 1)
    end = min(int(audio.shape[-1]), int(clip_samples))
    audio = audio[:, : max(1, end)].contiguous()

    fade_len = min(int(round(0.015 * sample_rate)), audio.shape[-1])
    if fade_len > 1:
        ramp = torch.linspace(
            1.0, 0.0, steps=fade_len, device=audio.device, dtype=audio.dtype
        )
        audio[:, -fade_len:] *= ramp

    wav_i16 = (audio * 32767.0).to(torch.int16).cpu()
    return wav_i16, sample_rate, elapsed


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    preferred_dtype = pick_preferred_dtype(device)
    print(f"🖥️  Device: {device} | dtype: {preferred_dtype}")

    model, model_config = load_model(
        MODEL_CONFIG_PATH, MODEL_CKPT_PATH, device, preferred_dtype
    )
    sample_rate = model_config["sample_rate"]
    print(f"🎵 Sample rate of the model: {sample_rate} Hz")

    results = []
    total_t0 = time.time()

    for i, entry in enumerate(PROMPTS):
        name = entry["name"]
        print(f"\n{'='*60}")
        print(
            f"[{i+1}/{len(PROMPTS)}] {name} | {entry['bars']} bars @ {entry['bpm']} BPM | {entry.get('key', '')}"
        )

        try:
            wav, sr, elapsed = run_generation(model, model_config, entry, device)

            out_path = os.path.join(OUTPUT_DIR, f"{i+1:02d}_{name}.wav")
            torchaudio.save(out_path, wav, sr)

            audio_duration = (entry["bars"] * 4 * 60.0) / entry["bpm"]
            rtf = elapsed / audio_duration

            print(
                f" ✅ {elapsed:.2f}s generation | audio duration: {audio_duration:.1f}s | RTF: {rtf:.2f}x"
            )
            print(f"   💾 Saved: {out_path}")

            results.append(
                {
                    "name": name,
                    "bars": entry["bars"],
                    "bpm": entry["bpm"],
                    "key": entry.get("key", ""),
                    "generation_time_s": round(elapsed, 2),
                    "audio_duration_s": round(audio_duration, 2),
                    "rtf": round(rtf, 2),
                    "status": "ok",
                }
            )

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({"name": name, "status": "error", "error": str(e)})

    total_elapsed = time.time() - total_t0

    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK REPORT — {len(PROMPTS)} prompts")
    print(f"{'='*60}")
    print(
        f"{'Prompt':<30} {'Bars':>5} {'BPM':>5} {'Génération':>12} {'Audio':>8} {'RTF':>6} {'Status':>8}"
    )
    print("-" * 80)
    for r in results:
        if r["status"] == "ok":
            print(
                f"{r['name']:<30} {r['bars']:>5} {r['bpm']:>5} "
                f"{r['generation_time_s']:>10.2f}s {r['audio_duration_s']:>6.1f}s "
                f"{r['rtf']:>6.2f}x {'✅':>8}"
            )
        else:
            print(f"{r['name']:<30} {'':>5} {'':>5} {'':>12} {'':>8} {'':>6} {'❌':>8}")

    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        avg_gen = sum(r["generation_time_s"] for r in ok_results) / len(ok_results)
        avg_rtf = sum(r["rtf"] for r in ok_results) / len(ok_results)
        print("-" * 80)
        print(
            f"{'MOYENNE':<30} {'':>5} {'':>5} {avg_gen:>10.2f}s {'':>8} {avg_rtf:>6.2f}x"
        )

    print(f"\n⏱️  Temps total : {total_elapsed:.1f}s")
    print(f"📁 Fichiers WAV dans : {os.path.abspath(OUTPUT_DIR)}/")

    json_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"📄 Résultats JSON : {json_path}")


if __name__ == "__main__":
    main()
