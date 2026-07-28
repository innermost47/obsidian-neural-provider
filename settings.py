import os
from dotenv import load_dotenv

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
SA3_KEEP_IN_RAM: bool = os.getenv("SA3_KEEP_IN_RAM", "true").lower() == "true"
STABLE_AUDIO_MODELS = {
    "foundation-1": (
        "RoyalCities/Foundation-1",
        "Foundation_1.safetensors",
        "model_config.json",
    ),
    "audialab-edm-elements": (
        "innermost47/obsidian-neural-models",
        "audialab-edm-elements.safetensors",
        "audialab-edm-elements_model_config.json",
    ),
    "rc-infinite-pianos": (
        "innermost47/obsidian-neural-models",
        "rc-infinite-pianos.safetensors",
        "rc-infinite-pianos_model_config.json",
    ),
    "rc-vocal-textures": (
        "innermost47/obsidian-neural-models",
        "rc-vocal-textures.safetensors",
        "rc-vocal-textures_model_config.json",
    ),
    "sao-instrumental": (
        "innermost47/obsidian-neural-models",
        "sao-instrumental.safetensors",
        "sao-instrumental_model_config.json",
    ),
    "stablebeat": (
        "innermost47/obsidian-neural-models",
        "stablebeat.safetensors",
        "stablebeat_model_config.json",
    ),
    "gluten-v1": (
        "innermost47/obsidian-neural-models",
        "gluten-v1.safetensors",
        "gluten-v1_model_config.json",
    ),
}

STABLE_AUDIO_3_MODELS = {
    "stable-audio-3-medium": "stabilityai/stable-audio-3-medium",
}

SUPPORTED_BPM = [100, 110, 120, 128, 130, 140, 150]
MAX_DURATION = 30
MIN_DURATION = 2
TARGET_SAMPLE_RATE = 44100

FORCED_75_STEPS_MODELS = {
    "foundation-1",
    "audialab-edm-elements",
    "rc-infinite-pianos",
    "rc-vocal-textures",
}
