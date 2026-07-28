from typing import Optional
from pydantic import BaseModel, field_validator, ConfigDict
from settings import (
    STABLE_AUDIO_MODELS,
    STABLE_AUDIO_3_MODELS,
    MIN_DURATION,
    MAX_DURATION,
)


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
        allowed = (
            {"stable-audio-open-1.0"}
            | set(STABLE_AUDIO_MODELS.keys())
            | set(STABLE_AUDIO_3_MODELS.keys())
        )
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

    @field_validator("prompt", mode="before")
    @classmethod
    def validate_prompt(cls, v, info):
        action = info.data.get("action") if info.data else None
        if action == "generate":
            if v is None or not str(v).strip():
                raise ValueError("prompt is required for generate action")
        return v

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v):
        if v is not None and not (0 <= v <= 2**31 - 1):
            raise ValueError("seed must be between 0 and 2147483647")
        return v
