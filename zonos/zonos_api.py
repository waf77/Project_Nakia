import io
import json
from pathlib import Path
import re
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import numpy as np
from pydantic import BaseModel
import soundfile as sf
import torch
import uvicorn
from transformers import AutoModel
from voice_effects import apply_voice_effects, read_presets

app = FastAPI(title="ZONOS3-Mini OpenAI-Compatible TTS API")

print("Loading Zyphra/ZONOS3-Mini-hf onto CUDA...")
model = AutoModel.from_pretrained("Zyphra/ZONOS3-Mini-hf", trust_remote_code=True).to("cuda")
raw_voices = model.voices()
available_voices = list(raw_voices.keys()) if isinstance(raw_voices, dict) else list(raw_voices)
default_voice = available_voices[0] if available_voices else "Savannah"
print(f"Model loaded! Available built-in voices: {available_voices}")


class SpeechRequest(BaseModel):
    model: str = "Zyphra/ZONOS3-Mini-hf"
    input: str
    voice: str = "Savannah"
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    presets = read_presets()

    # Default settings
    language = "en_us"
    preset_voice = req.voice if req.voice in available_voices else default_voice
    reference_audio = None
    pitch = 0.0
    hardness = 0.0
    whisper = 0.0
    cfg = 1.0
    rate = 0
    fmax = 0
    loudness = 0
    temp = 0.0
    norm = True
    seed = 0

    # If the requested 'voice' matches a saved preset name in presets.json, load its settings
    if req.voice in presets:
        p = presets[req.voice]
        language = p.get("language", "en_us")
        preset_voice = p.get("voice", default_voice)
        reference_audio = p.get("reference_audio", None)
        pitch = p.get("pitch", 0.0)
        hardness = p.get("hardness", 0.0)
        whisper = p.get("whisper", 0.0)
        cfg = p.get("cfg", 1.0)
        rate = p.get("rate", 0)
        fmax = p.get("fmax", 0)
        loudness = p.get("loudness", 0)
        temp = p.get("temp", 0.0)
        norm = p.get("norm", True)
        seed = p.get("seed", 0)
    elif Path(req.voice).exists():
        # Allows passing a direct path to a reference .wav file as the 'voice' field
        reference_audio = req.voice

    effective_rate = float(rate)
    if effective_rate == 0 and req.speed and req.speed != 1.0:
        effective_rate = 15.0 * float(req.speed)

    kwargs = {
        "language": language,
        "speaker_cfg_scale": float(cfg),
        "text_normalization": bool(norm),
        "seed": int(seed),
        "condition_values": [
            float(loudness) if loudness != 0 else None,
            float(fmax) if fmax != 0 else None,
            effective_rate if effective_rate != 0 else None,
        ],
    }

    if temp and float(temp) > 0:
        kwargs["temperature"] = float(temp)

    if reference_audio and Path(reference_audio).exists():
        audio_data, sample_rate = sf.read(reference_audio, dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        kwargs["speaker_audio"] = torch.from_numpy(audio_data)
        kwargs["speaker_sample_rate"] = int(sample_rate)
    else:
        kwargs["voice"] = preset_voice

    clean_text = re.sub(r"\*+", "", req.input).strip()
    if not clean_text:
        clean_text = req.input

    with torch.no_grad():
        out = model.generate(clean_text, **kwargs)

    # Apply pitch, hardness, and whisper post-processing effects
    waveform = apply_voice_effects(
        out.waveform.squeeze(),
        int(out.sampling_rate),
        pitch,
        hardness,
        whisper,
    )
    audio_np = waveform.cpu().numpy()

    # Add 0.35 seconds of clean silence at the end so the final word never clips off
    pad_samples = int(out.sampling_rate * 0.35)
    audio_np = np.concatenate([audio_np, np.zeros(pad_samples, dtype=audio_np.dtype)])

    buffer = io.BytesIO()
    sf.write(buffer, audio_np, out.sampling_rate, format="WAV")
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005)
