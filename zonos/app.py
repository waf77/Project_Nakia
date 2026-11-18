import gradio as gr
import soundfile as sf
import torch
from transformers import AutoModel
from voice_effects import apply_voice_effects, read_presets, write_presets

model = AutoModel.from_pretrained("Zyphra/ZONOS3-Mini-hf", trust_remote_code=True).to("cuda")
raw_voices = model.voices()
available_voices = list(raw_voices.keys()) if isinstance(raw_voices, dict) else list(raw_voices)
default_voice = available_voices if available_voices else "Savannah"

def save_preset(name, language, voice, pitch, hardness, whisper, cfg, rate, fmax, loudness, temp, norm, seed):
    name = (name or "").strip()
    if not name:
        return gr.update(), "Enter a preset name first."
    presets = read_presets()
    presets[name] = {
        "language": language, "voice": voice, "pitch": pitch,
        "hardness": hardness, "whisper": whisper, "cfg": cfg,
        "rate": rate, "fmax": fmax, "loudness": loudness,
        "temp": temp, "norm": norm, "seed": seed,
    }
    write_presets(presets)
    return gr.update(choices=list(presets.keys()), value=name), f"Saved preset '{name}'."

def load_preset(name):
    presets = read_presets()
    if not name or name not in presets:
        return [gr.update()] * 12 + ["Select a saved preset first."]
    p = presets[name]
    return [
        p.get("language", "en_us"), p.get("voice", default_voice),
        p.get("pitch", 0.0), p.get("hardness", 0.0), p.get("whisper", 0.0),
        p.get("cfg", 1.0), p.get("rate", 0), p.get("fmax", 0),
        p.get("loudness", 0), p.get("temp", 0.0), p.get("norm", True),
        p.get("seed", 0), f"Loaded preset '{name}'.",
    ]

def delete_preset(name):
    presets = read_presets()
    if name in presets:
        del presets[name]
        write_presets(presets)
        keys = list(presets.keys())
        return gr.update(choices=keys, value=keys if keys else None), f"Deleted '{name}'."
    return gr.update(), "No preset selected."

def generate_speech(text, language, preset_voice, reference_audio, pitch_shift, hardness, whisper_pct, speaker_cfg_scale, speaking_rate, fmax, loudness, temperature, text_normalization, seed):
    kwargs = {
        "language": language,
        "speaker_cfg_scale": float(speaker_cfg_scale),
        "text_normalization": bool(text_normalization),
        "seed": int(seed),
        "condition_values": [
            float(loudness) if loudness != 0 else None,
            float(fmax) if fmax != 0 else None,
            float(speaking_rate) if speaking_rate != 0 else None,
        ],
    }
    if temperature and float(temperature) > 0:
        kwargs["temperature"] = float(temperature)
    if reference_audio is not None:
        audio_data, sample_rate = sf.read(reference_audio, dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        kwargs["speaker_audio"] = torch.from_numpy(audio_data)
        kwargs["speaker_sample_rate"] = int(sample_rate)
    else:
        kwargs["voice"] = preset_voice
    out = model.generate(text, **kwargs)
    waveform = apply_voice_effects(out.waveform.squeeze(), int(out.sampling_rate), pitch_shift, hardness, whisper_pct)
    return (int(out.sampling_rate), waveform.cpu().numpy())

initial_presets = list(read_presets().keys())

compact_css = """
.compact-controls .form { gap: 4px !important; }
.compact-controls .block { padding: 5px 8px !important; }
.compact-controls label span { font-size: 0.82rem !important; }
"""

with gr.Blocks(title="ZONOS3-Mini Local Studio") as demo:
    gr.Markdown("## ZONOS3-Mini Local Studio (RTX 5060)")
    with gr.Accordion("Saved Presets", open=True):
        with gr.Row():
            preset_dropdown = gr.Dropdown(choices=initial_presets, value=initial_presets if initial_presets else None, label="Select Saved Preset", scale=2)
            load_btn = gr.Button("Load Preset", scale=1)
            delete_btn = gr.Button("Delete", scale=1)
            preset_name_box = gr.Textbox(label="New Preset Name", placeholder="e.g. UK Narrator", scale=2)
            save_btn = gr.Button("Save Current Settings", variant="secondary", scale=1)
        preset_status = gr.Markdown("")
    with gr.Row():
        with gr.Column(scale=5):
            text_input = gr.Textbox(label="Text to Speak", lines=5, value="Hello! Welcome to my local Zonos Voice Studio running on an RTX 5060.")
            with gr.Row():
                voice_dropdown = gr.Dropdown(choices=available_voices, value=default_voice, label="Built-in Voice", allow_custom_value=True)
                language_dropdown = gr.Dropdown(choices=["en_us", "en_gb", "fr", "de", "es", "ja", "zh", "ko"], value="en_us", label="Language / Accent", allow_custom_value=True)
            ref_audio = gr.Audio(type="filepath", label="Optional: Reference Audio to Clone Voice (overrides built-in voice)")
        with gr.Column(scale=4, elem_classes=["compact-controls"]):
            with gr.Row():
                pitch_slider = gr.Slider(minimum=-12.0, maximum=12.0, value=0.0, step=0.5, label="Voice Pitch (-12 deep, +12 high)")
                hardness_slider = gr.Slider(minimum=-10.0, maximum=10.0, value=0.0, step=0.5, label="Hard / Soft Voice (-10 soft, +10 hard)")
            with gr.Row():
                whisper_slider = gr.Slider(minimum=0, maximum=100, value=0, step=5, label="Whisper / Talk (0% talk, 100% whisper)")
                speaking_rate_slider = gr.Slider(minimum=0, maximum=35, value=0, step=1, label="Speaking Rate (0=auto, 12=slow, 22=fast)")
            with gr.Row():
                fmax_slider = gr.Slider(minimum=0, maximum=24000, value=0, step=500, label="Audio Clarity / fmax (0=auto, 22050=studio)")
                loudness_slider = gr.Slider(minimum=-40, maximum=0, value=0, step=1, label="Loudness LUFS (0=auto, -16=broadcast)")
            with gr.Row():
                cfg_slider = gr.Slider(minimum=0.5, maximum=3.0, value=1.0, step=0.1, label="Voice Similarity (speaker_cfg_scale)")
                temp_slider = gr.Slider(minimum=0.0, maximum=1.5, value=0.0, step=0.05, label="Expressiveness / Temp (0=default)")
            generate_btn = gr.Button("Generate Speech", variant="primary")
            with gr.Row():
                norm_check = gr.Checkbox(label="Text Normalisation (spells out numbers)", value=True, scale=2)
                seed_num = gr.Number(label="Seed", value=0, scale=1)
    audio_out = gr.Audio(label="Generated Speech")
    settings_components = [language_dropdown, voice_dropdown, pitch_slider, hardness_slider, whisper_slider, cfg_slider, speaking_rate_slider, fmax_slider, loudness_slider, temp_slider, norm_check, seed_num]
    save_btn.click(fn=save_preset, inputs=[preset_name_box] + settings_components, outputs=[preset_dropdown, preset_status])
    load_btn.click(fn=load_preset, inputs=[preset_dropdown], outputs=settings_components + [preset_status])
    delete_btn.click(fn=delete_preset, inputs=[preset_dropdown], outputs=[preset_dropdown, preset_status])
    generate_btn.click(fn=generate_speech, inputs=[text_input, language_dropdown, voice_dropdown, ref_audio, pitch_slider, hardness_slider, whisper_slider, cfg_slider, speaking_rate_slider, fmax_slider, loudness_slider, temp_slider, norm_check, seed_num], outputs=audio_out)

demo.launch(inbrowser=True, css=compact_css)
