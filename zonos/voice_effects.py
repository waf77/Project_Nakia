import json
from pathlib import Path
import torch
import torch.nn.functional as F

PRESETS_FILE = Path("presets.json")

def read_presets():
    if PRESETS_FILE.exists():
        try:
            return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def write_presets(presets):
    PRESETS_FILE.write_text(json.dumps(presets, indent=2), encoding="utf-8")

def fft_filter(wav, sr, low_cut=None, high_cut=None):
    n = wav.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1.0 / sr, device=wav.device)
    spec = torch.fft.rfft(wav)
    mask = torch.ones_like(freqs)
    if low_cut is not None and low_cut > 0:
        mask = mask * torch.sigmoid((freqs - low_cut) / max(40.0, low_cut * 0.15))
    if high_cut is not None and high_cut > 0:
        mask = mask * torch.sigmoid((high_cut - freqs) / max(100.0, high_cut * 0.15))
    return torch.fft.irfft(spec * mask, n=n)

def phase_vocoder_pitch_shift(wav, sr, semitones):
    rate = 2.0 ** (semitones / 12.0)
    if abs(rate - 1.0) < 1e-4:
        return wav
    n_fft = 1024
    hop = 256
    orig_len = wav.shape[-1]
    window = torch.hann_window(n_fft, device=wav.device)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    num_bins, num_frames = stft.shape
    time_steps = torch.arange(0, num_frames - 1, rate, device=wav.device)
    idx0 = torch.floor(time_steps).long()
    idx1 = (idx0 + 1).clamp(max=num_frames - 1)
    alpha = (time_steps - idx0.float()).unsqueeze(0)
    mag = (1.0 - alpha) * stft[:, idx0].abs() + alpha * stft[:, idx1].abs()
    phase_adv = (2.0 * torch.pi * hop / n_fft) * torch.arange(num_bins, device=wav.device).unsqueeze(1)
    dphi = torch.angle(stft[:, idx1]) - torch.angle(stft[:, idx0]) - phase_adv
    dphi = dphi - 2.0 * torch.pi * torch.round(dphi / (2.0 * torch.pi))
    phase_acc = torch.cumsum(phase_adv + dphi, dim=1)
    phase_acc = torch.cat([torch.angle(stft[:, :1]), phase_acc[:, :-1]], dim=1)
    stretched_wav = torch.istft(torch.polar(mag, phase_acc), n_fft=n_fft, hop_length=hop, window=window)
    return F.interpolate(stretched_wav.view(1, 1, -1), size=orig_len, mode="linear", align_corners=False).view(-1)

def apply_voice_effects(waveform, sr, pitch_shift, hardness, whisper_pct):
    wav = waveform.float()
    orig_peak = wav.abs().max().clamp(min=1e-6)
    if pitch_shift and float(pitch_shift) != 0.0:
        wav = phase_vocoder_pitch_shift(wav, sr, float(pitch_shift))
    h = float(hardness or 0.0)
    if h < 0:
        strength = abs(h) / 10.0
        cutoff = max(2800.0, 9500.0 - strength * 6000.0)
        soft_wav = fft_filter(wav, sr, high_cut=cutoff)
        wav = (1.0 - 0.65 * strength) * wav + (0.65 * strength) * soft_wav
    elif h > 0:
        strength = h / 10.0
        presence = fft_filter(wav, sr, low_cut=2200.0, high_cut=4800.0)
        wav = wav + presence * (strength * 0.85)
        drive = 1.0 + strength * 1.4
        wav = torch.tanh(wav * drive) / torch.tanh(torch.tensor(drive, device=wav.device))
    w = float(whisper_pct or 0.0) / 100.0
    if w > 0:
        win = max(1, int(sr * 0.015))
        env = F.avg_pool1d(wav.abs().view(1, 1, -1), kernel_size=win, stride=1, padding=win // 2).view(-1)[: wav.shape[-1]]
        noise = torch.randn_like(wav)
        breath = fft_filter(noise, sr, low_cut=900.0, high_cut=6500.0)
        breath = breath / breath.abs().max().clamp(min=1e-6)
        whisper_sig = breath * env * 2.4
        voiced = fft_filter(wav, sr, low_cut=220.0 + w * 450.0)
        wav = (1.0 - w * 0.85) * voiced + (w * 0.95) * whisper_sig
    new_peak = wav.abs().max().clamp(min=1e-6)
    return wav * (orig_peak / new_peak)
