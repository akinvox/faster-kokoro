"""Mapping Kokoro's decoder inputs onto the Vocos frame grid, and Vocos output back to a waveform."""
import torch
import torch.nn.functional as F

from .constants import KOKORO_HOP, VOCOS_HOP


def interpolate(z, hop: int, offset: float, num_frames: int, length=None):
    """Linearly sample z [..., L] at Vocos frames 0 .. num_frames - 1.

    Frame k of z is centred at `hop * k + offset` samples; Vocos frame j at `VOCOS_HOP * j`.
    `length` (int, or an integer tensor for the graph path) is the true number of valid frames when z is padded;
    positions are clamped to it. Defaults to the full size of z.
    """
    length = z.shape[-1] if length is None else length
    pos = (torch.arange(num_frames, device=z.device, dtype=torch.float32) * VOCOS_HOP - offset) / hop
    if torch.is_tensor(length):  # padded input (graph path): clamp on device, no host sync
        last = length - 1
        pos = torch.minimum(pos.clamp_min(0), last.float())
        lo = torch.minimum(pos.floor().long(), (last - 1).clamp_min(0))
        hi = torch.minimum(lo + 1, last)
    else:
        pos = pos.clamp(0, length - 1)
        lo = pos.floor().long().clamp(max=max(length - 2, 0))
        hi = (lo + 1).clamp(max=length - 1)
    w = pos - lo
    return z[..., lo] * (1 - w) + z[..., hi] * w


def head_inputs(asr, f0, n, num_frames: int, frames=None):
    """Kokoro decoder inputs -> [B, 515, num_frames] mel-head input.

    asr [B, 512, T] text features (40 Hz); f0, n [B, 2T] pitch and energy (80 Hz) become
    log-F0, voiced flag and energy. `frames` is the true T when the inputs are padded.
    """
    voiced = (f0 > 10).float()
    c = torch.stack([(torch.log(f0.clamp_min(10.0)) - 5.0) * voiced, voiced, n], 1)
    asr_length = None if frames is None else frames
    c_length = None if frames is None else 2 * frames
    return torch.cat(
        [
            interpolate(asr, 2 * KOKORO_HOP, KOKORO_HOP, num_frames, asr_length),
            interpolate(c, KOKORO_HOP, KOKORO_HOP / 2, num_frames, c_length),
        ],
        1,
    )


def vocos_frames(num_samples: int) -> int:
    """Number of Vocos mel frames that cover `num_samples` waveform samples."""
    return num_samples // VOCOS_HOP + 1


def decode_mel(vocos, mel, num_samples: int):
    """Vocos-decode a log-mel to exactly `num_samples` samples (trimmed or zero-padded). -> [B, num_samples]."""
    y = vocos.decode(mel)[..., :num_samples]
    return F.pad(y, (0, max(0, num_samples - y.shape[-1])))
