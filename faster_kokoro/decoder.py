"""Drop-in replacement for Kokoro's waveform decoder."""
import torch.nn as nn

from .constants import KOKORO_HOP
from .features import decode_mel, head_inputs, vocos_frames
from .mel_head import MelHead


class FasterDecoder(nn.Module):
    """Mel head followed by Vocos.

    (asr [B, 512, T], F0 [B, 2T], N [B, 2T], style [B, 128]) -> waveform [B, 1, 600 T].
    """

    def __init__(self, head: MelHead, vocos):
        super().__init__()
        self.head, self.vocos = head, vocos

    def mel(self, asr, f0, n, s):
        num_samples = f0.shape[-1] * KOKORO_HOP
        z = head_inputs(
            asr.float(),
            f0.float().reshape(f0.shape[0], -1),
            n.float().reshape(n.shape[0], -1),
            vocos_frames(num_samples),
        )
        return self.head(z, s.float())

    def forward(self, asr, F0_curve, N, s):
        num_samples = F0_curve.shape[-1] * KOKORO_HOP
        return decode_mel(self.vocos, self.mel(asr, F0_curve, N, s), num_samples)[:, None]
