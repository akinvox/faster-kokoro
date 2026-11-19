"""The mel head: Kokoro decoder inputs -> 100-band log-mel spectrogram for Vocos."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLN(nn.Module):
    """LayerNorm whose scale and shift come from a style vector."""

    def __init__(self, dim: int, style_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.fc = nn.Linear(style_dim, 2 * dim)


class Block(nn.Module):
    """ConvNeXt block: depthwise k7 convolution, style AdaLN, MLP, layer scale."""

    def __init__(self, dim: int, mult: int, style_dim: int, scale: float):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = AdaLN(dim, style_dim)
        self.pw1 = nn.Linear(dim, mult * dim)
        self.pw2 = nn.Linear(mult * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), scale))


class MelHead(nn.Module):
    """Predicts a Vocos log-mel from Kokoro's decoder inputs.

    Each block also sees whole-utterance statistics of its own features: Kokoro's
    decoder normalises over the whole utterance, which local context cannot reproduce.
    """

    def __init__(self, d_in=513, dim=512, layers=16, mult=3, n_mels=100, style_dim=128, global_rank=64):
        super().__init__()
        self.embed = nn.Conv1d(d_in + 2, dim, 7, padding=3)  # text features + log-F0, voiced flag, energy
        self.norm0 = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList(Block(dim, mult, style_dim, 1.0 / layers) for _ in range(layers))
        self.global_proj = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(2 * dim, elementwise_affine=False),
                nn.Linear(2 * dim, global_rank),
                nn.GELU(),
                nn.Linear(global_rank, 2 * dim),
            )
            for _ in range(layers)
        )
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.out = nn.Linear(dim, n_mels)

    @staticmethod
    def _block(block: Block, proj: nn.Module, h, s, m, n):
        """One residual block. h [B, C, T], style s [B, 128], mask m [B, 1, T], valid length n [B, 1]."""
        d = block.dw(h)
        df, mf = d.float(), m.float()
        mu = (df * mf).sum(-1) / n.float()
        sd = (((df - mu[..., None]) ** 2 * mf).sum(-1) / n.float() + 1e-5).sqrt()
        g_glob, b_glob = proj(torch.cat([mu, sd], -1).to(d.dtype)).unsqueeze(1).chunk(2, -1)
        g_sty, b_sty = block.norm.fc(s).unsqueeze(1).chunk(2, -1)
        y = block.norm.norm(d.transpose(1, 2)) * (1 + g_sty + g_glob) + b_sty + b_glob
        y = block.pw2(F.gelu(block.pw1(y)))
        return (h + (block.gamma * y).transpose(1, 2)) * m

    def forward(self, z, s, mask=None):
        """z [B, 515, T] on the Vocos frame grid, s [B, 128] -> log-mel [B, 100, T].

        `mask` [B, T] marks valid frames when z is padded; statistics then use valid frames only.
        """
        if mask is None:
            mask = torch.ones(z.shape[0], z.shape[-1], device=z.device, dtype=z.dtype)
        m = mask[:, None, :].to(z.dtype)
        n = m.sum(-1).clamp_min(1.0)
        h = self.norm0(self.embed(z).transpose(1, 2)).transpose(1, 2) * m
        for block, proj in zip(self.blocks, self.global_proj):
            h = self._block(block, proj, h, s, m, n)
        return self.out(self.norm1(h.transpose(1, 2))).transpose(1, 2)
