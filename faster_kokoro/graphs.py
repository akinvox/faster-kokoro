"""CUDA-graph runtime: the same model and audio as the normal path, several times faster.

Every stage runs on inputs padded to a fixed set of sizes ("buckets") with exact masking.
Each bucket is recorded once as a CUDA graph (about 30 s, about 330 MB of GPU memory) and
then serves any sentence that fits; the few largest buckets are recorded on first use.
One request at a time per instance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import KOKORO_HOP, VOCOS_HOP
from .features import decode_mel, head_inputs

TOKEN_BUCKETS = (16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 448, 512)
FRAME_BUCKETS = tuple(sorted({int(round(16 * 1.2**k)) for k in range(26)}))  # 16 ... ~1520 frames (~38 s)
# Longer chunks (a low `speed` stretches a full 510-phoneme chunk well past 38 s). These are large, so each is
# recorded the first time a request needs it instead of up front: ~1830 ... ~3170 frames (~79 s).
LAZY_FRAME_BUCKETS = tuple(int(round(16 * 1.2**k)) for k in range(26, 30))


class BucketOverflow(ValueError):
    """The request is longer than the largest bucket."""


class MaskedBiLSTM(nn.Module):
    """Single-layer bidirectional LSTM that ignores padding after the first n steps."""

    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        assert lstm.bidirectional and lstm.num_layers == 1 and lstm.batch_first
        self.fw = nn.LSTM(lstm.input_size, lstm.hidden_size, 1, batch_first=True)
        self.bw = nn.LSTM(lstm.input_size, lstm.hidden_size, 1, batch_first=True)
        with torch.no_grad():
            for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
                getattr(self.fw, name).copy_(getattr(lstm, name))
                getattr(self.bw, name).copy_(getattr(lstm, name + "_reverse"))

    def forward(self, x, n):
        """x [B, T, C] padded after n valid steps (n: [B] long, on device)."""
        t = torch.arange(x.shape[1], device=x.device)[None, :]
        rev = torch.where(t < n[:, None], n[:, None] - 1 - t, t)  # reverse inside the valid part only
        forward = self.fw(x)[0]
        reversed_x = x.gather(1, rev[..., None].expand(-1, -1, x.shape[2]))
        backward = self.bw(reversed_x)[0].gather(1, rev[..., None].expand(-1, -1, self.bw.hidden_size))
        return torch.cat([forward, backward], -1)


def masked_instance_norm(norm: nn.InstanceNorm1d, x, m):
    """InstanceNorm1d (affine) over valid frames only. x [B, C, T], m [B, 1, T] float."""
    n = m.sum(-1, keepdim=True).clamp_min(1.0)
    mu = (x * m).sum(-1, keepdim=True) / n
    var = (((x - mu) ** 2) * m).sum(-1, keepdim=True) / n
    y = (x - mu) / torch.sqrt(var + norm.eps)
    if norm.affine:
        y = y * norm.weight[None, :, None] + norm.bias[None, :, None]
    return y


def adain(mod, x, s, m):
    gamma, beta = torch.chunk(mod.fc(s)[..., None], 2, dim=1)
    return (1 + gamma) * masked_instance_norm(mod.norm, x, m) + beta


def adain_resblk(blk, x, s, m):
    """Kokoro's AdainResBlk1d with masks. m is the mask at the input rate; returns (out, mask at output rate)."""
    up = blk.upsample_type != "none"
    h = blk.actv(adain(blk.norm1, x, s, m)) * m
    m2 = F.interpolate(m, scale_factor=2, mode="nearest") if up else m
    h = blk.pool(h) * m2  # the depthwise ConvTranspose sees zeros past the end
    h = blk.conv1(h) * m2
    h = blk.actv(adain(blk.norm2, h, s, m2)) * m2
    h = blk.conv2(h)
    shortcut = blk.upsample(x)
    if blk.learned_sc:
        shortcut = blk.conv1x1(shortcut)
    return (h + shortcut) * (2.0**-0.5) * m2, m2  # = Kokoro's rsqrt(2) in float32; no tensor creation (graph-safe)


class BucketedModel(nn.Module):
    """Padded, mask-exact inference stages of a Faster Kokoro `KModel`.

    Takes ownership of the model: weight norm is folded in place, so pass a dedicated instance.
    """

    def __init__(self, model):
        super().__init__()
        m = model.eval().requires_grad_(False)
        for mod in m.modules():
            if hasattr(mod, "weight_g") and hasattr(mod, "weight_v"):
                torch.nn.utils.remove_weight_norm(mod)
        m.bert.attn_implementation = "eager"
        self.m = m
        p = m.predictor
        self.dur_lstms = nn.ModuleList(MaskedBiLSTM(b) if isinstance(b, nn.LSTM) else b for b in p.text_encoder.lstms)
        self.dur_lstm = MaskedBiLSTM(p.lstm)
        self.shared = MaskedBiLSTM(p.shared)
        self.text_lstm = MaskedBiLSTM(m.text_encoder.lstm)

    def prefix(self, ids, length, ref):
        """ids [1, Tb] padded with 0 after `length` tokens.

        Returns d [1, Tb, C], raw durations [1, Tb] (0 past length), text features [1, 512, Tb].
        """
        m = self.m
        T = ids.shape[1]
        valid = torch.arange(T, device=ids.device)[None, :] < length
        vf = valid.float()
        style = ref[:, 128:]
        x = m.bert_encoder(m.bert(ids, attention_mask=valid.int()))
        sx = style[:, None, :].expand(-1, T, -1)
        x = torch.cat([x, sx], -1) * vf[..., None]
        for blk in self.dur_lstms:
            if isinstance(blk, MaskedBiLSTM):
                x = blk(x, length.view(1))
            else:
                x = torch.cat([blk(x, style), sx], -1) * vf[..., None]  # AdaLayerNorm: [B, T, C] in and out
        d = x * vf[..., None]
        h = self.dur_lstm(d, length.view(1))
        raw = torch.sigmoid(m.predictor.duration_proj(h)).sum(-1) * vf
        te = m.text_encoder
        t = te.embedding(ids).transpose(1, 2) * vf[:, None, :]
        for c in te.cnn:
            t = c(t) * vf[:, None, :]
        t = self.text_lstm(t.transpose(1, 2), length.view(1)).transpose(1, 2) * vf[:, None, :]
        return d, raw, t

    def controls(self, en, frames, s):
        """en [1, C, Fb] padded after `frames`. Returns F0 and N, each [1, 2 Fb] (0 past 2 frames)."""
        mf = (torch.arange(en.shape[-1], device=en.device)[None, None, :] < frames).float()
        x = self.shared(en.transpose(1, 2), frames.view(1)).transpose(1, 2) * mf
        predictor = self.m.predictor
        out = []
        for blocks, proj in ((predictor.F0, predictor.F0_proj), (predictor.N, predictor.N_proj)):
            h, mask = x, mf
            for blk in blocks:
                h, mask = adain_resblk(blk, h, s, mask)
            out.append((proj(h) * mask).squeeze(1))
        return out[0], out[1]

    def head(self, asr, f0, n, frames, s_dec):
        """asr [1, 512, Fb], f0 / n [1, 2 Fb] padded. Returns mel [1, 100, Mb]; the valid part is the true Vocos frame count."""
        mb = (2 * asr.shape[-1] * KOKORO_HOP) // VOCOS_HOP + 1
        m_true = (2 * frames * KOKORO_HOP) // VOCOS_HOP + 1
        z = head_inputs(asr, f0, n, mb, frames)
        mask = (torch.arange(mb, device=asr.device)[None, :] < m_true).float()
        return self.m.decoder.head(z * mask[:, None, :], s_dec, mask)


class _Capture:
    """One CUDA graph for fixed input shapes: inputs are copied into captured buffers, then replayed."""

    def __init__(self, fn, args):
        self.buffers = tuple(x.clone() for x in args)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                fn(*self.buffers)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = fn(*self.buffers)

    def __call__(self, *args):
        for buffer, arg in zip(self.buffers, args):
            buffer.copy_(arg)
        self.graph.replay()
        return self.output


def _pick(n: int, buckets) -> int:
    for size in buckets:
        if size >= n:
            return size
    raise BucketOverflow(f"length {n} exceeds largest bucket {buckets[-1]}")


class BucketGraphRuntime:
    """Records every bucket once (`prepare`); each request then replays the graphs and runs Vocos eagerly on the exact mel."""

    def __init__(self, bucketed: BucketedModel, device: str = "cuda"):
        self.b, self.dev = bucketed, device
        self.prefix, self.controls, self.head = {}, {}, {}

    @torch.no_grad()
    def prepare(self, ref):
        dev = self.dev
        style, s_dec = ref[:, 128:], ref[:, :128]
        for size in TOKEN_BUCKETS:
            ids = torch.zeros(1, size, dtype=torch.long, device=dev)
            self.prefix[size] = _Capture(self.b.prefix, (ids, torch.tensor(min(size, 8), device=dev), ref))
        for fb in FRAME_BUCKETS:
            self._capture_frames(fb, style, s_dec)
        torch.cuda.synchronize()

    def _capture_frames(self, fb: int, style, s_dec):
        """Record the pitch/energy and mel-head graphs for one frame bucket."""
        dev = self.dev
        predictor_encoder = self.b.m.predictor.text_encoder
        channels = predictor_encoder.d_model + predictor_encoder.sty_dim
        frames = torch.tensor(min(fb, 12), device=dev)
        self.controls[fb] = _Capture(self.b.controls, (torch.zeros(1, channels, fb, device=dev), frames, style))
        self.head[fb] = _Capture(self.b.head, (
            torch.zeros(1, 512, fb, device=dev), torch.zeros(1, 2 * fb, device=dev),
            torch.zeros(1, 2 * fb, device=dev), frames, s_dec))

    @torch.no_grad()
    def infer(self, ids, ref, speed: float = 1.0):
        """ids [1, L] -> waveform [num_samples]. Raises BucketOverflow if the request exceeds the largest bucket."""
        dev = self.dev
        length = ids.shape[1]
        tokens = _pick(length, TOKEN_BUCKETS)
        d, raw, text = self.prefix[tokens](F.pad(ids, (0, tokens - length)), torch.tensor(length, device=dev), ref)
        dur = (raw[0, :length] / speed).round().clamp(min=1).long()
        frames = int(dur.sum())  # the one unavoidable sync
        idx = torch.repeat_interleave(torch.arange(length, device=dev), dur, output_size=frames)
        fb = _pick(frames, FRAME_BUCKETS + LAZY_FRAME_BUCKETS)
        if fb not in self.controls:  # a lazy bucket, first use
            self._capture_frames(fb, ref[:, 128:], ref[:, :128])
        idx = F.pad(idx, (0, fb - frames))
        en = d.transpose(1, 2).index_select(2, idx)
        asr = text.index_select(2, idx)
        true_frames = torch.tensor(frames, device=dev)
        f0, n = self.controls[fb](en, true_frames, ref[:, 128:])
        mel = self.head[fb](asr, f0, n, true_frames, ref[:, :128])
        num_samples = 2 * frames * KOKORO_HOP
        m_true = num_samples // VOCOS_HOP + 1
        return decode_mel(self.b.m.decoder.vocos, mel[..., :m_true], num_samples).squeeze()
