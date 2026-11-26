"""Public text-to-speech interface."""
import warnings
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import torch

from .constants import HEAD_REPO, KOKORO_REPO, SAMPLE_RATE
from .graphs import BucketedModel, BucketGraphRuntime, BucketOverflow
from .loading import load_model


class FasterKokoro:
    """Kokoro-82M with a faster waveform decoder. Same voices and languages as Kokoro.

    Args:
        device: "cuda" or "cpu". Defaults to CUDA when available.
        cuda_graphs: record CUDA graphs on first use (CUDA only; about 30 s once, about 330 MB
            of GPU memory) for several times faster synthesis. Output is unchanged.
        repo_id: Hugging Face repository holding the mel head.
        head_path, config_path: use local head files instead of downloading them.
    """

    sample_rate = SAMPLE_RATE

    def __init__(self, device: Optional[str] = None, cuda_graphs: bool = False, repo_id: str = HEAD_REPO,
                 head_path=None, config_path=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if cuda_graphs and not device.startswith("cuda"):
            raise ValueError("cuda_graphs requires a CUDA device")
        self.device = device
        self.model = load_model(device, repo_id, head_path, config_path)
        self._runtime = BucketGraphRuntime(BucketedModel(self.model).to(device), device) if cuda_graphs else None
        self._runtime_ready = False
        self._overflow_warned = False
        self._pipelines = {}

    def _pipeline(self, lang_code: str):
        """Text-to-phoneme pipeline for one language (no model attached)."""
        from kokoro import KPipeline

        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=KOKORO_REPO, model=False)
        return self._pipelines[lang_code]

    def _waveform(self, phonemes: str, pack, speed: float) -> torch.Tensor:
        vocab = self.model.vocab
        ids = torch.tensor([[0, *(vocab[p] for p in phonemes if p in vocab), 0]], device=self.device)
        ref = pack[len(phonemes) - 1]
        with torch.inference_mode():
            if self._runtime is not None:
                if not self._runtime_ready:
                    self._runtime.prepare(ref)
                    self._runtime_ready = True
                try:
                    return self._runtime.infer(ids, ref, speed).float().cpu()
                except BucketOverflow as overflow:  # longer than the largest graph bucket: normal path
                    if not self._overflow_warned:
                        self._overflow_warned = True
                        warnings.warn(
                            f"A chunk is too long for the CUDA graphs ({overflow}); it and any other such "
                            "chunk use the slower normal path. This usually means a low `speed`.",
                            RuntimeWarning, stacklevel=2)
            return self.model.forward_with_tokens(ids, ref, speed)[0].squeeze().float().cpu()

    def warmup(self, voice: str = "af_heart") -> None:
        """Record the CUDA graphs now instead of on the first request. No effect without `cuda_graphs`."""
        if self._runtime is not None and not self._runtime_ready:
            pipeline = self._pipeline(voice[0])
            self._waveform("həlˈO", pipeline.load_voice(voice).to(self.device), 1.0)

    def stream(self, text: str, voice: str = "af_heart", speed: float = 1.0,
               lang_code: Optional[str] = None) -> Iterator[np.ndarray]:
        """Yield 24 kHz float32 audio chunk by chunk (one per sentence-sized piece of `text`).

        The language comes from the first letter of the voice name ("af_heart" -> American
        English) unless `lang_code` is given.
        """
        pipeline = self._pipeline(lang_code or voice[0])
        pack = pipeline.load_voice(voice).to(self.device)
        for result in pipeline(text):
            if result.phonemes:
                yield self._waveform(result.phonemes, pack, speed).numpy()

    def __call__(self, text: str, voice: str = "af_heart", speed: float = 1.0,
                 lang_code: Optional[str] = None) -> np.ndarray:
        """Synthesize `text` to a 24 kHz float32 array."""
        chunks = list(self.stream(text, voice, speed, lang_code))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def save(self, text: str, path: Union[str, Path], voice: str = "af_heart", speed: float = 1.0,
             lang_code: Optional[str] = None) -> np.ndarray:
        """Synthesize `text`, write a 16-bit WAV to `path` and return the audio."""
        import soundfile as sf

        audio = self(text, voice, speed, lang_code)
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
        return audio
