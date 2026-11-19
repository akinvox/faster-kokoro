"""Download and assemble the model: stock Kokoro-82M, the Faster Kokoro head and Vocos."""
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .constants import (CONFIG_FILE, HEAD_FILE, HEAD_REPO, KOKORO_REPO, KOKORO_REVISION,
                        VOCOS_REPO, VOCOS_REVISION)
from .decoder import FasterDecoder
from .mel_head import MelHead


def load_head(repo_id: str = HEAD_REPO, head_path=None, config_path=None) -> MelHead:
    """Load the mel head. `head_path` / `config_path` use local files instead of `repo_id`."""
    config_path = config_path or hf_hub_download(repo_id, CONFIG_FILE)
    head_path = head_path or hf_hub_download(repo_id, HEAD_FILE)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)["head"]
    head = MelHead(**config)
    head.load_state_dict(load_file(head_path), strict=True)
    return head.eval().requires_grad_(False)


def load_vocos():
    """Load the public Vocos 24 kHz mel vocoder at its pinned revision."""
    from vocos import Vocos

    config = hf_hub_download(VOCOS_REPO, "config.yaml", revision=VOCOS_REVISION)
    weights = hf_hub_download(VOCOS_REPO, "pytorch_model.bin", revision=VOCOS_REVISION)
    vocos = Vocos.from_hparams(config)
    vocos.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return vocos.eval().requires_grad_(False)


def load_model(device=None, repo_id: str = HEAD_REPO, head_path=None, config_path=None):
    """Stock Kokoro-82M with its decoder replaced by the mel head and Vocos.

    Returns a `kokoro.KModel`, usable with `kokoro.KPipeline(lang_code, model=model)`.
    """
    from kokoro import KModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = KModel(
        repo_id=KOKORO_REPO,
        config=hf_hub_download(KOKORO_REPO, "config.json", revision=KOKORO_REVISION),
        model=hf_hub_download(KOKORO_REPO, "kokoro-v1_0.pth", revision=KOKORO_REVISION),
    )
    model.decoder = FasterDecoder(load_head(repo_id, head_path, config_path), load_vocos())
    return model.to(device).eval().requires_grad_(False)
