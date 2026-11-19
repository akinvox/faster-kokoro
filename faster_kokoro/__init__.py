"""AkinVox Faster Kokoro: Kokoro-82M with a faster waveform decoder."""
from .api import FasterKokoro
from .loading import load_model

__all__ = ["FasterKokoro", "load_model"]
__version__ = "1.0.0"
