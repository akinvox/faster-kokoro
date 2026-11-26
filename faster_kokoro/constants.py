"""Pinned upstream sources and audio-grid constants."""

HEAD_REPO = "AKinvox/faster-kokoro-v1"
HEAD_REVISION = "db715c1d59f3e54b06ed1fd413371442de6b1f40"
HEAD_FILE = "faster_kokoro_head.safetensors"
CONFIG_FILE = "config.json"

KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
VOCOS_REPO = "charactr/vocos-mel-24khz"
VOCOS_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"

SAMPLE_RATE = 24000
KOKORO_HOP = 300  # samples per Kokoro control frame (F0 / N grid)
VOCOS_HOP = 256  # samples per Vocos mel frame
