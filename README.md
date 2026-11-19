# AkinVox Faster Kokoro

**Kokoro-82M, 4x faster on GPU and 3.6x faster on CPU. Same 54 voices, same 9 languages.**

I'm building this for [AkinVox](https://akinvox.com), my audiobook site. A single book is hours of audio, so speed matters as much as sound. Faster Kokoro keeps everything in Kokoro that decides what is said and how it is said, and replaces only the last stage, the one that turns it into a waveform, with a smaller and faster one. This is the first release, and I'm still improving it (see [Still improving](#still-improving)).

[Listen and download on Hugging Face](https://huggingface.co/AKinvox/faster-kokoro-v1) · [Code on GitHub](https://github.com/akinvox/faster-kokoro)

## Speed

Text in, 24 kHz audio out, including Kokoro's text processing. 36 English sentences (3 minutes of speech), measured after warm-up.

| Hardware | Mode | Speed | 1 minute of speech takes | Typical sentence |
| --- | --- | ---: | ---: | ---: |
| NVIDIA RTX 3090 Ti | CUDA graphs | 253x real time | 0.24 s | 24 ms |
| NVIDIA RTX 3090 Ti | standard | 95x real time | 0.63 s | 59 ms |
| AMD Ryzen 9 7940HS, 8 threads | CPU | 13.7x real time | 4.4 s | 456 ms |

On the same machines that is 4.1x faster than stock Kokoro on the GPU with CUDA graphs (1.5x without them) and 3.6x faster on the CPU.

## Quality

Measured against stock Kokoro on the same text, voice, timing and pitch.

| Measure | Faster Kokoro | Stock Kokoro |
| --- | ---: | ---: |
| Word error rate, 20 long English passages (8 minutes) | 3.35 % | 3.42 % |
| Voice match to stock, all 54 voices (0 to 1) | 0.958 | 1 |
| Long passages, all 54 voices and 9 languages | no dropouts or drift | |

Word error rate uses Whisper large-v3. Voice match is ECAPA speaker similarity on 20 to 35 second passages: 38 of the 54 voices are at 0.95 or above, and the Hindi voices are lowest (0.90 to 0.93).

## Listen

One sentence in every language, next to stock Kokoro, is on the [Hugging Face page](https://huggingface.co/AKinvox/faster-kokoro-v1).

## Quick start

Python 3.10 or newer, Linux or WSL. CPU works; an NVIDIA GPU is optional. The first run downloads Kokoro-82M, Vocos and the Faster Kokoro weights (about 126 MB); later runs reuse the cache.

```bash
git clone https://github.com/akinvox/faster-kokoro.git
cd faster-kokoro
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -m spacy download en_core_web_sm
```

For an NVIDIA GPU, install the matching [PyTorch CUDA build](https://pytorch.org/get-started/locally/) first.

```python
from faster_kokoro import FasterKokoro

tts = FasterKokoro(cuda_graphs=True)  # leave out cuda_graphs on CPU
tts.save("A new story begins at the little bookshop beside the station.", "story.wav", voice="am_adam")

for chunk in tts.stream(long_text, voice="bf_emma"):
    play(chunk)  # 24 kHz float32, one chunk per sentence-sized piece
```

The language follows the voice name as in Kokoro (`am_adam` American English, `bf_emma` British English, `ef_dora` Spanish, and so on). `speed=` works as in Kokoro.

CUDA graphs are recorded once, the first time the model speaks: about 15 to 30 seconds and about 330 MB of GPU memory. Call `tts.warmup()` to do it at start-up. The output is the same with or without them.

It also works inside Kokoro's own pipeline:

```python
from kokoro import KPipeline
from faster_kokoro import load_model

pipeline = KPipeline(lang_code="a", model=load_model())
```

## How it works

Everything before Kokoro's decoder is stock [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M): text processing, PL-BERT, durations, pitch, energy and the voices. The decoder is replaced by a mel head (31 M parameters), distilled from stock Kokoro's own output, followed by the public [Vocos](https://huggingface.co/charactr/vocos-mel-24khz) vocoder.

## Still improving

I'm still training the next version:

- **Smaller and faster again.** It feeds the Vocos vocoder directly, without the spectrogram in between. The biggest gain should be on CPU.
- **Closer to stock on every voice.** The aim is a voice match of 0.95 or above for all 54 voices, starting with the Hindi voices.

Updates will appear here.

## License

Apache-2.0. Kokoro-82M is Apache-2.0 and Vocos is MIT. See [NOTICE](NOTICE) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
