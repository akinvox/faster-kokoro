# Third-party notices

The Apache-2.0 license covers AkinVox's original contributions and the released
mel-head weights. It does not relicense third-party dependencies.

| Component | Use | License / source |
|---|---|---|
| Kokoro-82M | Text frontend, predictors and voice packs; original backbone | [Apache-2.0](licenses/KOKORO_LICENSE); [model card](https://huggingface.co/hexgrad/Kokoro-82M) |
| kokoro (Python package) | Inference pipeline | [Apache-2.0](https://github.com/hexgrad/kokoro/blob/main/LICENSE) |
| Misaki | Phonemization | [Apache-2.0](https://github.com/hexgrad/misaki/blob/main/LICENSE) |
| Vocos (`charactr/vocos-mel-24khz`) | Unmodified mel vocoder, code and weights | [MIT, Copyright 2023 Charactr Inc.](licenses/VOCOS_LICENSE); [model card](https://huggingface.co/charactr/vocos-mel-24khz) |
| eSpeak NG / phonemizer | Fallback pronunciation, installed through Misaki dependencies | [eSpeak NG GPL-3.0](https://github.com/espeak-ng/espeak-ng/blob/master/COPYING), [phonemizer GPL-3.0](https://github.com/bootphon/phonemizer/blob/master/LICENSE) |

Vocos was not trained or modified for this release. PyTorch, Transformers,
SoundFile and the other installed packages retain their respective upstream
licenses. Review those terms when redistributing a complete environment or
application.
