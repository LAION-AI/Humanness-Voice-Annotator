# Attribution and retained licenses

Our original code, documentation and new annotations: **CC BY 4.0**, LAION.

## Included procedural timing planner

`humanness_voice_annotator/_vendor/procedural_prompt_format.py` is an **unchanged** copy of `prompt_format.py` from [LAION-AI/procedural-voice-captions](https://github.com/LAION-AI/procedural-voice-captions), revision `df9514b05f234c692a9e88df7633f5460346d2c9`. Attribution: LAION, 2026. Its Humaneness Medium source, derived captions and normalizer are published under CC BY 4.0 in [LICENSE-HUMANENESS-MEDIUM.md](https://github.com/LAION-AI/procedural-voice-captions/blob/df9514b05f234c692a9e88df7633f5460346d2c9/LICENSE-HUMANENESS-MEDIUM.md).

Our formatting module adapts the planner's *output* to combined round-bracket speech durations and safely quoted free-form MOSS descriptions. It does not modify that included source file. Downloaded procedural files are verified against the source lock and retain their original attribution.

## Models downloaded separately

- [MOSS voice-captioner](https://huggingface.co/laion/moss-voice-captioner) and [MOSS vocal-burst-captioner](https://huggingface.co/laion/moss-vocal-burst-captioner): LAION LoRA weights/documentation/new annotations under CC BY 4.0. Their redistributed [OpenMOSS base](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct) and official source retain **Apache 2.0**. The releases include `BASE_MODEL_LICENSE` and `upstream/LICENSE`.
- [Humaneness Ears](https://huggingface.co/laion/humaneness-ears-base-medium): see the released model card, code and original upstream model credits. The Medium network originates from the [OpenAI Whisper Small](https://huggingface.co/openai/whisper-small) encoder.
- [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3): retains its released license and notices; inspect them before redistribution.
- [German-English Voice Acting Challenges](https://huggingface.co/datasets/laion/german-english-voice-acting-challenges): synthetic audio and new annotations are released under CC BY 4.0; source challenge and voice metadata are retained.

## Software

PyTorch, Transformers, Hugging Face Hub, PEFT, Accelerate, NumPy, SciPy, SoundFile, FFmpeg and their dependencies are installed separately and retain their own licenses. Model and package licenses are not replaced by this repository's CC BY 4.0 notice. No third-party model weights are committed here.
