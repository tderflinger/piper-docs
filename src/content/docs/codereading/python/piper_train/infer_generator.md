---
title: infer_generator.py
description: infer_generator.py
---

## Code Explained

The provided `main` function is a Python script designed to perform inference using a pre-trained text-to-speech (TTS) generator model. It processes input phoneme sequences, generates corresponding audio waveforms, and writes the results as WAV files to a specified output directory. The script is particularly useful for batch inference tasks, where multiple utterances need to be synthesized from phoneme sequences.

### Argument Parsing and Setup
The script begins by configuring logging with a debug level using `logging.basicConfig(level=logging.DEBUG)`, which ensures detailed logs are available during execution. It then defines an argument parser using `argparse.ArgumentParser` to handle command-line arguments:
1. `--model`: Specifies the path to the pre-trained generator model file (in `.pt` format).
2. `--output-dir`: Specifies the directory where the generated WAV files will be saved.
3. `--sample-rate`: Specifies the audio sample rate for the output WAV files, with a default value of 22,050 Hz.

The parsed arguments are stored in the `args` object. The script ensures that the output directory exists by creating it if necessary using `Path(args.output_dir).mkdir(parents=True, exist_ok=True)`.

### Loading the Model
The pre-trained generator model is loaded using `torch.load(args.model)`. The model is then set to evaluation mode with `model.eval()`, which disables certain behaviors specific to training, such as dropout. This ensures that the model operates deterministically during inference.

### Processing Input and Generating Audio
The script reads input data line-by-line from `sys.stdin`. Each line is expected to be a JSON object containing the following keys:
- `"phoneme_ids"`: A list of phoneme IDs representing the input sequence.
- `"speaker_id"` (optional): An ID specifying the speaker for multi-speaker models.

For each input line:
1. The phoneme IDs are converted into a PyTorch tensor (`torch.LongTensor`) and reshaped to include a batch dimension using `.unsqueeze(0)`.
2. The length of the phoneme sequence is calculated and stored in another tensor (`text_lengths`).
3. If a speaker ID is provided, it is converted into a tensor (`sid`); otherwise, `sid` is set to `None`.

The script then performs inference by passing the input tensors (`text`, `text_lengths`, and `sid`) to the model. The model generates an audio waveform, which is detached from the computation graph and converted to a NumPy array using `.detach().numpy()`.

### Post-Processing and Saving Audio
The generated audio waveform is normalized and converted to 16-bit integer format using the `audio_float_to_int16` function. The script calculates the duration of the audio (`audio_duration_sec`) and the time taken for inference (`infer_sec`). It then computes the real-time factor (RTF), which is the ratio of inference time to audio duration. This metric is logged for each utterance to provide insights into the model's performance.

Finally, the audio is saved as a WAV file using the `write_wav` function. The output file is named using the utterance index (`utt_id`) and saved in the specified output directory.

### Summary
This script is a practical tool for performing inference with a TTS generator model. It supports batch processing of input phoneme sequences, generates high-quality audio waveforms, and provides detailed logging for performance monitoring. By leveraging PyTorch and efficient file handling, the script ensures scalability and ease of use in real-world TTS applications.

## Source Code

```py
#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

from .vits.utils import audio_float_to_int16
from .vits.wavfile import write as write_wav

_LOGGER = logging.getLogger("piper_train.infer_generator")


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(prog="piper_train.infer_generator")
    parser.add_argument("--model", required=True, help="Path to generator (.pt)")
    parser.add_argument("--output-dir", required=True, help="Path to write WAV files")
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = torch.load(args.model)

    # Inference only
    model.eval()

    for i, line in enumerate(sys.stdin):
        line = line.strip()
        if not line:
            continue

        utt = json.loads(line)
        utt_id = str(i)
        phoneme_ids = utt["phoneme_ids"]
        speaker_id = utt.get("speaker_id")

        text = torch.LongTensor(phoneme_ids).unsqueeze(0)
        text_lengths = torch.LongTensor([len(phoneme_ids)])
        sid = torch.LongTensor([speaker_id]) if speaker_id is not None else None

        start_time = time.perf_counter()
        audio = (
            model(
                text,
                text_lengths,
                sid,
                # torch.FloatTensor([0.667]),
                # torch.FloatTensor([1.0]),
                # torch.FloatTensor([0.8]),
            )[0]
            .detach()
            .numpy()
        )
        audio = audio_float_to_int16(audio)
        end_time = time.perf_counter()

        audio_duration_sec = audio.shape[-1] / args.sample_rate
        infer_sec = end_time - start_time
        real_time_factor = (
            infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
        )

        _LOGGER.debug(
            "Real-time factor for %s: %0.2f (infer=%0.2f sec, audio=%0.2f sec)",
            i + 1,
            real_time_factor,
            infer_sec,
            audio_duration_sec,
        )

        output_path = args.output_dir / f"{utt_id}.wav"
        write_wav(str(output_path), args.sample_rate, audio)


if __name__ == "__main__":
    main()
```