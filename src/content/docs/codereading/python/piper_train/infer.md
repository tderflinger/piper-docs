---
title: infer.py
description: infer.py
---

## Code Explained

The provided `main` function is a Python script designed to perform inference using a pre-trained Variational Inference Text-to-Speech (VITS) model. It processes phoneme sequences, generates corresponding audio waveforms, and saves the results as WAV files. Below is a detailed explanation of its functionality:

---

### **Argument Parsing and Setup**
The script begins by configuring logging with a debug level using `logging.basicConfig(level=logging.DEBUG)`, ensuring detailed logs are available during execution. It then defines an argument parser using `argparse.ArgumentParser` to handle command-line arguments:
1. `--checkpoint`: Specifies the path to the model checkpoint file (`.ckpt`) containing the pre-trained VITS model.
2. `--output-dir`: Specifies the directory where the generated WAV files will be saved.
3. `--sample-rate`: Specifies the audio sample rate for the output WAV files, with a default value of 22,050 Hz.
4. `--noise-scale`, `--length-scale`, and `--noise-w`: Control the variability, duration, and noise characteristics of the generated audio.

The parsed arguments are stored in the `args` object. The script ensures that the output directory exists by creating it if necessary using `Path(args.output_dir).mkdir(parents=True, exist_ok=True)`.

---

### **Loading the VITS Model**
The pre-trained VITS model is loaded using `VitsModel.load_from_checkpoint(args.checkpoint, dataset=None)`. This method restores the model's weights and state from the specified checkpoint file. The model is then set to evaluation mode with `model.eval()`, which disables certain behaviors specific to training, such as dropout, ensuring deterministic inference.

Additionally, the script removes weight normalization from the decoder component of the model (`model.model_g.dec.remove_weight_norm()`) using `torch.no_grad()`. This step optimizes the model for inference by simplifying the computation graph.

---

### **Processing Input and Generating Audio**
The script reads input data line-by-line from `sys.stdin`. Each line is expected to be a JSON object containing:
- `"phoneme_ids"`: A list of phoneme IDs representing the input sequence.
- `"speaker_id"` (optional): An ID specifying the speaker for multi-speaker models.

For each input line:
1. The phoneme IDs are converted into a PyTorch tensor (`torch.LongTensor`) and reshaped to include a batch dimension using `.unsqueeze(0)`.
2. The length of the phoneme sequence is calculated and stored in another tensor (`text_lengths`).
3. If a speaker ID is provided, it is converted into a tensor (`sid`); otherwise, `sid` is set to `None`.

The script then performs inference by passing the input tensors (`text`, `text_lengths`, `scales`, and `sid`) to the model. The model generates an audio waveform, which is detached from the computation graph and converted to a NumPy array using `.detach().numpy()`.

---

### **Post-Processing and Saving Audio**
The generated audio waveform is normalized and converted to 16-bit integer format using the `audio_float_to_int16` function. This ensures compatibility with standard WAV file formats. The script calculates the duration of the audio (`audio_duration_sec`) and the time taken for inference (`infer_sec`). It then computes the real-time factor (RTF), which is the ratio of inference time to audio duration. This metric is logged for each utterance to provide insights into the model's performance.

Finally, the audio is saved as a WAV file using the `write_wav` function. The output file is named using the utterance index (`utt_id`) and saved in the specified output directory.

---

### **Key Features**
1. **VITS Model Integration**: The script leverages a pre-trained VITS model for high-quality text-to-speech synthesis.
2. **Real-Time Performance Metrics**: Logs the real-time factor (RTF) for each utterance, providing insights into the model's efficiency.
3. **Batch Processing**: Supports batch inference by reading multiple input lines from `sys.stdin`.
4. **Scalability**: Handles multi-speaker models by accepting optional speaker IDs as input.

---

### **Use Case**
This script is ideal for batch inference tasks in TTS systems. It processes phoneme sequences, generates high-quality audio waveforms, and provides detailed performance metrics. Its modular design and support for VITS models make it a robust solution for deploying TTS models in production environments.

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

from .vits.lightning import VitsModel
from .vits.utils import audio_float_to_int16
from .vits.wavfile import write as write_wav

_LOGGER = logging.getLogger("piper_train.infer")


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(prog="piper_train.infer")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument("--output-dir", required=True, help="Path to write WAV files")
    parser.add_argument("--sample-rate", type=int, default=22050)
    #
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-w", type=float, default=0.8)
    #
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = VitsModel.load_from_checkpoint(args.checkpoint, dataset=None)

    # Inference only
    model.eval()

    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()

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
        scales = [args.noise_scale, args.length_scale, args.noise_w]
        sid = torch.LongTensor([speaker_id]) if speaker_id is not None else None

        start_time = time.perf_counter()
        audio = model(text, text_lengths, scales, sid=sid).detach().numpy()
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