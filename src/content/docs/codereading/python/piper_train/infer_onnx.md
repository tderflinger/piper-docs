---
title: infer_onnx.py
description: infer_onnx.py
---

## Code Explained

The provided Python script is designed for performing inference using an ONNX-based text-to-speech (TTS) model. It processes phoneme sequences, generates corresponding audio waveforms, and writes the results as WAV files. Additionally, it includes utility functions for spectral transformations, denoising, and audio reconstruction. Below is an explanation of the key components:

---

### **`main` Function**
The `main` function serves as the entry point for the script. It:
1. **Configures Logging**: Sets the logging level to `DEBUG` for detailed output during execution.
2. **Parses Command-Line Arguments**: Defines arguments for specifying the ONNX model path, output directory, sample rate, and scaling factors for noise and length. These parameters control the behavior of the inference pipeline.
3. **Initializes the ONNX Model**: Loads the ONNX model using `onnxruntime.InferenceSession` and logs the successful loading of the model.
4. **Processes Input**: Reads input JSON lines from `sys.stdin`, where each line contains phoneme IDs and optionally a speaker ID. These inputs are converted into NumPy arrays for inference.
5. **Performs Inference**: Calls the ONNX model with the input tensors (phoneme IDs, text lengths, scaling factors, and speaker ID). The model generates audio waveforms, which are converted to 16-bit integer format for saving as WAV files.
6. **Logs Performance Metrics**: Measures the inference time and calculates the real-time factor (RTF), which is the ratio of inference time to audio duration. This metric is logged for each utterance.
7. **Writes Output**: Saves the generated audio as WAV files in the specified output directory.

---

### **`denoise` Function**
The `denoise` function reduces noise in the synthesized audio using a bias spectrogram and a denoiser strength parameter. It:
1. **Transforms Audio**: Converts the input audio into its magnitude and phase components using the `transform` function.
2. **Repeats Bias Spectrogram**: Matches the length of the bias spectrogram to the audio spectrogram by repeating it along the time axis.
3. **Applies Denoising**: Subtracts the scaled bias spectrogram from the audio spectrogram and clips negative values to zero.
4. **Reconstructs Audio**: Converts the denoised spectrogram back into the time domain using the `inverse` function.

---

### **Spectral Transformation Functions**
1. **`stft`**: Computes the Short-Time Fourier Transform (STFT) of a time-domain signal. It:
   - Applies a Hanning window to each frame of the signal.
   - Computes the FFT for each frame.
   - Returns a 2D array where rows represent time slices and columns represent frequency bins.

2. **`istft`**: Inverts an STFT back into a time-domain signal. It:
   - Applies the inverse FFT to each time slice.
   - Overlaps and adds the reconstructed frames to produce the final signal.

3. **`inverse`**: Reconstructs a time-domain signal from its magnitude and phase components. It:
   - Converts the polar representation (magnitude and phase) back into complex numbers.
   - Calls the `istft` function to reconstruct the time-domain signal for each batch of spectrograms.

4. **`transform`**: Computes the magnitude and phase of a time-domain signal. It:
   - Calls the `stft` function to compute the spectrogram for each input signal.
   - Separates the real and imaginary components of the spectrogram.
   - Calculates the magnitude and phase from the real and imaginary parts.

---

### **Key Features**
1. **ONNX Runtime Integration**: The script uses ONNX Runtime for efficient inference, making it compatible with a wide range of platforms and hardware accelerators.
2. **Real-Time Performance Metrics**: Logs the real-time factor (RTF) for each utterance, providing insights into the model's performance.
3. **Denoising Support**: Includes functionality for reducing noise in the synthesized audio, improving output quality.
4. **Spectral Transformations**: Provides utility functions for STFT, ISTFT, and audio reconstruction, which are essential for audio processing tasks.

---

### **Use Case**
This script is ideal for batch inference tasks in TTS systems. It processes phoneme sequences, generates high-quality audio waveforms, and provides detailed performance metrics. Its modular design and support for ONNX models make it a robust solution for deploying TTS models in production environments.

## Source Code

```py
#!/usr/bin/env python3
import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime

from .vits.utils import audio_float_to_int16
from .vits.wavfile import write as write_wav

_LOGGER = logging.getLogger("piper_train.infer_onnx")


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(prog="piper_train.infer_onnx")
    parser.add_argument("--model", required=True, help="Path to model (.onnx)")
    parser.add_argument("--output-dir", required=True, help="Path to write WAV files")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sess_options = onnxruntime.SessionOptions()
    _LOGGER.debug("Loading model from %s", args.model)
    model = onnxruntime.InferenceSession(str(args.model), sess_options=sess_options)
    _LOGGER.info("Loaded model from %s", args.model)

    # text_empty = np.zeros((1, 300), dtype=np.int64)
    # text_lengths_empty = np.array([text_empty.shape[1]], dtype=np.int64)
    # scales = np.array(
    #     [args.noise_scale, args.length_scale, args.noise_scale_w],
    #     dtype=np.float32,
    # )
    # bias_audio = model.run(
    #     None,
    #     {"input": text_empty, "input_lengths": text_lengths_empty, "scales": scales},
    # )[0].squeeze((0, 1))
    # bias_spec, _ = transform(bias_audio)

    for i, line in enumerate(sys.stdin):
        line = line.strip()
        if not line:
            continue

        utt = json.loads(line)
        # utt_id = utt["id"]
        utt_id = str(i)
        phoneme_ids = utt["phoneme_ids"]
        speaker_id = utt.get("speaker_id")

        text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        text_lengths = np.array([text.shape[1]], dtype=np.int64)
        scales = np.array(
            [args.noise_scale, args.length_scale, args.noise_scale_w],
            dtype=np.float32,
        )
        sid = None

        if speaker_id is not None:
            sid = np.array([speaker_id], dtype=np.int64)

        start_time = time.perf_counter()
        audio = model.run(
            None,
            {
                "input": text,
                "input_lengths": text_lengths,
                "scales": scales,
                "sid": sid,
            },
        )[0].squeeze((0, 1))
        # audio = denoise(audio, bias_spec, 10)
        audio = audio_float_to_int16(audio.squeeze())
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


def denoise(
    audio: np.ndarray, bias_spec: np.ndarray, denoiser_strength: float
) -> np.ndarray:
    audio_spec, audio_angles = transform(audio)

    a = bias_spec.shape[-1]
    b = audio_spec.shape[-1]
    repeats = max(1, math.ceil(b / a))
    bias_spec_repeat = np.repeat(bias_spec, repeats, axis=-1)[..., :b]

    audio_spec_denoised = audio_spec - (bias_spec_repeat * denoiser_strength)
    audio_spec_denoised = np.clip(audio_spec_denoised, a_min=0.0, a_max=None)
    audio_denoised = inverse(audio_spec_denoised, audio_angles)

    return audio_denoised


def stft(x, fft_size, hopsamp):
    """Compute and return the STFT of the supplied time domain signal x.
    Args:
        x (1-dim Numpy array): A time domain signal.
        fft_size (int): FFT size. Should be a power of 2, otherwise DFT will be used.
        hopsamp (int):
    Returns:
        The STFT. The rows are the time slices and columns are the frequency bins.
    """
    window = np.hanning(fft_size)
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    return np.array(
        [
            np.fft.rfft(window * x[i : i + fft_size])
            for i in range(0, len(x) - fft_size, hopsamp)
        ]
    )


def istft(X, fft_size, hopsamp):
    """Invert a STFT into a time domain signal.
    Args:
        X (2-dim Numpy array): Input spectrogram. The rows are the time slices and columns are the frequency bins.
        fft_size (int):
        hopsamp (int): The hop size, in samples.
    Returns:
        The inverse STFT.
    """
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    window = np.hanning(fft_size)
    time_slices = X.shape[0]
    len_samples = int(time_slices * hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n, i in enumerate(range(0, len(x) - fft_size, hopsamp)):
        x[i : i + fft_size] += window * np.real(np.fft.irfft(X[n]))
    return x


def inverse(magnitude, phase):
    recombine_magnitude_phase = np.concatenate(
        [magnitude * np.cos(phase), magnitude * np.sin(phase)], axis=1
    )

    x_org = recombine_magnitude_phase
    n_b, n_f, n_t = x_org.shape  # pylint: disable=unpacking-non-sequence
    x = np.empty([n_b, n_f // 2, n_t], dtype=np.complex64)
    x.real = x_org[:, : n_f // 2]
    x.imag = x_org[:, n_f // 2 :]
    inverse_transform = []
    for y in x:
        y_ = istft(y.T, fft_size=1024, hopsamp=256)
        inverse_transform.append(y_[None, :])

    inverse_transform = np.concatenate(inverse_transform, 0)

    return inverse_transform


def transform(input_data):
    x = input_data
    real_part = []
    imag_part = []
    for y in x:
        y_ = stft(y, fft_size=1024, hopsamp=256).T
        real_part.append(y_.real[None, :, :])  # pylint: disable=unsubscriptable-object
        imag_part.append(y_.imag[None, :, :])  # pylint: disable=unsubscriptable-object
    real_part = np.concatenate(real_part, 0)
    imag_part = np.concatenate(imag_part, 0)

    magnitude = np.sqrt(real_part**2 + imag_part**2)
    phase = np.arctan2(imag_part.data, real_part.data)

    return magnitude, phase


if __name__ == "__main__":
    main()
```