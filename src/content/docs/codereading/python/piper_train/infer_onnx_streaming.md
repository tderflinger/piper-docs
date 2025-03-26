---
title: infer_onnx_streaming.py
description: infer_onnx_streaming.py
---

## Code Explained

The `SpeechStreamer` class is designed to enable real-time streaming of synthesized speech using ONNX models for both the encoder and decoder components of a text-to-speech (TTS) system. It provides functionality for efficient inference, chunk-based processing, and streaming of audio data, making it suitable for low-latency applications such as live speech synthesis.

### Class Overview
The class takes the following arguments during initialization:
- `encoder_path`: Path to the ONNX model for the encoder.
- `decoder_path`: Path to the ONNX model for the decoder.
- `sample_rate`: The output audio sample rate.
- `chunk_size`: The number of mel frames to decode in each step (default is 45).
- `chunk_padding`: The number of mel frames to pad at the start and end of each chunk to reduce decoding artifacts (default is 10).

The class initializes ONNX runtime sessions for the encoder and decoder models and stores the provided parameters for use during inference.

---

### `encoder_infer` Method
This method performs inference using the encoder model. It:
1. Measures the start time using `time.perf_counter`.
2. Runs the encoder model with the provided input using `self.encoder.run`.
3. Calculates the inference time and logs it in milliseconds.
4. Computes the real-time factor (RTF), which is the ratio of inference time to the duration of the generated audio, and logs it.
5. Returns the encoder's output.

---

### `decoder_infer` Method
This method performs inference using the decoder model. It:
1. Prepares the input dictionary for the decoder, including latent variables (`z`), masks (`y_mask`), and optionally speaker embeddings (`g`).
2. Measures the start time and runs the decoder model using `self.decoder.run`.
3. Logs the inference time and calculates the RTF for the decoder.
4. Returns the generated audio waveform.

---

### `chunk` Method
The `chunk` method splits the encoder's output into smaller chunks for streaming. It:
1. Extracts latent variables (`z`) and masks (`y_mask`) from the encoder's output.
2. Checks if the number of frames is too small for streaming; if so, it processes the entire output in one step.
3. Splits the latent variables and masks into chunks based on the `chunk_size` and adds padding at the start and end of each chunk to reduce artifacts.
4. Iterates through the chunks, performs inference using the `decoder_infer` method, and yields the processed audio for each chunk.

---

### `stream` Method
The `stream` method orchestrates the real-time synthesis process. It:
1. Starts a timer to measure latency.
2. Runs the encoder inference using `encoder_infer`.
3. Processes the encoder's output in chunks using the `chunk` method.
4. Converts the generated audio to 16-bit integer format using `audio_float_to_int16` and yields it as byte data.
5. Logs the latency for the first chunk and indicates when synthesis is complete.

---

### Key Features
1. **Real-Time Processing**: The class is optimized for low-latency applications, with detailed logging of inference times and real-time factors.
2. **Chunk-Based Streaming**: By splitting the encoder's output into smaller chunks, the class supports efficient streaming of audio data while minimizing artifacts.
3. **ONNX Runtime Integration**: The use of ONNX models for both the encoder and decoder ensures compatibility with a wide range of platforms and hardware accelerators.
4. **Padding for Artifact Reduction**: The `chunk` method adds padding to reduce decoding artifacts at chunk boundaries, improving audio quality.

---

### Use Case
The `SpeechStreamer` class is ideal for applications requiring real-time speech synthesis, such as virtual assistants, live captioning, or interactive voice systems. Its modular design and efficient processing make it a robust solution for streaming TTS systems.

The provided code is a Python script designed for real-time speech synthesis using ONNX models for both the encoder and decoder components of a text-to-speech (TTS) system. It includes functionality for streaming audio synthesis, denoising, and spectral transformations such as Short-Time Fourier Transform (STFT) and its inverse (ISTFT). Below is an explanation of the key components:

---

### **`main` Function**
The `main` function serves as the entry point for the script. It:
1. **Configures Logging**: Sets the logging level to `DEBUG` for detailed output during execution.
2. **Parses Command-Line Arguments**: Defines arguments for specifying paths to the encoder and decoder ONNX models, sample rate, noise scaling factors, chunk size, and padding. These parameters control the behavior of the speech synthesis pipeline.
3. **Initializes the SpeechStreamer**: Creates an instance of the `SpeechStreamer` class, which handles real-time streaming of synthesized speech. The streamer is initialized with the provided encoder and decoder paths, sample rate, chunk size, and padding.
4. **Processes Input**: Reads input JSON lines from `sys.stdin`, where each line contains phoneme IDs and optionally a speaker ID. These inputs are converted into NumPy arrays for inference.
5. **Streams Audio**: Calls the `stream` method of the `SpeechStreamer` to generate audio chunks in real time. The synthesized audio chunks are written to `sys.stdout.buffer` for immediate playback or further processing.

---

### **`denoise` Function**
The `denoise` function reduces noise in the synthesized audio using a bias spectrogram and a denoiser strength parameter. It:
1. **Transforms Audio**: Converts the input audio into its magnitude and phase components using the `transform` function.
2. **Repeats Bias Spectrogram**: Matches the length of the bias spectrogram to the audio spectrogram by repeating it along the time axis.
3. **Applies Denoising**: Subtracts the scaled bias spectrogram from the audio spectrogram and clips negative values to zero.
4. **Reconstructs Audio**: Converts the denoised spectrogram back into the time domain using the `inverse` function.

---

### **`stft` and `istft` Functions**
These functions perform spectral transformations:
1. **`stft`**: Computes the Short-Time Fourier Transform (STFT) of a time-domain signal. It:
   - Applies a Hanning window to each frame of the signal.
   - Computes the FFT for each frame.
   - Returns a 2D array where rows represent time slices and columns represent frequency bins.
2. **`istft`**: Inverts an STFT back into a time-domain signal. It:
   - Applies the inverse FFT to each time slice.
   - Overlaps and adds the reconstructed frames to produce the final signal.

---

### **`inverse` Function**
The `inverse` function reconstructs a time-domain signal from its magnitude and phase components. It:
1. **Recombines Magnitude and Phase**: Converts the polar representation (magnitude and phase) back into complex numbers.
2. **Performs ISTFT**: Calls the `istft` function to reconstruct the time-domain signal for each batch of spectrograms.
3. **Returns the Reconstructed Signal**: Concatenates the reconstructed signals across batches.

---

### **`transform` Function**
The `transform` function computes the magnitude and phase of a time-domain signal. It:
1. **Performs STFT**: Calls the `stft` function to compute the spectrogram for each input signal.
2. **Extracts Real and Imaginary Parts**: Separates the real and imaginary components of the spectrogram.
3. **Computes Magnitude and Phase**: Calculates the magnitude and phase from the real and imaginary parts.
4. **Returns Magnitude and Phase**: These components are used for further processing, such as denoising or reconstruction.

---

### **Real-Time Streaming with `SpeechStreamer`**
The `SpeechStreamer` class is central to the script's functionality. It:
1. **Handles ONNX Models**: Loads the encoder and decoder ONNX models for inference.
2. **Performs Inference**: Processes input phoneme sequences using the encoder and generates audio using the decoder.
3. **Streams Audio**: Splits the encoder's output into chunks, applies padding to reduce artifacts, and streams the synthesized audio in real time.

The `stream` method in `SpeechStreamer` is particularly important for low-latency applications. It:
- Starts a timer to measure latency.
- Calls the encoder and decoder in sequence to generate audio chunks.
- Converts the audio to 16-bit integer format and yields it as byte data.

---

### **Use Case**
This script is ideal for real-time TTS applications, such as virtual assistants, live captioning, or interactive voice systems. Its modular design, efficient streaming, and support for denoising make it a robust solution for deploying ONNX-based TTS models in production environments.

## Source Code

```py
#!/usr/bin/env python3

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime

from .vits.utils import audio_float_to_int16

_LOGGER = logging.getLogger("piper_train.infer_onnx")


class SpeechStreamer:
    """
    Stream speech in real time.

    Args:
        encoder_path: path to encoder ONNX model
        decoder_path: path to decoder ONNX model
        sample_rate: output sample rate
        chunk_size: number of mel frames to decode in each steps (time in secs = chunk_size * 256)
        chunk_padding: number of mel frames to be concatinated to the start and end of the current chunk to reduce decoding artifacts
    """

    def __init__(
        self,
        encoder_path,
        decoder_path,
        sample_rate,
        chunk_size=45,
        chunk_padding=10,
    ):
        sess_options = onnxruntime.SessionOptions()
        _LOGGER.debug("Loading encoder model from %s", encoder_path)
        self.encoder = onnxruntime.InferenceSession(
            encoder_path, sess_options=sess_options
        )
        _LOGGER.debug("Loading decoder model from %s", decoder_path)
        self.decoder = onnxruntime.InferenceSession(
            decoder_path, sess_options=sess_options
        )

        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.chunk_padding = chunk_padding

    def encoder_infer(self, enc_input):
        ENC_START = time.perf_counter()
        enc_output = self.encoder.run(None, enc_input)
        ENC_INFER = time.perf_counter() - ENC_START
        _LOGGER.debug(f"Encoder inference {round(ENC_INFER * 1000)}")
        wav_length = enc_output[0].shape[2] * 256
        enc_rtf = round(ENC_INFER / (wav_length / self.sample_rate), 2)
        _LOGGER.debug(f"Encoder RTF {enc_rtf}")
        return enc_output

    def decoder_infer(self, z, y_mask, g=None):
        dec_input = {"z": z, "y_mask": y_mask}
        if g:
            dec_input["g"] = g
        DEC_START = time.perf_counter()
        audio = self.decoder.run(None, dec_input)[0].squeeze()
        DEC_INFER = time.perf_counter() - DEC_START
        _LOGGER.debug(f"Decoder inference {round(DEC_INFER * 1000)}")
        dec_rtf = round(DEC_INFER / (len(audio) / self.sample_rate), 2)
        _LOGGER.debug(f"Decoder RTF {dec_rtf}")
        return audio

    def chunk(self, enc_output):
        z, y_mask, *dec_args = enc_output
        n_frames = z.shape[2]
        if n_frames <= (self.chunk_size + (2 * self.chunk_padding)):
            # Too short to stream
            return self.decoder_infer(z, y_mask, *dec_args)
        split_at = [
            i * self.chunk_size for i in range(1, math.ceil(n_frames / self.chunk_size))
        ]
        chunks = list(
            zip(
                np.split(z, split_at, axis=2),
                np.split(y_mask, split_at, axis=2),
            )
        )
        wav_start_pad = wav_end_pad = None
        for idx, (z_chunk, y_mask_chunk) in enumerate(chunks):
            if idx > 0:
                prev_z, prev_y_mask = chunks[idx - 1]
                start_zpad = prev_z[:, :, -self.chunk_padding :]
                start_ypad = prev_y_mask[:, :, -self.chunk_padding :]
                z_chunk = np.concatenate([start_zpad, z_chunk], axis=2)
                y_mask_chunk = np.concatenate([start_ypad, y_mask_chunk], axis=2)
                wav_start_pad = start_zpad.shape[2] * 256
            if (idx + 1) < len(chunks):
                next_z, next_y_mask = chunks[idx + 1]
                end_zpad = next_z[:, :, : self.chunk_padding]
                end_ypad = next_y_mask[:, :, : self.chunk_padding]
                z_chunk = np.concatenate([z_chunk, end_zpad], axis=2)
                y_mask_chunk = np.concatenate([y_mask_chunk, end_ypad], axis=2)
                wav_end_pad = end_zpad.shape[2] * 256
            audio = self.decoder_infer(z_chunk, y_mask_chunk, *dec_args)
            yield audio[wav_start_pad:-wav_end_pad]

    def stream(self, encoder_input):
        start_time = time.perf_counter()
        has_shown_latency = False
        _LOGGER.debug("Starting synthesis")
        enc_output = self.encoder_infer(encoder_input)
        for wav in self.chunk(enc_output):
            if len(wav) == 0:
                continue
            if not has_shown_latency:
                LATENCY = round((time.perf_counter() - start_time) * 1000)
                _LOGGER.debug(f"Latency {LATENCY}")
                has_shown_latency = True
            audio = audio_float_to_int16(wav)
            yield audio.tobytes()
        _LOGGER.debug("Synthesis done!")


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(prog="piper_train.infer_onnx_streaming")
    parser.add_argument(
        "--encoder", required=True, help="Path to encoder model (.onnx)"
    )
    parser.add_argument(
        "--decoder", required=True, help="Path to decoder  model (.onnx)"
    )
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=45,
        help="Number of mel frames to decode at each step"
    )
    parser.add_argument(
        "--chunk-padding",
        type=int,
        default=5,
        help="Number of mel frames to add to the start and end of the current chunk to reduce decoding artifacts"
    )

    args = parser.parse_args()

    streamer = SpeechStreamer(
        encoder_path=os.fspath(args.encoder),
        decoder_path=os.fspath(args.decoder),
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        chunk_padding=args.chunk_padding,
    )

    output_buffer = sys.stdout.buffer

    for i, line in enumerate(sys.stdin):
        line = line.strip()
        if not line:
            continue

        utt = json.loads(line)
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

        stream = streamer.stream(
            {
                "input": text,
                "input_lengths": text_lengths,
                "scales": scales,
                "sid": sid,
            }
        )
        for wav_chunk in stream:
            output_buffer.write(wav_chunk)
            output_buffer.flush()


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