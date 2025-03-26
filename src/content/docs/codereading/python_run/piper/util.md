---
title: util.py
description: Util
---

## Code Explained

The provided function, `audio_float_to_int16`, is designed to process audio data represented as a NumPy array of floating-point values and convert it into a 16-bit integer format. This is a common step in audio processing pipelines, as many audio file formats (e.g., WAV) store audio samples as 16-bit integers. The function also normalizes the audio data to ensure it fits within the valid range for 16-bit integers, which is from -32768 to 32767.

The function begins by normalizing the audio data. It calculates a scaling factor based on the maximum absolute value of the audio samples (`np.max(np.abs(audio))`) and the `max_wav_value` parameter, which defaults to 32767.0. This scaling ensures that the audio data is adjusted proportionally so that its peak amplitude matches the maximum allowable value for 16-bit integers. A small constant (`0.01`) is used as a lower bound for the maximum value to prevent division by zero or excessively large scaling when the audio data is silent or nearly silent.

Next, the function uses `np.clip` to constrain the normalized audio values within the range of -32767 to 32767. This step ensures that any values exceeding the 16-bit integer range are truncated to the nearest valid value, preventing overflow or distortion when the data is converted to integers.

Finally, the normalized and clipped audio data is cast to the `int16` data type using NumPy's `astype` method. This converts the floating-point values into 16-bit integers, which are suitable for storage or further processing in audio applications. The function then returns the processed audio data as a NumPy array of type `int16`.

In summary, this function is a utility for preparing audio data for storage or playback in formats that require 16-bit integer samples. It ensures that the audio is properly normalized, clipped to the valid range, and converted to the appropriate data type, making it a crucial step in many audio processing workflows.

## Source Code

```py
"""Utilities"""
import numpy as np


def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    """Normalize audio and convert to int16 range"""
    audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm
```