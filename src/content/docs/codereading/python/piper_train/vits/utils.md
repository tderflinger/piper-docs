---
title: utils.py
description: utils.py
---

## Code Explained

The provided code contains two utility functions: `to_gpu` and `audio_float_to_int16`. These functions are designed to handle tensor operations and audio data normalization, respectively. Here's a detailed explanation:

---

### **1. `to_gpu`**
This function moves a PyTorch tensor to the GPU for accelerated computation.

- **Input**:
  - `x`: A PyTorch tensor that needs to be transferred to the GPU.

- **Behavior**:
  - The `contiguous()` method ensures that the tensor's memory layout is contiguous, which is often required for efficient GPU operations.
  - The `cuda(non_blocking=True)` method transfers the tensor to the GPU. The `non_blocking=True` argument allows the transfer to occur asynchronously if the tensor is pinned in memory, improving performance in certain scenarios.

- **Output**:
  - Returns the tensor in GPU memory, ready for computation.

This function is a common utility in deep learning workflows where tensors need to be processed on the GPU for faster computation.

---

### **2. `audio_float_to_int16`**
This function normalizes audio data and converts it from floating-point format to 16-bit integer format.

- **Inputs**:
  - `audio`: A NumPy array representing the audio signal in floating-point format.
  - `max_wav_value`: A scaling factor, typically set to 32767.0, which corresponds to the maximum value for 16-bit audio.

- **Behavior**:
  - The audio signal is normalized by scaling it to the range defined by `max_wav_value`. The normalization ensures that the audio's amplitude fits within the 16-bit integer range.
  - The `np.clip` function ensures that the values are bounded between `-max_wav_value` and `max_wav_value`, preventing overflow.
  - The `astype("int16")` method converts the normalized audio to 16-bit integer format.

- **Output**:
  - Returns the audio signal as a NumPy array in 16-bit integer format.

This function is particularly useful in audio processing pipelines where audio data needs to be prepared for storage, playback, or further processing in a standard format.

---

### **Use Cases**
- **`to_gpu`**: Used in machine learning models to transfer tensors to the GPU for training or inference.
- **`audio_float_to_int16`**: Used in audio preprocessing pipelines to convert raw audio data into a format compatible with audio playback systems or storage formats like WAV files.

Both functions are lightweight and efficient, making them essential utilities in their respective domains.

## Source Code

```py
import numpy as np
import torch


def to_gpu(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().cuda(non_blocking=True)


def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    """Normalize audio and convert to int16 range"""
    audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm
```