---
title: mel_processing.py
description: mel_processing.py
---

## Code Explained

The provided code implements a set of functions for audio signal processing, specifically focusing on spectrogram and mel-spectrogram generation using PyTorch. These functions are essential for tasks like speech synthesis, audio analysis, and machine learning models that operate on audio features. Below is a detailed explanation of the key components:

---

### **Dynamic Range Compression and Decompression**
1. **`dynamic_range_compression_torch`**:
   - This function applies logarithmic compression to an input tensor `x` to reduce its dynamic range. 
   - A small value `clip_val` ensures numerical stability by preventing logarithms of zero or negative values.
   - The compression factor `C` scales the input before applying the logarithm.
   - **Use Case**: This is commonly used to normalize audio magnitudes, making them more suitable for machine learning models.

2. **`dynamic_range_decompression_torch`**:
   - This function reverses the compression by applying the exponential function and dividing by the compression factor `C`.
   - **Use Case**: It restores the original dynamic range of compressed audio magnitudes.

---

### **Spectral Normalization and Denormalization**
1. **`spectral_normalize_torch`**:
   - This function normalizes spectrogram magnitudes by applying dynamic range compression.
   - **Use Case**: It prepares spectrograms for input into models by reducing their dynamic range.

2. **`spectral_de_normalize_torch`**:
   - This function reverses the normalization by applying dynamic range decompression.
   - **Use Case**: It restores the original spectrogram magnitudes after processing.

---

### **Spectrogram Generation**
1. **`spectrogram_torch`**:
   - This function computes the spectrogram of an audio signal `y` using the Short-Time Fourier Transform (STFT).
   - **Key Steps**:
     - A Hann window is applied to the signal to reduce spectral leakage.
     - The signal is padded to ensure proper alignment for the STFT.
     - The STFT is computed, and the magnitude of the complex spectrogram is calculated.
   - **Global Variables**:
     - `hann_window`: A cache for precomputed Hann windows to avoid redundant computations.
   - **Use Case**: Spectrograms are a fundamental representation of audio signals, capturing frequency and time information.

---

### **Mel-Spectrogram Conversion**
1. **`spec_to_mel_torch`**:
   - Converts a spectrogram to a mel-spectrogram using a mel filter bank.
   - **Key Steps**:
     - A mel filter bank is generated using `librosa_mel_fn` and cached in the `mel_basis` dictionary.
     - The spectrogram is multiplied by the mel filter bank to project it onto the mel scale.
     - The resulting mel-spectrogram is normalized using `spectral_normalize_torch`.
   - **Global Variables**:
     - `mel_basis`: A cache for precomputed mel filter banks to avoid redundant computations.
   - **Use Case**: Mel-spectrograms are widely used in speech synthesis and audio processing tasks due to their perceptual relevance.

---

### **Mel-Spectrogram Generation**
1. **`mel_spectrogram_torch`**:
   - Combines the functionality of `spectrogram_torch` and `spec_to_mel_torch` to compute a mel-spectrogram directly from an audio signal.
   - **Key Steps**:
     - Validates the input signal's range to ensure it lies between -1.0 and 1.0.
     - Computes the spectrogram using `spectrogram_torch`.
     - Converts the spectrogram to a mel-spectrogram using `spec_to_mel_torch`.
   - **Global Variables**:
     - `hann_window` and `mel_basis` are used to cache precomputed Hann windows and mel filter banks.
   - **Use Case**: This function is a one-stop solution for generating mel-spectrograms, which are commonly used as input features for deep learning models in tasks like text-to-speech (TTS) and automatic speech recognition (ASR).

---

### **Key Features**
1. **Caching for Efficiency**:
   - Both `hann_window` and `mel_basis` use global dictionaries to cache precomputed values, reducing redundant computations and improving performance.
2. **Numerical Stability**:
   - Functions like `dynamic_range_compression_torch` and `spectrogram_torch` include safeguards (e.g., `clip_val` and small epsilon values) to ensure numerical stability during logarithmic and square root operations.
3. **Flexibility**:
   - The functions support various configurations, such as different FFT sizes, hop sizes, and mel frequency ranges, making them adaptable to different audio processing tasks.

---

### **Applications**
These functions are essential building blocks for audio processing pipelines in machine learning projects. They are particularly useful in:
- **Speech Synthesis**: Generating mel-spectrograms as input to TTS models.
- **Audio Analysis**: Extracting spectrograms and mel-spectrograms for tasks like music analysis and environmental sound classification.
- **Deep Learning**: Preparing audio features for neural networks in tasks like ASR and audio generation.

By combining these functions, developers can efficiently preprocess audio data for a wide range of applications.

## Source Code

```py
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).type_as(y)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[wnsize_dtype_device],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).type_as(spec)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).type_as(y)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).type_as(y)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[wnsize_dtype_device],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
```