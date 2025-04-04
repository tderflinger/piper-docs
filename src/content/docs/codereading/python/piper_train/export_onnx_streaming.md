---
title: export_onnx_streaming.py
description: export_onnx_streaming.py
---

## Code Explained

The `VitsEncoder` class is a PyTorch module (`nn.Module`) that represents the encoder component of a Variational Inference Text-to-Speech (VITS) model. This encoder is responsible for processing input sequences (e.g., phonemes) and generating intermediate latent representations (`z_p`) that are used in subsequent stages of the model. The class encapsulates the logic for encoding input sequences, handling speaker embeddings, and generating attention-based alignments.

### Constructor (`__init__`)
The constructor initializes the `VitsEncoder` by accepting a generator object (`gen`) as an argument. This generator contains the core components of the VITS model, such as the phoneme encoder (`enc_p`), duration predictor (`dp`), and speaker embedding module (`emb_g`). The `super().__init__()` call ensures that the parent `nn.Module` class is properly initialized, enabling the use of PyTorch's module features like parameter registration and forward propagation.

### Forward Method
The `forward` method defines the computation performed by the encoder during a forward pass. It takes the following inputs:
- `x`: The input sequence (e.g., phoneme IDs) represented as a tensor.
- `x_lengths`: The lengths of the input sequences, used to create masks for variable-length inputs.
- `scales`: A tensor containing three scaling factors: `noise_scale`, `length_scale`, and `noise_scale_w`, which control the noise and duration adjustments in the model.
- `sid` (optional): The speaker ID, used for multi-speaker models to retrieve speaker-specific embeddings.

#### Encoding and Speaker Embeddings
The input sequence `x` is passed through the phoneme encoder (`gen.enc_p`), which outputs:
- `m_p`: The mean of the latent representation.
- `logs_p`: The log standard deviation of the latent representation.
- `x_mask`: A mask indicating valid positions in the input sequence.

If the model supports multiple speakers (`gen.n_speakers > 1`), the speaker ID (`sid`) is used to retrieve a speaker embedding (`g`) via the speaker embedding module (`gen.emb_g`). This embedding is reshaped to match the expected dimensions. For single-speaker models, `g` is set to `None`.

#### Duration Prediction and Alignment
The duration predictor (`gen.dp`) estimates the duration of each phoneme in the input sequence. If stochastic duration prediction (`use_sdp`) is enabled, the predictor operates in reverse mode with added noise (`noise_scale_w`). The predicted durations (`logw`) are exponentiated and scaled by the `length_scale` to compute the adjusted durations (`w`). These durations are then used to calculate the output sequence lengths (`y_lengths`) and masks (`y_mask`).

The attention mask (`attn_mask`) is computed by combining the input and output masks. The `commons.generate_path` function generates an attention alignment path (`attn`) based on the predicted durations, ensuring that the input and output sequences are aligned.

#### Latent Representation
The latent representation (`z_p`) is computed by adding noise to the mean (`m_p`) scaled by the standard deviation (`exp(logs_p)`) and the `noise_scale`. This representation captures the variability in the input sequence and serves as the output of the encoder.

### Outputs
The `forward` method returns three outputs:
1. `z_p`: The latent representation of the input sequence.
2. `y_mask`: The mask for the output sequence, indicating valid positions.
3. `g`: The speaker embedding (if applicable), or `None` for single-speaker models.

### Summary
The `VitsEncoder` class is a critical component of the VITS model, responsible for transforming input sequences into latent representations while handling speaker-specific adjustments and duration-based alignments. Its design leverages PyTorch's modular structure and integrates key functionalities like stochastic duration prediction and attention alignment, making it a versatile and efficient encoder for text-to-speech tasks.

The `VitsDecoder` class is a PyTorch module (`nn.Module`) that represents the decoder component of a Variational Inference Text-to-Speech (VITS) model. This decoder is responsible for transforming latent representations (`z`) generated by the encoder into audio waveforms or spectrograms. It encapsulates the logic for applying flow-based transformations and decoding the processed latent representations into meaningful outputs.

### Constructor (`__init__`)
The constructor initializes the `VitsDecoder` by accepting a generator object (`gen`) as an argument. This generator contains the core components of the VITS model, such as the flow module (`gen.flow`) and the decoder module (`gen.dec`). The `super().__init__()` call ensures that the parent `nn.Module` class is properly initialized, enabling the use of PyTorch's features like parameter registration and forward propagation. The `gen` object is stored as an instance attribute, allowing the decoder to access the necessary submodules during the forward pass.

### Forward Method
The `forward` method defines the computation performed by the decoder during a forward pass. It takes the following inputs:
- `z`: The latent representation of the input sequence, typically produced by the encoder.
- `y_mask`: A mask indicating valid positions in the output sequence, used to handle variable-length inputs.
- `g` (optional): The speaker embedding, used for multi-speaker models to condition the decoding process on speaker-specific characteristics.

#### Flow-Based Transformation
The latent representation `z` is passed through the flow module (`gen.flow`) in reverse mode. Flow-based models are invertible transformations that map between latent and data spaces. In this case, the reverse operation transforms the latent representation back into a space suitable for decoding. The `y_mask` and optional speaker embedding (`g`) are provided as additional inputs to condition the transformation.

#### Decoding
The transformed representation is then passed to the decoder module (`gen.dec`), which generates the final output. The decoder applies the mask (`z * y_mask`) to ensure that only valid positions are processed. If a speaker embedding (`g`) is provided, it is used to condition the decoding process, enabling the generation of speaker-specific outputs in multi-speaker models.

### Outputs
The `forward` method returns the final output of the decoder, which could be an audio waveform, spectrogram, or another representation depending on the model's design. This output is the result of transforming the latent representation into a meaningful format suitable for downstream tasks or playback.

### Summary
The `VitsDecoder` class is a critical component of the VITS model, responsible for converting latent representations into audio outputs. By leveraging flow-based transformations and a flexible decoding process, it supports both single-speaker and multi-speaker scenarios. Its modular design, built on PyTorch's `nn.Module`, ensures that it can be easily integrated into the larger VITS architecture while maintaining clarity and efficiency.

The provided code is a Python script designed to export a trained Variational Inference Text-to-Speech (VITS) model into the ONNX format. ONNX (Open Neural Network Exchange) is a widely used format for deploying machine learning models across different platforms and frameworks. The script includes three main functions: `main`, `export_encoder`, and `export_decoder`, which collectively handle the process of loading a model checkpoint, preparing it for inference, and exporting its encoder and decoder components.

### `main` Function
The `main` function serves as the entry point of the script. It begins by setting a manual seed for PyTorch (`torch.manual_seed(1234)`) to ensure reproducibility of results. It then uses the `argparse` module to define and parse command-line arguments:
- `checkpoint`: Specifies the path to the model checkpoint file (`.ckpt`).
- `output_dir`: Specifies the directory where the exported ONNX files will be saved.
- `--debug`: An optional flag to enable debug-level logging for detailed output.

The logging level is configured based on the `--debug` flag, and the parsed arguments are logged for debugging purposes. The script ensures that the output directory exists by creating it if necessary (`mkdir` with `parents=True` and `exist_ok=True`).

Next, the script loads the VITS model from the specified checkpoint using `VitsModel.load_from_checkpoint`. The generator component of the model (`model_g`) is extracted, and weight normalization is removed from the decoder (`model_g.dec.remove_weight_norm`) to optimize it for inference. The script then calls the `export_encoder` and `export_decoder` functions to export the encoder and decoder components of the model to ONNX format. Informational logs are generated to indicate the progress of the export process.

### `export_encoder` Function
The `export_encoder` function handles the export of the encoder component of the VITS model. It initializes a `VitsEncoder` instance using the generator (`model_g`) and sets it to evaluation mode (`model.eval()`). Dummy input data is created to simulate the input expected by the encoder:
- `sequences`: A tensor of random integers representing phoneme IDs.
- `sequence_lengths`: A tensor indicating the lengths of the input sequences.
- `scales`: A tensor containing scaling factors for noise, duration, and other parameters.
- `sid`: An optional speaker ID tensor, used for multi-speaker models.

The function defines the input and output names for the ONNX export and specifies dynamic axes to handle variable batch sizes and sequence lengths. The encoder is exported to ONNX format using `torch.onnx.export`, and the exported file is saved to the specified output directory. The function returns the output of the encoder for use in the decoder export.

### `export_decoder` Function
The `export_decoder` function exports the decoder component of the VITS model. It initializes a `VitsDecoder` instance using the generator (`model_g`) and sets it to evaluation mode. The function uses the output of the encoder as the input to the decoder. Input names and dynamic axes are defined to handle variable batch sizes and sequence lengths. The decoder is exported to ONNX format using `torch.onnx.export`, and the exported file is saved to the output directory. Informational logs are generated to confirm the successful export.

### Summary
This script automates the process of exporting a trained VITS model into ONNX format, making it suitable for deployment in production environments. The `main` function orchestrates the workflow, while the `export_encoder` and `export_decoder` functions handle the specific tasks of exporting the encoder and decoder components. By leveraging PyTorch's ONNX export capabilities, the script ensures compatibility with a wide range of platforms and frameworks, enabling efficient and flexible deployment of text-to-speech models.

## Source Code

```py
#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .vits import commons
from .vits.lightning import VitsModel

_LOGGER = logging.getLogger("piper_train.export_onnx")
OPSET_VERSION = 15


class VitsEncoder(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, x, x_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]

        gen = self.gen
        x, m_p, logs_p, x_mask = gen.enc_p(x, x_lengths)
        if gen.n_speakers > 1:
            assert sid is not None, "Missing speaker id"
            g = gen.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        if gen.use_sdp:
            logw = gen.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = gen.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(
            commons.sequence_mask(y_lengths, y_lengths.max()), 1
        ).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        return z_p, y_mask, g


class VitsDecoder(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, z, y_mask, g=None):
        z = self.gen.flow(z, y_mask, g=g, reverse=True)
        output = self.gen.dec((z * y_mask), g=g)
        return output


def main() -> None:
    """Main entry point"""
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint (.ckpt)")
    parser.add_argument("output_dir", help="Path to output directory")

    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    args.checkpoint = Path(args.checkpoint)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = VitsModel.load_from_checkpoint(args.checkpoint, dataset=None)
    model_g = model.model_g

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    _LOGGER.info("Exporting encoder...")
    decoder_input = export_encoder(args, model_g)
    _LOGGER.info("Exporting decoder...")
    export_decoder(args, model_g, decoder_input)
    _LOGGER.info("Exported model to  %s", str(args.output_dir))


def export_encoder(args, model_g):
    model = VitsEncoder(model_g)
    model.eval()

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])

    sid: Optional[torch.LongTensor] = None
    if num_speakers > 1:
        sid = torch.LongTensor([0])

    # noise, noise_w, length
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy_input = (sequences, sequence_lengths, scales, sid)

    output_names = [
        "z",
        "y_mask",
    ]
    if model_g.n_speakers > 1:
        output_names.append("g")

    onnx_path = os.fspath(args.output_dir.joinpath("encoder.onnx"))

    # Export
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=onnx_path,
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 2: "time"},
        },
    )
    _LOGGER.info("Exported encoder to %s", onnx_path)

    return model(*dummy_input)


def export_decoder(args, model_g, decoder_input):
    model = VitsDecoder(model_g)
    model.eval()

    input_names = [
        "z",
        "y_mask",
    ]
    if model_g.n_speakers > 1:
        input_names.append("g")

    onnx_path = os.fspath(args.output_dir.joinpath("decoder.onnx"))

    # Export
    torch.onnx.export(
        model=model,
        args=decoder_input,
        f=onnx_path,
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes={
            "z": {0: "batch_size", 2: "time"},
            "y_mask": {0: "batch_size", 2: "time"},
            "output": {0: "batch_size", 1: "time"},
        },
    )

    _LOGGER.info("Exported decoder to %s", onnx_path)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

```