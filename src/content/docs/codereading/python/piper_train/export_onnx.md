---
title: export_onnx.py
description: export_onnx.py
---

## Code Explained

The provided Python script is designed to export a trained Variational Inference Text-to-Speech (VITS) model into the ONNX format. ONNX (Open Neural Network Exchange) is a widely used format for deploying machine learning models across different platforms and frameworks. The script uses PyTorch's ONNX export functionality to convert the model into a format suitable for inference in production environments.

### Overview of the `main` Function
The `main` function serves as the entry point of the script. It begins by setting a manual seed for PyTorch (`torch.manual_seed(1234)`) to ensure reproducibility of results. It then uses the `argparse` module to define and parse command-line arguments:
1. `checkpoint`: Specifies the path to the model checkpoint file (`.ckpt`), which contains the trained model's weights and state.
2. `output`: Specifies the path where the exported ONNX model will be saved.
3. `--debug`: An optional flag to enable debug-level logging for detailed output.

The logging level is configured based on the `--debug` flag, and the parsed arguments are logged for debugging purposes. The script ensures that the output directory exists by creating it if necessary (`mkdir` with `parents=True` and `exist_ok=True`).

### Loading the Model
The script loads the VITS model from the specified checkpoint using `VitsModel.load_from_checkpoint`. The generator component of the model (`model_g`) is extracted, which is responsible for generating audio outputs. The number of symbols (`num_symbols`) and speakers (`num_speakers`) supported by the model are retrieved from the generator.

To prepare the model for inference, it is set to evaluation mode using `model_g.eval()`. Additionally, weight normalization is removed from the decoder (`model_g.dec.remove_weight_norm`) to optimize it for inference.

### Custom Forward Method
The script defines a custom `infer_forward` method to replace the generator's default forward method. This method takes input text, text lengths, scaling factors (`scales`), and an optional speaker ID (`sid`). It uses the generator's `infer` method to produce audio outputs, applying the specified noise and length scaling factors. The output audio is reshaped to include a channel dimension (`unsqueeze(1)`) and returned.

The custom `infer_forward` method is assigned to `model_g.forward`, ensuring that the generator uses this method during the ONNX export process.

### Dummy Input for Export
To simulate real input data during the export process, the script creates dummy input tensors:
- `sequences`: A tensor of random integers representing phoneme IDs.
- `sequence_lengths`: A tensor indicating the lengths of the input sequences.
- `scales`: A tensor containing scaling factors for noise, duration, and other parameters.
- `sid`: An optional speaker ID tensor, used for multi-speaker models.

These inputs are packaged into a tuple (`dummy_input`) and passed to the ONNX export function.

### Exporting to ONNX
The script uses `torch.onnx.export` to convert the generator model (`model_g`) into ONNX format. Key parameters for the export include:
- `args=dummy_input`: The dummy input data used to trace the model's computation graph.
- `f=str(args.output)`: The path where the ONNX model will be saved.
- `input_names` and `output_names`: Names for the model's inputs and outputs, which improve readability and usability.
- `dynamic_axes`: Specifies dynamic dimensions (e.g., batch size and sequence length) to make the exported model flexible for varying input sizes.

The exported ONNX model is saved to the specified output path, and a log message confirms the successful export.

### Summary
This script automates the process of converting a trained VITS model into ONNX format, making it suitable for deployment in production environments. By defining a custom forward method and using dummy input data, the script ensures that the exported model is optimized for inference while supporting dynamic input sizes. The use of PyTorch's ONNX export functionality and detailed logging makes the script robust and easy to use in machine learning workflows.

## Source Code

```py
#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger("piper_train.export_onnx")

OPSET_VERSION = 15


def main() -> None:
    """Main entry point"""
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint (.ckpt)")
    parser.add_argument("output", help="Path to output model (.onnx)")

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
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = VitsModel.load_from_checkpoint(args.checkpoint, dataset=None)
    model_g = model.model_g

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

    # Inference only
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    # old_forward = model_g.infer

    def infer_forward(text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0].unsqueeze(1)

        return audio

    model_g.forward = infer_forward

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

    # Export
    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=str(args.output),
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
    )

    _LOGGER.info("Exported model to %s", args.output)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
```
