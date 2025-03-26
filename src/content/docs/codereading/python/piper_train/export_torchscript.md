---
title: export_torchscript.py
description: export_torchscript.py
---

## Code Explained

The provided Python script is designed to export a trained Variational Inference Text-to-Speech (VITS) model into the TorchScript format. TorchScript is a PyTorch feature that allows models to be serialized and optimized for deployment in production environments. This script automates the process of loading a model checkpoint, preparing it for inference, and saving it as a TorchScript file.

### Overview of the `main` Function
The `main` function serves as the entry point of the script. It begins by setting a manual seed for PyTorch (`torch.manual_seed(1234)`) to ensure reproducibility of results. The script uses the `argparse` module to define and parse command-line arguments:
1. `checkpoint`: Specifies the path to the model checkpoint file (`.ckpt`), which contains the trained model's weights and state.
2. `output`: Specifies the path where the exported TorchScript model will be saved.
3. `--debug`: An optional flag to enable debug-level logging for detailed output.

The logging level is configured based on the `--debug` flag, and the parsed arguments are logged for debugging purposes. The script ensures that the output directory exists by creating it if necessary (`mkdir` with `parents=True` and `exist_ok=True`).

### Loading the Model
The script loads the VITS model from the specified checkpoint using `VitsModel.load_from_checkpoint`. The generator component of the model (`model_g`) is extracted, which is responsible for generating audio outputs. The number of symbols (`num_symbols`) supported by the model is retrieved from the generator.

To prepare the model for inference, it is set to evaluation mode using `model_g.eval()`. Additionally, weight normalization is removed from the decoder (`model_g.dec.remove_weight_norm`) to optimize it for inference. The generator's forward method is replaced with its `infer` method (`model_g.forward = model_g.infer`), ensuring that the model uses the appropriate logic for inference.

### Creating Dummy Input
To simulate real input data during the TorchScript tracing process, the script creates dummy input tensors:
- `sequences`: A tensor of random integers representing phoneme IDs, with a fixed length of 50.
- `sequence_lengths`: A tensor indicating the length of the input sequence.
- `sid`: A tensor representing the speaker ID, set to 0 for single-speaker models.
- Scaling factors (`torch.FloatTensor`): Tensors representing noise scale, length scale, and noise width, which control the variability and duration of the generated audio.

These inputs are packaged into a tuple (`dummy_input`) and passed to the TorchScript tracing function.

### Exporting to TorchScript
The script uses `torch.jit.trace` to convert the generator model (`model_g`) into a TorchScript representation. The tracing process records the operations performed by the model when it is executed with the dummy input, creating a static computation graph. The resulting TorchScript model is saved to the specified output path using `torch.jit.save`. A log message confirms the successful export of the model.

### Summary
This script automates the process of converting a trained VITS model into TorchScript format, making it suitable for deployment in production environments. By defining dummy input data and using PyTorch's TorchScript tracing functionality, the script ensures that the exported model is optimized for inference while supporting dynamic input sizes. The use of logging and argument parsing makes the script robust and user-friendly, enabling seamless integration into machine learning workflows.

## Source Code

```py
#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger("piper_train.export_torchscript")


def main():
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

    # Inference only
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    model_g.forward = model_g.infer

    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])

    sid = torch.LongTensor([0])

    dummy_input = (
        sequences,
        sequence_lengths,
        sid,
        torch.FloatTensor([0.667]),
        torch.FloatTensor([1.0]),
        torch.FloatTensor([0.8]),
    )

    jitted_model = torch.jit.trace(model_g, dummy_input)
    torch.jit.save(jitted_model, str(args.output))

    _LOGGER.info("Saved TorchScript model to %s", args.output)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
```
