---
title: __main__.py
description: config.py
---

## Code Explained

The provided Python script is a command-line utility designed for training a text-to-speech (TTS) model, specifically a VITS (Variational Inference Text-to-Speech) model. The `main` function orchestrates the entire process, from parsing command-line arguments to configuring the model and initiating the training process. It uses libraries such as `argparse` for argument parsing, `torch` for deep learning operations, and a custom `Trainer` class for managing the training workflow.

The script begins by setting up logging for debugging purposes and defining command-line arguments. These arguments allow users to specify the dataset directory (`--dataset-dir`), checkpoint saving frequency (`--checkpoint-epochs`), model quality (`--quality`), and whether to resume training from a single-speaker checkpoint (`--resume_from_single_speaker_checkpoint`). Additional arguments are added by the `Trainer` and `VitsModel` classes, which provide flexibility for configuring the training process. The parsed arguments are logged for debugging and converted into a dictionary for later use.

The script ensures reproducibility by setting a random seed for PyTorch and enabling the `cudnn.benchmark` feature for optimized GPU performance. It then reads the configuration and dataset files from the specified dataset directory. The configuration file (`config.json`) contains essential parameters such as the number of phonemes (`num_symbols`), the number of speakers (`num_speakers`), and the audio sample rate. These values are extracted and used to initialize the model.

The `quality` argument determines the model's architecture and hyperparameters. For example, if the quality is set to "x-low," the script configures the model with smaller hidden channels and filter sizes to reduce computational requirements. Conversely, the "high" quality setting uses larger kernel sizes, dilation sizes, and upsampling rates for higher fidelity audio. These configurations are dynamically added to the argument dictionary (`dict_args`) and passed to the `VitsModel` constructor.

If the `--resume_from_single_speaker_checkpoint` argument is provided, the script handles the special case of converting a single-speaker checkpoint into a multi-speaker model. It loads the single-speaker checkpoint and removes incompatible keys related to speaker embeddings. The remaining weights are copied into the multi-speaker model's generator (`model_g`) and discriminator (`model_d`). This process ensures that the model can resume training while adapting to the multi-speaker architecture.

Finally, the script initializes the `Trainer` object and begins the training process by calling `trainer.fit(model)`. If the `--checkpoint-epochs` argument is specified, the trainer is configured to save model checkpoints at the specified interval. Throughout the process, the script logs key events and configurations, providing transparency and aiding in debugging. This design makes the script a robust and flexible tool for training TTS models with varying configurations and datasets.

The `load_state_dict` function is a utility designed to load a saved state dictionary (`saved_state_dict`) into a model's current state dictionary (`state_dict`). This is a common operation in machine learning workflows, particularly when resuming training from a checkpoint or transferring weights between models. The function ensures that the model's parameters are updated with values from the saved state while gracefully handling any mismatches between the two dictionaries.

The function begins by retrieving the current state dictionary of the model using `model.state_dict()`. This dictionary contains the model's parameters (e.g., weights and biases) as key-value pairs, where the keys are parameter names and the values are their corresponding tensors. A new dictionary, `new_state_dict`, is initialized to store the updated parameters.

The function then iterates over each key-value pair in the current state dictionary. For each parameter, it checks whether the key exists in the `saved_state_dict`. If the key is found, the corresponding value from the `saved_state_dict` is used to update `new_state_dict`. This ensures that the model's parameters are restored from the saved state. If the key is not found in the `saved_state_dict`, the function logs a debug message using `_LOGGER.debug`, indicating that the parameter is missing from the checkpoint. In such cases, the function retains the model's initialized value for that parameter, ensuring that the model remains functional even if some parameters are not present in the saved state.

Finally, the function calls `model.load_state_dict(new_state_dict)` to update the model's parameters with the values in `new_state_dict`. This step applies the updated state dictionary to the model, completing the process of loading the saved state. By handling missing parameters gracefully and logging debug messages for transparency, the `load_state_dict` function provides a robust mechanism for managing model state updates in scenarios where the saved state and the model's current state may not fully align.

## Source Code

```py
import argparse
import json
import logging
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--quality",
        default="medium",
        choices=("x-low", "medium", "high"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker and resumes training",
    )
    Trainer.add_argparse_args(parser)
    VitsModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See preprocess.py for format
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])

    trainer = Trainer.from_argparse_args(args)
    if args.checkpoint_epochs is not None:
        trainer.callbacks = [ModelCheckpoint(every_n_epochs=args.checkpoint_epochs)]
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )

    dict_args = vars(args)
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)

    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )

    if args.resume_from_single_speaker_checkpoint:
        assert (
            num_speakers > 1
        ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_from_checkpoint for single-speaker models."

        # Load single-speaker checkpoint
        _LOGGER.debug(
            "Resuming from single-speaker checkpoint: %s",
            args.resume_from_single_speaker_checkpoint,
        )
        model_single = VitsModel.load_from_checkpoint(
            args.resume_from_single_speaker_checkpoint,
            dataset=None,
        )
        g_dict = model_single.model_g.state_dict()
        for key in list(g_dict.keys()):
            # Remove keys that can't be copied over due to missing speaker embedding
            if (
                key.startswith("dec.cond")
                or key.startswith("dp.cond")
                or ("enc.cond_layer" in key)
            ):
                g_dict.pop(key, None)

        # Copy over the multi-speaker model, excluding keys related to the
        # speaker embedding (which is missing from the single-speaker model).
        load_state_dict(model.model_g, g_dict)
        load_state_dict(model.model_d, model_single.model_d.state_dict())
        _LOGGER.info(
            "Successfully converted single-speaker checkpoint to multi-speaker"
        )

    trainer.fit(model)


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
```