---
title: clean_cached_audio.py
description: clean_cached_audio.py
---

## Code Explained

The provided `main` function is a Python script designed to validate and optionally clean up cached audio or spectrogram files stored in a specified directory. These files are expected to be PyTorch `.pt` files, which typically contain serialized tensors or model data. The script uses multithreading to efficiently process multiple files in parallel, making it suitable for directories with a large number of files.

The function begins by setting up an argument parser using the `argparse` module. It defines three command-line arguments:
1. `--cache-dir`: A required argument specifying the directory containing the `.pt` files to be checked.
2. `--delete`: An optional flag that, when set, deletes files that fail to load.
3. `--debug`: An optional flag that enables debug-level logging for more detailed output.

After parsing the arguments, the script configures the logging system using `logging.basicConfig`. The logging level is set to `DEBUG` if the `--debug` flag is provided; otherwise, it defaults to `INFO`. This allows the script to provide detailed feedback during execution, which is especially useful for debugging.

The `cache_dir` variable is initialized as a `Path` object representing the directory specified by the `--cache-dir` argument. The script also initializes a counter, `num_deleted`, to track the number of files deleted during execution.

The core logic for processing files is encapsulated in the `check_file` function. This function takes a file path (`pt_path`) as input and attempts to load the file using `torch.load`. If the file loads successfully, it is considered valid. If an exception occurs (e.g., due to file corruption or an incompatible format), the script logs an error message. If the `--delete` flag is set, the script deletes the problematic file using the `unlink` method of the `Path` object and increments the `num_deleted` counter.

To process files in parallel, the script uses a `ThreadPoolExecutor`. It iterates over all `.pt` files in the specified directory using the `glob` method and submits each file to the `check_file` function as a separate task. This approach leverages multithreading to improve performance, especially when dealing with a large number of files.

Finally, the script prints the total number of files deleted to the console. This provides a summary of the cleanup operation, allowing the user to verify the results. Overall, the script is a practical tool for maintaining the integrity of cached PyTorch files, with options for debugging and automated cleanup.

## Source Code

```py
#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

import torch

_LOGGER = logging.getLogger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Path to directory with audio/spectrogram files (*.pt)",
    )
    parser.add_argument(
        "--delete", action="store_true", help="Delete files that fail to load"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    cache_dir = Path(args.cache_dir)
    num_deleted = 0

    def check_file(pt_path: Path) -> None:
        nonlocal num_deleted

        try:
            _LOGGER.debug("Checking %s", pt_path)
            torch.load(str(pt_path))
        except Exception:
            _LOGGER.error(pt_path)
            if args.delete:
                pt_path.unlink()
                num_deleted += 1

    with ThreadPoolExecutor() as executor:
        for pt_path in cache_dir.glob("*.pt"):
            executor.submit(check_file, pt_path)

    print("Deleted:", num_deleted, "file(s)")


if __name__ == "__main__":
    main()
```
