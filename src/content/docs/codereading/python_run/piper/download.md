---
title: download.py
description: download.py
---

## Code Explained

The `get_voices` function is a utility designed to load a JSON file containing metadata about available voices for a text-to-speech (TTS) system. This metadata is either downloaded from a remote server or retrieved from an embedded file within the project. The function ensures that the most up-to-date voice information is used when requested, while also providing a fallback to a local embedded file if the downloaded file is unavailable.

The function accepts two parameters: `download_dir` and `update_voices`. The `download_dir` parameter specifies the directory where the downloaded `voices.json` file should be stored. It can be provided as a string or a `Path` object. The `update_voices` parameter is a boolean flag that determines whether the function should attempt to download the latest version of the `voices.json` file from a remote server.

If `update_voices` is set to `True`, the function constructs the URL for the `voices.json` file using a predefined format (`URL_FORMAT`). It then downloads the file from the remote server using `urlopen` and saves it to the specified `download_dir` using `shutil.copyfileobj`. This ensures that the local copy of `voices.json` is updated with the latest metadata.

After handling the download (if applicable), the function determines which `voices.json` file to load. It first checks if the downloaded file exists in the `download_dir`. If it does, the function uses this file. Otherwise, it falls back to an embedded `voices.json` file located in the `_DIR` directory, which is part of the project. This fallback mechanism ensures that the function can still operate even if the download fails or is not requested.

Finally, the function opens the selected `voices.json` file in read mode with UTF-8 encoding and loads its contents into a Python dictionary using the `json.load` function. This dictionary, which contains metadata about the available voices, is returned to the caller. Throughout the process, the function uses logging to provide debug information, such as the URLs being accessed and the file paths being loaded. This makes it easier to trace the function's behavior during execution.

The provided Python function, `find_voice`, is designed to locate the necessary files for a specific voice model in a text-to-speech (TTS) system. It searches through a list of directories (`data_dirs`) to find two files associated with the given voice name: an ONNX model file (`.onnx`) and its corresponding configuration file (`.onnx.json`). These files are essential for loading and using the voice model in the TTS pipeline.

The function takes two parameters: `name`, a string representing the name of the voice, and `data_dirs`, an iterable containing paths to directories where the files might be located. Each directory in `data_dirs` is converted into a `Path` object using Python's `pathlib` module, which provides a convenient and platform-independent way to handle file paths.

For each directory, the function constructs the expected paths for the ONNX model file and its configuration file by appending the voice name with the appropriate extensions (`.onnx` and `.onnx.json`). It then checks if both files exist in the directory using the `exists` method of the `Path` object. If both files are found, the function immediately returns a tuple containing the paths to the ONNX model and its configuration file.

If the function iterates through all the directories in `data_dirs` without finding the required files, it raises a `ValueError` with a descriptive error message indicating that the files for the specified voice are missing. This ensures that the caller is informed of the issue and can take corrective action, such as verifying the voice name or the directory paths.

In summary, `find_voice` is a utility function that simplifies the process of locating voice model files in a TTS system. It ensures that both the model and its configuration are present before proceeding, making it a critical step in initializing and using a voice model. The use of `pathlib` enhances the code's readability and cross-platform compatibility.

The `ensure_voice_exists` function is a utility designed to ensure that all necessary files for a specific voice model in a text-to-speech (TTS) system are available locally. If any required files are missing or corrupted, the function downloads them from a remote server. This ensures that the TTS system can operate reliably without manual intervention to manage voice files.

The function begins by validating its inputs. It asserts that the `data_dirs` parameter, which specifies directories to search for voice files, is not empty. It also checks if the requested voice name exists in the `voices_info` dictionary, which contains metadata about available voices. If the voice is not found, a custom `VoiceNotFoundError` exception is raised.

The metadata for the requested voice is retrieved from `voices_info`, including a list of required files and their properties (e.g., size and MD5 hash). The function iterates over the provided `data_dirs` to locate these files. For each file, it checks whether the file exists in the directory. If a file is missing, it is added to a `files_to_download` set. If the file exists, its size and MD5 hash are compared against the expected values from the metadata. Any discrepancies (e.g., incorrect size or hash) result in the file being marked for download.

If no files are found locally and no files are marked for download, the function raises a `ValueError`, indicating that the voice cannot be located or downloaded. This acts as a safeguard to ensure that the function does not proceed with incomplete or invalid data.

For files that need to be downloaded, the function constructs their URLs using a predefined format (`URL_FORMAT`) and downloads them to the specified `download_dir`. The directory structure is created if it does not already exist. The `urlopen` function is used to fetch the file from the remote server, and the file is saved locally using `shutil.copyfileobj`. After each successful download, a log message is generated to indicate the completion of the operation.

Throughout the process, the function uses logging extensively to provide debug and warning messages. These logs help track the status of file checks, missing files, and download operations, making it easier to diagnose issues during execution. By combining local file validation with remote downloads, the `ensure_voice_exists` function ensures that the TTS system has all the resources it needs to function correctly.


## Source Code

```py
"""Utility for downloading Piper voices."""
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Set, Tuple, Union
from urllib.request import urlopen

from .file_hash import get_file_hash

URL_FORMAT = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/{file}"

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)

_SKIP_FILES = {"MODEL_CARD"}


class VoiceNotFoundError(Exception):
    pass


def get_voices(
    download_dir: Union[str, Path], update_voices: bool = False
) -> Dict[str, Any]:
    """Loads available voices from downloaded or embedded JSON file."""
    download_dir = Path(download_dir)
    voices_download = download_dir / "voices.json"

    if update_voices:
        # Download latest voices.json
        voices_url = URL_FORMAT.format(file="voices.json")
        _LOGGER.debug("Downloading %s to %s", voices_url, voices_download)
        with urlopen(voices_url) as response, open(
            voices_download, "wb"
        ) as download_file:
            shutil.copyfileobj(response, download_file)

    # Prefer downloaded file to embedded
    voices_embedded = _DIR / "voices.json"
    voices_path = voices_download if voices_download.exists() else voices_embedded

    _LOGGER.debug("Loading %s", voices_path)
    with open(voices_path, "r", encoding="utf-8") as voices_file:
        return json.load(voices_file)


def ensure_voice_exists(
    name: str,
    data_dirs: Iterable[Union[str, Path]],
    download_dir: Union[str, Path],
    voices_info: Dict[str, Any],
):
    assert data_dirs, "No data dirs"
    if name not in voices_info:
        raise VoiceNotFoundError(name)

    voice_info = voices_info[name]
    voice_files = voice_info["files"]
    files_to_download: Set[str] = set()

    for data_dir in data_dirs:
        data_dir = Path(data_dir)

        # Check sizes/hashes
        for file_path, file_info in voice_files.items():
            if file_path in files_to_download:
                # Already planning to download
                continue

            file_name = Path(file_path).name
            if file_name in _SKIP_FILES:
                continue

            data_file_path = data_dir / file_name
            _LOGGER.debug("Checking %s", data_file_path)
            if not data_file_path.exists():
                _LOGGER.debug("Missing %s", data_file_path)
                files_to_download.add(file_path)
                continue

            expected_size = file_info["size_bytes"]
            actual_size = data_file_path.stat().st_size
            if expected_size != actual_size:
                _LOGGER.warning(
                    "Wrong size (expected=%s, actual=%s) for %s",
                    expected_size,
                    actual_size,
                    data_file_path,
                )
                files_to_download.add(file_path)
                continue

            expected_hash = file_info["md5_digest"]
            actual_hash = get_file_hash(data_file_path)
            if expected_hash != actual_hash:
                _LOGGER.warning(
                    "Wrong hash (expected=%s, actual=%s) for %s",
                    expected_hash,
                    actual_hash,
                    data_file_path,
                )
                files_to_download.add(file_path)
                continue

    if (not voice_files) and (not files_to_download):
        raise ValueError(f"Unable to find or download voice: {name}")

    # Download missing files
    download_dir = Path(download_dir)

    for file_path in files_to_download:
        file_name = Path(file_path).name
        if file_name in _SKIP_FILES:
            continue

        file_url = URL_FORMAT.format(file=file_path)
        download_file_path = download_dir / file_name
        download_file_path.parent.mkdir(parents=True, exist_ok=True)

        _LOGGER.debug("Downloading %s to %s", file_url, download_file_path)
        with urlopen(file_url) as response, open(
            download_file_path, "wb"
        ) as download_file:
            shutil.copyfileobj(response, download_file)

        _LOGGER.info("Downloaded %s (%s)", download_file_path, file_url)


def find_voice(name: str, data_dirs: Iterable[Union[str, Path]]) -> Tuple[Path, Path]:
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        onnx_path = data_dir / f"{name}.onnx"
        config_path = data_dir / f"{name}.onnx.json"

        if onnx_path.exists() and config_path.exists():
            return onnx_path, config_path

    raise ValueError(f"Missing files for voice {name}")

```
