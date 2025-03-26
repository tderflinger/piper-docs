---
title: preprocess.py
description: preprocess.py
---

## Code Explained

The provided Python script is a comprehensive preprocessing pipeline for preparing audio datasets for text-to-speech (TTS) training. It supports multiple dataset formats, phoneme generation methods, and configurations for single or multi-speaker datasets. Below is a detailed explanation of its key components:

---

### **PhonemeType Enum**
The `PhonemeType` class defines two types of phoneme sources:
1. `ESPEAK`: Phonemes generated using the `espeak-ng` tool.
2. `TEXT`: Phonemes derived directly from the text itself.

This enum is used to determine the phoneme generation method during preprocessing.

---

### **Main Function**
The `main` function orchestrates the entire preprocessing workflow. It:
1. **Parses Command-Line Arguments**: Defines arguments for input/output directories, dataset format, phoneme type, text casing, and other configurations. It also validates incompatible options like `--single-speaker` and `--speaker-id`.
2. **Configures Logging**: Sets the logging level based on the `--debug` flag and suppresses unnecessary logs from external libraries like `numba`.
3. **Prepares Directories**: Converts input/output paths to `Path` objects and ensures the output and cache directories exist.
4. **Selects Dataset Format**: Chooses between `ljspeech_dataset` and `mycroft_dataset` based on the `--dataset-format` argument.
5. **Counts Speakers and Utterances**: Iterates through the dataset to count the number of speakers and utterances, determining whether the dataset is single-speaker or multi-speaker.
6. **Writes Configuration**: Generates a `config.json` file containing dataset metadata, phoneme mappings, and other settings required for training.
7. **Initializes Workers**: Spawns multiple worker processes to parallelize phoneme generation and audio preprocessing. The number of workers is determined by the `--max-workers` argument or the number of CPU cores.
8. **Processes Utterances**: Reads batches of utterances, processes them using the selected phoneme generation method (`phonemize_batch_espeak` or `phonemize_batch_text`), and writes the results to a `dataset.jsonl` file.
9. **Handles Missing Phonemes**: Logs warnings for any missing phonemes encountered during processing.
10. **Cleans Up Workers**: Signals worker processes to stop and waits for them to terminate.

---

### **Phoneme Generation**
The script supports two phoneme generation methods:
1. **`phonemize_batch_espeak`**: Uses `espeak-ng` to generate phonemes for each utterance. It also supports diacritization of Arabic text using the `--tashkeel` flag.
2. **`phonemize_batch_text`**: Generates phonemes directly from the text using Unicode codepoints.

Both methods handle text casing (e.g., lowercasing, uppercasing) and silence detection for audio preprocessing.

---

### **Dataset Handling**
The script supports two dataset formats:
1. **LJSpeech**: Expects a `metadata.csv` file containing `filename|speaker|text` rows. The speaker column is optional for single-speaker datasets.
2. **Mycroft**: Expects multiple `*-metadata.txt` files, each containing `filename|text|length` rows. The parent directory name is used as the speaker ID for multi-speaker datasets.

The `ljspeech_dataset` and `mycroft_dataset` functions yield `Utterance` objects for each valid entry in the dataset.

---

### **Utterance Class**
The `Utterance` class represents a single entry in the dataset. It includes attributes for the text, audio file path, speaker information, phonemes, and preprocessed audio paths. It also tracks missing phonemes encountered during processing.

---

### **Utility Functions**
1. **`get_text_casing`**: Returns a function to apply the specified text casing (`lower`, `upper`, `casefold`, or ignore).
2. **`batched`**: Splits an iterable into batches of a specified size, ensuring efficient processing of large datasets.
3. **`PathEncoder`**: A custom JSON encoder that converts `Path` objects to strings for serialization.

---

### **Parallel Processing**
The script uses `multiprocessing` to parallelize phoneme generation and audio preprocessing. It creates a `JoinableQueue` for input batches and a `Queue` for processed utterances. Worker processes consume batches from the input queue, process them, and place the results in the output queue.

---

### **Output Files**
1. **`config.json`**: Contains metadata about the dataset, phoneme mappings, and training configurations.
2. **`dataset.jsonl`**: A JSONL (JSON Lines) file where each line represents a processed utterance, including phonemes, speaker IDs, and audio paths.

---

### **Key Features**
1. **Multi-Speaker Support**: Automatically detects and assigns speaker IDs for multi-speaker datasets.
2. **Phoneme Flexibility**: Supports both `espeak-ng` and text-based phoneme generation.
3. **Parallel Processing**: Leverages multiple CPU cores for efficient preprocessing.
4. **Customizable Text Casing**: Allows users to specify how text should be cased before phoneme generation.
5. **Error Handling**: Logs warnings for missing phonemes and skips invalid utterances.

---

### **Use Case**
This script is ideal for preparing audio datasets for TTS training. It handles dataset parsing, phoneme generation, and audio preprocessing, producing a ready-to-use dataset for training models like VITS or Tacotron. Its flexibility and parallel processing capabilities make it suitable for both small and large-scale datasets.

## Source Code

```py
#!/usr/bin/env python3
import argparse
import csv
import dataclasses
import itertools
import json
import logging
import os
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import JoinableQueue, Process, Queue
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from piper_phonemize import (
    phonemize_espeak,
    phonemize_codepoints,
    phoneme_ids_espeak,
    phoneme_ids_codepoints,
    get_codepoints_map,
    get_espeak_map,
    get_max_phonemes,
    tashkeel_run,
)

from .norm_audio import cache_norm_audio, make_silence_detector

_DIR = Path(__file__).parent
_VERSION = (_DIR / "VERSION").read_text(encoding="utf-8").strip()
_LOGGER = logging.getLogger("preprocess")


class PhonemeType(str, Enum):
    ESPEAK = "espeak"
    """Phonemes come from espeak-ng"""

    TEXT = "text"
    """Phonemes come from text itself"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", required=True, help="Directory with audio dataset"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output files for training",
    )
    parser.add_argument("--language", required=True, help="eSpeak-ng voice")
    parser.add_argument(
        "--sample-rate",
        type=int,
        required=True,
        help="Target sample rate for voice (hertz)",
    )
    parser.add_argument(
        "--dataset-format", choices=("ljspeech", "mycroft"), required=True
    )
    parser.add_argument("--cache-dir", help="Directory to cache processed audio files")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument(
        "--single-speaker", action="store_true", help="Force single speaker dataset"
    )
    parser.add_argument(
        "--speaker-id", type=int, help="Add speaker id to single speaker dataset"
    )
    #
    parser.add_argument(
        "--phoneme-type",
        choices=list(PhonemeType),
        default=PhonemeType.ESPEAK,
        help="Type of phonemes to use (default: espeak)",
    )
    parser.add_argument(
        "--text-casing",
        choices=("ignore", "lower", "upper", "casefold"),
        default="ignore",
        help="Casing applied to utterance text",
    )
    #
    parser.add_argument(
        "--dataset-name",
        help="Name of dataset to put in config (default: name of <ouput_dir>/../)",
    )
    parser.add_argument(
        "--audio-quality",
        help="Audio quality to put in config (default: name of <output_dir>)",
    )
    #
    parser.add_argument(
        "--tashkeel",
        action="store_true",
        help="Diacritize Arabic text with libtashkeel",
    )
    #
    parser.add_argument(
        "--skip-audio", action="store_true", help="Don't preprocess audio"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.single_speaker and (args.speaker_id is not None):
        _LOGGER.fatal("--single-speaker and --speaker-id cannot both be provided")
        return

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)

    # Prevent log spam
    logging.getLogger("numba").setLevel(logging.WARNING)

    # Ensure enum
    args.phoneme_type = PhonemeType(args.phoneme_type)

    # Convert to paths and create output directories
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else args.output_dir / "cache" / str(args.sample_rate)
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_format == "mycroft":
        make_dataset = mycroft_dataset
    else:
        make_dataset = ljspeech_dataset

    # Count speakers
    _LOGGER.debug("Counting number of speakers/utterances in the dataset")
    speaker_counts: "Counter[str]" = Counter()
    num_utterances = 0
    for utt in make_dataset(args):
        speaker = utt.speaker or ""
        speaker_counts[speaker] += 1
        num_utterances += 1

    assert num_utterances > 0, "No utterances found"

    is_multispeaker = len(speaker_counts) > 1
    speaker_ids: Dict[str, int] = {}

    if is_multispeaker:
        _LOGGER.info("%s speakers detected", len(speaker_counts))

        # Assign speaker ids by most number of utterances first
        for speaker_id, (speaker, _speaker_count) in enumerate(
            speaker_counts.most_common()
        ):
            speaker_ids[speaker] = speaker_id
    else:
        _LOGGER.info("Single speaker dataset")

    # Write config
    audio_quality = args.audio_quality or args.output_dir.name
    dataset_name = args.dataset_name or args.output_dir.parent.name

    with open(args.output_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(
            {
                "dataset": dataset_name,
                "audio": {
                    "sample_rate": args.sample_rate,
                    "quality": audio_quality,
                },
                "espeak": {
                    "voice": args.language,
                },
                "language": {
                    "code": args.language,
                },
                "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
                "phoneme_type": args.phoneme_type.value,
                "phoneme_map": {},
                "phoneme_id_map": get_codepoints_map()[args.language]
                if args.phoneme_type == PhonemeType.TEXT
                else get_espeak_map(),
                "num_symbols": get_max_phonemes(),
                "num_speakers": len(speaker_counts),
                "speaker_id_map": speaker_ids,
                "piper_version": _VERSION,
            },
            config_file,
            ensure_ascii=False,
            indent=4,
        )
    _LOGGER.info("Wrote dataset config")

    if (args.max_workers is None) or (args.max_workers < 1):
        args.max_workers = os.cpu_count()

    assert args.max_workers is not None

    batch_size = int(num_utterances / (args.max_workers * 2))
    queue_in: "Queue[Iterable[Utterance]]" = JoinableQueue()
    queue_out: "Queue[Optional[Utterance]]" = Queue()

    # Start workers
    if args.phoneme_type == PhonemeType.TEXT:
        target = phonemize_batch_text
    else:
        target = phonemize_batch_espeak

    processes = [
        Process(target=target, args=(args, queue_in, queue_out))
        for _ in range(args.max_workers)
    ]
    for proc in processes:
        proc.start()

    _LOGGER.info(
        "Processing %s utterance(s) with %s worker(s)", num_utterances, args.max_workers
    )
    with open(args.output_dir / "dataset.jsonl", "w", encoding="utf-8") as dataset_file:
        for utt_batch in batched(
            make_dataset(args),
            batch_size,
        ):
            queue_in.put(utt_batch)

        _LOGGER.debug("Waiting for jobs to finish")
        missing_phonemes: "Counter[str]" = Counter()
        for _ in range(num_utterances):
            utt = queue_out.get()
            if utt is not None:
                if utt.speaker is not None:
                    utt.speaker_id = speaker_ids[utt.speaker]

                utt_dict = dataclasses.asdict(utt)
                utt_dict.pop("missing_phonemes")

                # JSONL
                json.dump(
                    utt_dict,
                    dataset_file,
                    ensure_ascii=False,
                    cls=PathEncoder,
                )
                print("", file=dataset_file)

                missing_phonemes.update(utt.missing_phonemes)

        if missing_phonemes:
            for phoneme, count in missing_phonemes.most_common():
                _LOGGER.warning("Missing %s (%s)", phoneme, count)

            _LOGGER.warning("Missing %s phoneme(s)", len(missing_phonemes))

    # Signal workers to stop
    for proc in processes:
        queue_in.put(None)

    # Wait for workers to stop
    for proc in processes:
        proc.join(timeout=1)


# -----------------------------------------------------------------------------


def get_text_casing(casing: str):
    if casing == "lower":
        return str.lower

    if casing == "upper":
        return str.upper

    if casing == "casefold":
        return str.casefold

    return lambda s: s


def phonemize_batch_espeak(
    args: argparse.Namespace, queue_in: JoinableQueue, queue_out: Queue
):
    try:
        casing = get_text_casing(args.text_casing)
        silence_detector = make_silence_detector()

        while True:
            utt_batch = queue_in.get()
            if utt_batch is None:
                break

            for utt in utt_batch:
                try:
                    if args.tashkeel:
                        utt.text = tashkeel_run(utt.text)

                    _LOGGER.debug(utt)
                    all_phonemes = phonemize_espeak(casing(utt.text), args.language)

                    # Flatten
                    utt.phonemes = [
                        phoneme
                        for sentence_phonemes in all_phonemes
                        for phoneme in sentence_phonemes
                    ]
                    utt.phoneme_ids = phoneme_ids_espeak(
                        utt.phonemes,
                        missing_phonemes=utt.missing_phonemes,
                    )
                    if not args.skip_audio:
                        utt.audio_norm_path, utt.audio_spec_path = cache_norm_audio(
                            utt.audio_path,
                            args.cache_dir,
                            silence_detector,
                            args.sample_rate,
                        )
                    queue_out.put(utt)
                except TimeoutError:
                    _LOGGER.error("Skipping utterance due to timeout: %s", utt)
                except Exception:
                    _LOGGER.exception("Failed to process utterance: %s", utt)
                    queue_out.put(None)

            queue_in.task_done()
    except Exception:
        _LOGGER.exception("phonemize_batch_espeak")


def phonemize_batch_text(
    args: argparse.Namespace, queue_in: JoinableQueue, queue_out: Queue
):
    try:
        casing = get_text_casing(args.text_casing)
        silence_detector = make_silence_detector()

        while True:
            utt_batch = queue_in.get()
            if utt_batch is None:
                break

            for utt in utt_batch:
                try:
                    if args.tashkeel:
                        utt.text = tashkeel_run(utt.text)

                    _LOGGER.debug(utt)
                    all_phonemes = phonemize_codepoints(casing(utt.text))
                    # Flatten
                    utt.phonemes = [
                        phoneme
                        for sentence_phonemes in all_phonemes
                        for phoneme in sentence_phonemes
                    ]
                    utt.phoneme_ids = phoneme_ids_codepoints(
                        args.language,
                        utt.phonemes,
                        missing_phonemes=utt.missing_phonemes,
                    )
                    if not args.skip_audio:
                        utt.audio_norm_path, utt.audio_spec_path = cache_norm_audio(
                            utt.audio_path,
                            args.cache_dir,
                            silence_detector,
                            args.sample_rate,
                        )
                    queue_out.put(utt)
                except TimeoutError:
                    _LOGGER.error("Skipping utterance due to timeout: %s", utt)
                except Exception:
                    _LOGGER.exception("Failed to process utterance: %s", utt)
                    queue_out.put(None)

            queue_in.task_done()
    except Exception:
        _LOGGER.exception("phonemize_batch_text")


# -----------------------------------------------------------------------------


@dataclass
class Utterance:
    text: str
    audio_path: Path
    speaker: Optional[str] = None
    speaker_id: Optional[int] = None
    phonemes: Optional[List[str]] = None
    phoneme_ids: Optional[List[int]] = None
    audio_norm_path: Optional[Path] = None
    audio_spec_path: Optional[Path] = None
    missing_phonemes: "Counter[str]" = field(default_factory=Counter)


class PathEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def ljspeech_dataset(args: argparse.Namespace) -> Iterable[Utterance]:
    dataset_dir = args.input_dir
    is_single_speaker = args.single_speaker
    speaker_id = args.speaker_id
    skip_audio = args.skip_audio

    # filename|speaker|text
    # speaker is optional
    metadata_path = dataset_dir / "metadata.csv"
    assert metadata_path.exists(), f"Missing {metadata_path}"

    wav_dir = dataset_dir / "wav"
    if not wav_dir.is_dir():
        wav_dir = dataset_dir / "wavs"

    with open(metadata_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter="|")
        for row in reader:
            assert len(row) >= 2, "Not enough columns"

            speaker: Optional[str] = None
            if is_single_speaker or (len(row) == 2):
                filename, text = row[0], row[-1]
            else:
                filename, speaker, text = row[0], row[1], row[-1]

            # Try file name relative to metadata
            wav_path = metadata_path.parent / filename

            if not wav_path.exists():
                # Try with .wav
                wav_path = metadata_path.parent / f"{filename}.wav"

            if not wav_path.exists():
                # Try wav/ or wavs/
                wav_path = wav_dir / filename

            if not wav_path.exists():
                # Try with .wav
                wav_path = wav_dir / f"{filename}.wav"

            if not skip_audio:
                if not wav_path.exists():
                    _LOGGER.warning("Missing %s", filename)
                    continue

                if wav_path.stat().st_size == 0:
                    _LOGGER.warning("Empty file: %s", wav_path)
                    continue

            yield Utterance(
                text=text, audio_path=wav_path, speaker=speaker, speaker_id=speaker_id
            )


def mycroft_dataset(args: argparse.Namespace) -> Iterable[Utterance]:
    dataset_dir = args.input_dir
    is_single_speaker = args.single_speaker
    skip_audio = args.skip_audio

    speaker_id = 0
    for metadata_path in dataset_dir.glob("**/*-metadata.txt"):
        speaker = metadata_path.parent.name if not is_single_speaker else None
        with open(metadata_path, "r", encoding="utf-8") as csv_file:
            # filename|text|length
            reader = csv.reader(csv_file, delimiter="|")
            for row in reader:
                filename, text = row[0], row[1]
                wav_path = metadata_path.parent / filename
                if skip_audio or (wav_path.exists() and (wav_path.stat().st_size > 0)):
                    yield Utterance(
                        text=text,
                        audio_path=wav_path,
                        speaker=speaker,
                        speaker_id=speaker_id if not is_single_speaker else None,
                    )
        speaker_id += 1


# -----------------------------------------------------------------------------


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    batch = list(itertools.islice(it, n))
    while batch:
        yield batch
        batch = list(itertools.islice(it, n))


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
```