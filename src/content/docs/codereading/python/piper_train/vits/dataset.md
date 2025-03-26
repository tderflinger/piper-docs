---
title: dataset.py
description: dataset.py
---

## Code Explained

The provided code defines a pipeline for managing and processing datasets in a text-to-speech (TTS) system. It includes classes and methods for representing individual utterances, converting them into tensors, batching them for model training, and handling dataset loading and collation. Below is a detailed explanation of the key components:

---

### **1. `Utterance` Class**
The `Utterance` class represents a single data point in the dataset. It includes:
- **`phoneme_ids`**: A list of integers representing the phonemes in the utterance.
- **`audio_norm_path`**: A file path to the normalized audio waveform.
- **`audio_spec_path`**: A file path to the spectrogram of the audio.
- **`speaker_id`**: (Optional) An identifier for the speaker, useful in multi-speaker datasets.
- **`text`**: (Optional) The text transcription of the utterance.

This class serves as a lightweight container for metadata and file paths associated with an utterance.

---

### **2. `UtteranceTensors` Class**
The `UtteranceTensors` class converts the `Utterance` data into PyTorch tensors for model training. It includes:
- **`phoneme_ids`**: A `LongTensor` of phoneme IDs.
- **`spectrogram`**: A `FloatTensor` representing the mel spectrogram.
- **`audio_norm`**: A `FloatTensor` of the normalized audio waveform.
- **`speaker_id`**: (Optional) A `LongTensor` for the speaker ID.
- **`text`**: (Optional) The text transcription.

The `spec_length` property computes the length of the spectrogram (number of time steps), which is useful for padding and batching.

---

### **3. `Batch` Class**
The `Batch` class represents a batch of utterances prepared for training. It includes:
- **`phoneme_ids`**: A padded `LongTensor` of phoneme IDs for all utterances in the batch.
- **`phoneme_lengths`**: A `LongTensor` of the lengths of phoneme sequences.
- **`spectrograms`**: A padded `FloatTensor` of spectrograms.
- **`spectrogram_lengths`**: A `LongTensor` of spectrogram lengths.
- **`audios`**: A padded `FloatTensor` of audio waveforms.
- **`audio_lengths`**: A `LongTensor` of audio lengths.
- **`speaker_ids`**: (Optional) A `LongTensor` of speaker IDs for multi-speaker datasets.

This class ensures that all data in a batch is properly padded and aligned for efficient processing by the model.

---

### **4. `PiperDataset` Class**
The `PiperDataset` class handles dataset loading and provides access to individual utterances. Key methods include:
- **`__init__`**: Loads the dataset from one or more file paths. Each file is expected to contain JSON lines, where each line represents an utterance.
- **`__len__`**: Returns the number of utterances in the dataset.
- **`__getitem__`**: Converts an `Utterance` object into an `UtteranceTensors` object by loading the corresponding audio and spectrogram files.

#### Static Methods:
- **`load_dataset`**: Reads a dataset file line by line, parses each line into an `Utterance` object, and skips utterances exceeding the `max_phoneme_ids` limit.
- **`load_utterance`**: Parses a single JSON line into an `Utterance` object.

This class abstracts the dataset format and provides a PyTorch-compatible interface for data loading.

---

### **5. `UtteranceCollate` Class**
The `UtteranceCollate` class is responsible for collating a list of `UtteranceTensors` into a `Batch`. Key steps include:
1. **Determine Maximum Lengths**: Computes the maximum lengths of phoneme sequences, spectrograms, and audio waveforms in the batch.
2. **Create Padded Tensors**: Initializes zero-padded tensors for phonemes, spectrograms, and audio waveforms.
3. **Sort and Populate**: Sorts utterances by spectrogram length (for efficiency) and populates the padded tensors with data from each utterance.
4. **Handle Multi-Speaker Data**: Ensures that speaker IDs are included if the dataset is multi-speaker.

The `__call__` method returns a `Batch` object, ready for input into a model.

---

### **Key Features**
1. **Modular Design**: The separation of `Utterance`, `UtteranceTensors`, and `Batch` ensures clear boundaries between raw data, tensorized data, and batched data.
2. **Dataset Flexibility**: The `PiperDataset` class supports multi-file datasets and handles optional fields like speaker IDs and text.
3. **Efficient Collation**: The `UtteranceCollate` class ensures that batches are efficiently padded and sorted, minimizing computational overhead during training.
4. **Error Handling**: The `load_dataset` method logs and skips invalid or oversized utterances, ensuring robustness.

---

### **Use Case**
This pipeline is designed for text-to-speech (TTS) systems, where phoneme sequences, spectrograms, and audio waveforms are the primary inputs and outputs. It supports both single-speaker and multi-speaker datasets, making it suitable for a wide range of TTS applications, including voice cloning and multi-speaker synthesis.

## Source Code

```py
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import torch
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset

_LOGGER = logging.getLogger("vits.dataset")


@dataclass
class Utterance:
    phoneme_ids: List[int]
    audio_norm_path: Path
    audio_spec_path: Path
    speaker_id: Optional[int] = None
    text: Optional[str] = None


@dataclass
class UtteranceTensors:
    phoneme_ids: LongTensor
    spectrogram: FloatTensor
    audio_norm: FloatTensor
    speaker_id: Optional[LongTensor] = None
    text: Optional[str] = None

    @property
    def spec_length(self) -> int:
        return self.spectrogram.size(1)


@dataclass
class Batch:
    phoneme_ids: LongTensor
    phoneme_lengths: LongTensor
    spectrograms: FloatTensor
    spectrogram_lengths: LongTensor
    audios: FloatTensor
    audio_lengths: LongTensor
    speaker_ids: Optional[LongTensor] = None


class PiperDataset(Dataset):
    """
    Dataset format:

    * phoneme_ids (required)
    * audio_norm_path (required)
    * audio_spec_path (required)
    * text (optional)
    * phonemes (optional)
    * audio_path (optional)
    """

    def __init__(
        self,
        dataset_paths: List[Union[str, Path]],
        max_phoneme_ids: Optional[int] = None,
    ):
        self.utterances: List[Utterance] = []

        for dataset_path in dataset_paths:
            dataset_path = Path(dataset_path)
            _LOGGER.debug("Loading dataset: %s", dataset_path)
            self.utterances.extend(
                PiperDataset.load_dataset(dataset_path, max_phoneme_ids=max_phoneme_ids)
            )

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx) -> UtteranceTensors:
        utt = self.utterances[idx]
        return UtteranceTensors(
            phoneme_ids=LongTensor(utt.phoneme_ids),
            audio_norm=torch.load(utt.audio_norm_path),
            spectrogram=torch.load(utt.audio_spec_path),
            speaker_id=LongTensor([utt.speaker_id])
            if utt.speaker_id is not None
            else None,
            text=utt.text,
        )

    @staticmethod
    def load_dataset(
        dataset_path: Path,
        max_phoneme_ids: Optional[int] = None,
    ) -> Iterable[Utterance]:
        num_skipped = 0

        with open(dataset_path, "r", encoding="utf-8") as dataset_file:
            for line_idx, line in enumerate(dataset_file):
                line = line.strip()
                if not line:
                    continue

                try:
                    utt = PiperDataset.load_utterance(line)
                    if (max_phoneme_ids is None) or (
                        len(utt.phoneme_ids) <= max_phoneme_ids
                    ):
                        yield utt
                    else:
                        num_skipped += 1
                except Exception:
                    _LOGGER.exception(
                        "Error on line %s of %s: %s",
                        line_idx + 1,
                        dataset_path,
                        line,
                    )

        if num_skipped > 0:
            _LOGGER.warning("Skipped %s utterance(s)", num_skipped)

    @staticmethod
    def load_utterance(line: str) -> Utterance:
        utt_dict = json.loads(line)
        return Utterance(
            phoneme_ids=utt_dict["phoneme_ids"],
            audio_norm_path=Path(utt_dict["audio_norm_path"]),
            audio_spec_path=Path(utt_dict["audio_spec_path"]),
            speaker_id=utt_dict.get("speaker_id"),
            text=utt_dict.get("text"),
        )


class UtteranceCollate:
    def __init__(self, is_multispeaker: bool, segment_size: int):
        self.is_multispeaker = is_multispeaker
        self.segment_size = segment_size

    def __call__(self, utterances: Sequence[UtteranceTensors]) -> Batch:
        num_utterances = len(utterances)
        assert num_utterances > 0, "No utterances"

        max_phonemes_length = 0
        max_spec_length = 0
        max_audio_length = 0

        num_mels = 0

        # Determine lengths
        for utt_idx, utt in enumerate(utterances):
            assert utt.spectrogram is not None
            assert utt.audio_norm is not None

            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(1)

            max_phonemes_length = max(max_phonemes_length, phoneme_length)
            max_spec_length = max(max_spec_length, spec_length)
            max_audio_length = max(max_audio_length, audio_length)

            num_mels = utt.spectrogram.size(0)
            if self.is_multispeaker:
                assert utt.speaker_id is not None, "Missing speaker id"

        # Audio cannot be smaller than segment size (8192)
        max_audio_length = max(max_audio_length, self.segment_size)

        # Create padded tensors
        phonemes_padded = LongTensor(num_utterances, max_phonemes_length)
        spec_padded = FloatTensor(num_utterances, num_mels, max_spec_length)
        audio_padded = FloatTensor(num_utterances, 1, max_audio_length)

        phonemes_padded.zero_()
        spec_padded.zero_()
        audio_padded.zero_()

        phoneme_lengths = LongTensor(num_utterances)
        spec_lengths = LongTensor(num_utterances)
        audio_lengths = LongTensor(num_utterances)

        speaker_ids: Optional[LongTensor] = None
        if self.is_multispeaker:
            speaker_ids = LongTensor(num_utterances)

        # Sort by decreasing spectrogram length
        sorted_utterances = sorted(
            utterances, key=lambda u: u.spectrogram.size(1), reverse=True
        )
        for utt_idx, utt in enumerate(sorted_utterances):
            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(1)

            phonemes_padded[utt_idx, :phoneme_length] = utt.phoneme_ids
            phoneme_lengths[utt_idx] = phoneme_length

            spec_padded[utt_idx, :, :spec_length] = utt.spectrogram
            spec_lengths[utt_idx] = spec_length

            audio_padded[utt_idx, :, :audio_length] = utt.audio_norm
            audio_lengths[utt_idx] = audio_length

            if utt.speaker_id is not None:
                assert speaker_ids is not None
                speaker_ids[utt_idx] = utt.speaker_id

        return Batch(
            phoneme_ids=phonemes_padded,
            phoneme_lengths=phoneme_lengths,
            spectrograms=spec_padded,
            spectrogram_lengths=spec_lengths,
            audios=audio_padded,
            audio_lengths=audio_lengths,
            speaker_ids=speaker_ids,
        )
```