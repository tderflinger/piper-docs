---
title: check_phonemes.py
description: check_phonemes.py
---

## Code Explained

The provided Python code is a utility script designed to analyze phoneme usage in a dataset. Phonemes are the smallest units of sound in speech, and this script processes input data to identify which phonemes are used and which are missing from a predefined mapping (`DEFAULT_PHONEME_ID_MAP`). The script reads JSON-formatted input line-by-line from standard input (`sys.stdin`), making it suitable for use in a pipeline or as part of a larger text-to-speech (TTS) preprocessing workflow.

The script begins by initializing two `Counter` objects, `used_phonemes` and `missing_phonemes`. These counters are used to track the frequency of phonemes encountered in the input data. The `used_phonemes` counter keeps a tally of all phonemes found, while the `missing_phonemes` counter specifically tracks phonemes that are not present in the `DEFAULT_PHONEME_ID_MAP`.

The main loop processes each line of input from `sys.stdin`. Each line is stripped of leading and trailing whitespace, and empty lines are skipped. The script assumes that each line contains a JSON object, which is parsed using `json.loads`. The parsed object is expected to have a key `"phonemes"` containing a list of phonemes. For each phoneme in this list, the script increments its count in the `used_phonemes` counter. If the phoneme is not found in `DEFAULT_PHONEME_ID_MAP`, it is also added to the `missing_phonemes` counter.

After processing all input lines, the script checks if there are any missing phonemes. If so, it prints a message to standard error (`sys.stderr`) indicating the number of missing phonemes. This provides immediate feedback to the user about potential issues with the dataset or phoneme mapping.

Finally, the script outputs a JSON object to standard output (`sys.stdout`) containing detailed information about the phonemes. The output includes two sections: `"used"` and `"missing"`. Each section lists the phonemes along with their frequency (`"count"`), Unicode hexadecimal representation (`"hex"`), and Unicode category (`"category"`). This information is generated using Python's `unicodedata` module, which provides utilities for working with Unicode characters. The phonemes are sorted by frequency, with the most common ones appearing first.

In summary, this script is a diagnostic tool for analyzing phoneme usage in a dataset. It helps identify missing phonemes and provides detailed metadata about the phonemes encountered. This information can be used to refine the phoneme mapping, improve the dataset, or debug issues in a TTS pipeline. The use of `Counter` objects and JSON input/output makes the script efficient and easy to integrate into larger workflows.


## Source Code

```py
#!/usr/bin/env python3
import json
import sys
import unicodedata
from collections import Counter

from .phonemize import DEFAULT_PHONEME_ID_MAP


def main() -> None:
    used_phonemes: "Counter[str]" = Counter()
    missing_phonemes: "Counter[str]" = Counter()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        utt = json.loads(line)
        for phoneme in utt["phonemes"]:
            used_phonemes[phoneme] += 1

            if phoneme not in DEFAULT_PHONEME_ID_MAP:
                missing_phonemes[phoneme] += 1

    if missing_phonemes:
        print("Missing", len(missing_phonemes), "phoneme(s)", file=sys.stderr)

    json.dump(
        {
            "used": {
                phoneme: {
                    "count": count,
                    "hex": f"\\u{hex(ord(phoneme))}",
                    "name": unicodedata.category(phoneme),
                    "category": unicodedata.category(phoneme),
                }
                for phoneme, count in used_phonemes.most_common()
            },
            "missing": {
                phoneme: {
                    "count": count,
                    "hex": f"\\u{hex(ord(phoneme))}",
                    "name": unicodedata.category(phoneme),
                    "category": unicodedata.category(phoneme),
                }
                for phoneme, count in missing_phonemes.most_common()
            },
        },
        sys.stdout,
    )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
```
