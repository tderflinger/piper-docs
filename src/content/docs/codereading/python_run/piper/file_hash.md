---
title: file_hash.py
description: file_hash.py
---

## Code Explained

The provided Python function, `get_file_hash`, computes the MD5 hash of a file in chunks. Hashing is a process of generating a fixed-size string (the hash) from input data, which is commonly used to verify file integrity or uniquely identify files. This function is particularly useful for hashing large files efficiently, as it processes the file in smaller chunks rather than loading the entire file into memory.

The function takes two parameters: `path` and `bytes_per_chunk`. The `path` parameter specifies the file to be hashed and can be provided as either a string or a `Path` object from the `pathlib` module. The `bytes_per_chunk` parameter determines the size of each chunk to be read from the file, with a default value of 8192 bytes (8 KB). This chunked approach minimizes memory usage, making the function suitable for large files.

Inside the function, an MD5 hash object is created using `hashlib.md5()`. This object will be used to incrementally compute the hash as chunks of the file are read. The file is then opened in binary mode (`"rb"`) to ensure that the raw byte data is read, which is necessary for hashing.

The function reads the file in a loop, processing one chunk at a time. Each chunk is read using the `read` method, which retrieves up to `bytes_per_chunk` bytes from the file. The `update` method of the hash object is called with the chunk, adding its data to the ongoing hash computation. The loop continues until the end of the file is reached, at which point `read` returns an empty byte string, causing the loop to terminate.

Finally, the `hexdigest` method of the hash object is called to retrieve the computed hash as a hexadecimal string. This string is returned as the result of the function. The MD5 hash can then be used to verify the file's integrity or compare it with other files.

It is worth noting that while MD5 is fast and widely supported, it is not considered cryptographically secure for sensitive applications due to vulnerabilities to collision attacks. For stronger security, algorithms like SHA-256 (available in the `hashlib` module) are recommended.

The provided code defines the `main` function, which serves as the entry point for a command-line utility to compute and output the MD5 hashes of one or more files. The function uses the `argparse` module to parse command-line arguments, allowing users to specify the files to hash and an optional parent directory for relative path computation.

The `argparse.ArgumentParser` is configured to accept a positional argument, `file`, which can take one or more file paths (indicated by `nargs="+"`). Additionally, an optional argument, `--dir`, allows the user to specify a parent directory. This directory is used to compute relative paths for the files when generating the output.

After parsing the arguments, the code checks if the `--dir` argument was provided. If so, it converts the directory path into a `Path` object using Python's `pathlib` module, which provides convenient methods for file and directory manipulation.

The function then initializes an empty dictionary, `hashes`, to store the computed hashes. It iterates over each file path provided in the `file` argument, converting each path into a `Path` object. The `get_file_hash` function (defined elsewhere in the code) is called to compute the MD5 hash of the file. If the `--dir` argument was specified, the file path is converted to a relative path with respect to the parent directory using the `relative_to` method. This ensures that the output contains relative paths instead of absolute ones.

Finally, the `hashes` dictionary, which maps file paths (as strings) to their corresponding MD5 hashes, is serialized to JSON format using `json.dump`. The JSON output is written to `sys.stdout`, allowing the user to redirect or capture the output as needed. This design makes the utility versatile and suitable for integration into larger workflows or scripts.


## Source Code

```py
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Union


def get_file_hash(path: Union[str, Path], bytes_per_chunk: int = 8192) -> str:
    """Hash a file in chunks using md5."""
    path_hash = hashlib.md5()
    with open(path, "rb") as path_file:
        chunk = path_file.read(bytes_per_chunk)
        while chunk:
            path_hash.update(chunk)
            chunk = path_file.read(bytes_per_chunk)

    return path_hash.hexdigest()


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+")
    parser.add_argument("--dir", help="Parent directory")
    args = parser.parse_args()

    if args.dir:
        args.dir = Path(args.dir)

    hashes = {}
    for path_str in args.file:
        path = Path(path_str)
        path_hash = get_file_hash(path)
        if args.dir:
            path = path.relative_to(args.dir)

        hashes[str(path)] = path_hash

    json.dump(hashes, sys.stdout)


if __name__ == "__main__":
    main()
```