---
title: http_server.py
description: http_server.py
---

## Code Explained

The provided code defines a Python script that sets up an HTTP server for a text-to-speech (TTS) application. The server uses the Flask framework to handle HTTP requests and synthesizes speech using an ONNX-based TTS model. The script is designed to be run as a standalone application, with various configuration options provided via command-line arguments.

The main function begins by defining an argument parser using the argparse module. This parser allows users to specify various parameters, such as the host and port for the HTTP server, the path to the ONNX model and its configuration file, and optional parameters like speaker ID, phoneme length, noise levels, and silence duration between sentences. Additional arguments include options for enabling GPU acceleration (--cuda), specifying directories for data and downloads, and enabling debug logging. These arguments provide flexibility for configuring the TTS system based on user needs.

The script ensures that the required ONNX model file exists. If the model file is missing, it attempts to download the necessary voice data using helper functions like get_voices, ensure_voice_exists, and find_voice. These functions handle tasks such as resolving aliases for backward compatibility, updating voice metadata, and locating the appropriate model and configuration files. This ensures that the TTS system is ready to synthesize speech even if the required files are not initially present.

Once the model and configuration are loaded, the script initializes a PiperVoice instance using the PiperVoice.load method. This method sets up the ONNX inference session and loads the model configuration. The script also prepares a dictionary of synthesis parameters (synthesize_args), which includes options like speaker ID, phoneme length, noise levels, and sentence silence. These parameters are passed to the PiperVoice.synthesize method during speech synthesis.

The Flask application is then created to serve as the HTTP server. It defines a single route (/) that supports both GET and POST requests. When a request is received, the server extracts the input text from the query parameters (for GET) or the request body (for POST). The text is stripped of whitespace and validated to ensure it is not empty. The server then uses the PiperVoice.synthesize method to generate speech from the input text, writing the synthesized audio to a WAV file in memory. The resulting audio data is returned as the HTTP response.

Finally, the Flask server is started using the host and port specified in the command-line arguments. This allows users to interact with the TTS system by sending HTTP requests to the server, making it a convenient and flexible interface for generating speech from text. The script is designed to handle various configurations and edge cases, ensuring robustness and usability in different environments.

## Source Code

```py
#!/usr/bin/env python3
import argparse
import io
import logging
import wave
from pathlib import Path
from typing import Any, Dict

from flask import Flask, request

from . import PiperVoice
from .download import ensure_voice_exists, find_voice, get_voices

_LOGGER = logging.getLogger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port")
    #
    parser.add_argument("-m", "--model", required=True, help="Path to Onnx model file")
    parser.add_argument("-c", "--config", help="Path to model config file")
    #
    parser.add_argument("-s", "--speaker", type=int, help="Id of speaker (default: 0)")
    parser.add_argument(
        "--length-scale", "--length_scale", type=float, help="Phoneme length"
    )
    parser.add_argument(
        "--noise-scale", "--noise_scale", type=float, help="Generator noise"
    )
    parser.add_argument(
        "--noise-w", "--noise_w", type=float, help="Phoneme width noise"
    )
    #
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    #
    parser.add_argument(
        "--sentence-silence",
        "--sentence_silence",
        type=float,
        default=0.0,
        help="Seconds of silence after each sentence",
    )
    #
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        action="append",
        default=[str(Path.cwd())],
        help="Data directory to check for downloaded models (default: current directory)",
    )
    parser.add_argument(
        "--download-dir",
        "--download_dir",
        help="Directory to download voices into (default: first data dir)",
    )
    #
    parser.add_argument(
        "--update-voices",
        action="store_true",
        help="Download latest voices.json during startup",
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    if not args.download_dir:
        # Download to first data directory by default
        args.download_dir = args.data_dir[0]

    # Download voice if file doesn't exist
    model_path = Path(args.model)
    if not model_path.exists():
        # Load voice info
        voices_info = get_voices(args.download_dir, update_voices=args.update_voices)

        # Resolve aliases for backwards compatibility with old voice names
        aliases_info: Dict[str, Any] = {}
        for voice_info in voices_info.values():
            for voice_alias in voice_info.get("aliases", []):
                aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

        voices_info.update(aliases_info)
        ensure_voice_exists(args.model, args.data_dir, args.download_dir, voices_info)
        args.model, args.config = find_voice(args.model, args.data_dir)

    # Load voice
    voice = PiperVoice.load(args.model, config_path=args.config, use_cuda=args.cuda)
    synthesize_args = {
        "speaker_id": args.speaker,
        "length_scale": args.length_scale,
        "noise_scale": args.noise_scale,
        "noise_w": args.noise_w,
        "sentence_silence": args.sentence_silence,
    }

    # Create web server
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def app_synthesize() -> bytes:
        if request.method == "POST":
            text = request.data.decode("utf-8")
        else:
            text = request.args.get("text", "")

        text = text.strip()
        if not text:
            raise ValueError("No text provided")

        _LOGGER.debug("Synthesizing text: %s", text)
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                voice.synthesize(text, wav_file, **synthesize_args)

            return wav_io.getvalue()

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```