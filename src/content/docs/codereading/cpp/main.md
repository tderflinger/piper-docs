---
title: main.cpp
description: main.cpp
---

## Code Explained

The provided code snippet defines the core configuration and main execution logic for a text-to-speech (TTS) application. Here's a detailed explanation of the active selection:

---

### **1. `RunConfig` Structure**
The `RunConfig` structure encapsulates all the configuration options required to run the TTS application. It includes paths, output settings, and synthesis parameters:

- **Model and Configuration Paths**:
  - `modelPath`: Specifies the path to the `.onnx` voice model file, which contains the neural network for speech synthesis.
  - `modelConfigPath`: Points to the JSON configuration file for the model, which defines additional metadata like speaker mappings.

- **Output Settings**:
  - `outputType`: Determines the type of output to produce:
    - `OUTPUT_FILE`: Writes the synthesized audio to a specified file.
    - `OUTPUT_DIRECTORY`: Saves audio files in a directory, typically with auto-generated names.
    - `OUTPUT_STDOUT`: Streams the audio data to standard output.
    - `OUTPUT_RAW`: Outputs raw audio data directly to standard output.
  - `outputPath`: Specifies the file or directory path for the output, defaulting to the current directory.

- **Synthesis Parameters**:
  - `speakerId`: Allows selection of a specific speaker in multi-speaker models.
  - `noiseScale`: Controls the amount of noise added during audio generation, affecting voice variability.
  - `lengthScale`: Adjusts the speaking speed (e.g., values < 1 make speech faster).
  - `noiseW`: Introduces variation in phoneme lengths for naturalness.
  - `sentenceSilenceSeconds`: Adds silence after each sentence for better pacing.

- **Additional Options**:
  - `eSpeakDataPath`: Path to the `espeak-ng` data directory, used for phoneme generation.
  - `tashkeelModelPath`: Path to the `libtashkeel` model for Arabic text processing.
  - `jsonInput`: Indicates whether input is in JSON format, allowing structured input with fields like `text`, `speaker_id`, and `output_file`.
  - `phonemeSilenceSeconds`: Maps specific phonemes to additional silence durations.
  - `useCuda`: Enables CUDA for GPU acceleration during synthesis.

This structure centralizes all runtime configurations, making the application flexible and customizable.

---

### **2. `OutputType` Enumeration**
The `OutputType` enum defines the possible output modes for the synthesized audio:
- `OUTPUT_FILE`: Outputs to a single WAV file.
- `OUTPUT_DIRECTORY`: Saves multiple WAV files in a directory.
- `OUTPUT_STDOUT`: Streams the WAV data to standard output.
- `OUTPUT_RAW`: Outputs raw audio data directly, bypassing WAV formatting.

This enum simplifies the handling of different output scenarios in the main logic.

---

### **3. Main Function**
The `main` function orchestrates the TTS application's execution:

- **Initialization**:
  - Sets up logging using `spdlog`.
  - Parses command-line arguments into a `RunConfig` instance using `parseArgs`.

- **Voice Loading**:
  - Loads the voice model and configuration using `loadVoice`, measuring the time taken for this operation.
  - Determines paths for auxiliary resources like `espeak-ng` data and `libtashkeel` models, defaulting to locations near the executable if not explicitly provided.

- **Configuration Application**:
  - Applies synthesis parameters (e.g., `noiseScale`, `lengthScale`) from `RunConfig` to the `voice` object.
  - Handles phoneme-specific silence settings, either overwriting or merging with existing configurations.

- **Input Processing**:
  - Reads input lines (either plain text or JSON) and processes them based on the `outputType`:
    - **Directory Output**: Generates timestamped filenames for WAV files.
    - **File Output**: Writes all input to a single WAV file.
    - **Standard Output**: Streams WAV or raw audio data to `stdout`.

- **Synthesis Execution**:
  - Uses `piper::textToWavFile` or `piper::textToAudio` to synthesize audio for each input line.
  - Logs performance metrics like the real-time factor (ratio of synthesis time to audio duration).

- **Termination**:
  - Cleans up resources by calling `piper::terminate`.

---

### **4. Supporting Functions**
- **`parseArgs`**:
  - Parses command-line arguments to populate the `RunConfig` structure.
  - Validates the existence of required files (e.g., model and config files).
  - Supports various options like `--use-cuda`, `--json-input`, and `--output_raw`.

- **`rawOutputProc`**:
  - Handles raw audio output in a separate thread, ensuring smooth streaming to `stdout`.

- **`printUsage`**:
  - Displays usage instructions for the command-line interface.

---

### **Purpose and Use Case**
This code is part of a TTS system that synthesizes speech from text using a neural network model. It supports multiple output formats, customizable synthesis parameters, and auxiliary tools like `espeak-ng` for phoneme generation. The modular design and flexible configuration make it suitable for various applications, including real-time speech synthesis, batch processing, and integration into larger systems.

## Source Code

```cpp
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "json.hpp"
#include "piper.hpp"

using namespace std;
using json = nlohmann::json;

enum OutputType { OUTPUT_FILE, OUTPUT_DIRECTORY, OUTPUT_STDOUT, OUTPUT_RAW };

struct RunConfig {
  // Path to .onnx voice file
  filesystem::path modelPath;

  // Path to JSON voice config file
  filesystem::path modelConfigPath;

  // Type of output to produce.
  // Default is to write a WAV file in the current directory.
  OutputType outputType = OUTPUT_DIRECTORY;

  // Path for output
  optional<filesystem::path> outputPath = filesystem::path(".");

  // Numerical id of the default speaker (multi-speaker voices)
  optional<piper::SpeakerId> speakerId;

  // Amount of noise to add during audio generation
  optional<float> noiseScale;

  // Speed of speaking (1 = normal, < 1 is faster, > 1 is slower)
  optional<float> lengthScale;

  // Variation in phoneme lengths
  optional<float> noiseW;

  // Seconds of silence to add after each sentence
  optional<float> sentenceSilenceSeconds;

  // Path to espeak-ng data directory (default is next to piper executable)
  optional<filesystem::path> eSpeakDataPath;

  // Path to libtashkeel ort model
  // https://github.com/mush42/libtashkeel/
  optional<filesystem::path> tashkeelModelPath;

  // stdin input is lines of JSON instead of text with format:
  // {
  //   "text": str,               (required)
  //   "speaker_id": int,         (optional)
  //   "speaker": str,            (optional)
  //   "output_file": str,        (optional)
  // }
  bool jsonInput = false;

  // Seconds of extra silence to insert after a single phoneme
  optional<std::map<piper::Phoneme, float>> phonemeSilenceSeconds;

  // true to use CUDA execution provider
  bool useCuda = false;
};

void parseArgs(int argc, char *argv[], RunConfig &runConfig);
void rawOutputProc(vector<int16_t> &sharedAudioBuffer, mutex &mutAudio,
                   condition_variable &cvAudio, bool &audioReady,
                   bool &audioFinished);

// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  spdlog::set_default_logger(spdlog::stderr_color_st("piper"));

  RunConfig runConfig;
  parseArgs(argc, argv, runConfig);

#ifdef _WIN32
  // Required on Windows to show IPA symbols
  SetConsoleOutputCP(CP_UTF8);
#endif

  piper::PiperConfig piperConfig;
  piper::Voice voice;

  spdlog::debug("Loading voice from {} (config={})",
                runConfig.modelPath.string(),
                runConfig.modelConfigPath.string());

  auto startTime = chrono::steady_clock::now();
  loadVoice(piperConfig, runConfig.modelPath.string(),
            runConfig.modelConfigPath.string(), voice, runConfig.speakerId,
            runConfig.useCuda);
  auto endTime = chrono::steady_clock::now();
  spdlog::info("Loaded voice in {} second(s)",
               chrono::duration<double>(endTime - startTime).count());

  // Get the path to the piper executable so we can locate espeak-ng-data, etc.
  // next to it.
#ifdef _MSC_VER
  auto exePath = []() {
    wchar_t moduleFileName[MAX_PATH] = {0};
    GetModuleFileNameW(nullptr, moduleFileName, std::size(moduleFileName));
    return filesystem::path(moduleFileName);
  }();
#else
#ifdef __APPLE__
  auto exePath = []() {
    char moduleFileName[PATH_MAX] = {0};
    uint32_t moduleFileNameSize = std::size(moduleFileName);
    _NSGetExecutablePath(moduleFileName, &moduleFileNameSize);
    return filesystem::path(moduleFileName);
  }();
#else
  auto exePath = filesystem::canonical("/proc/self/exe");
#endif
#endif

  if (voice.phonemizeConfig.phonemeType == piper::eSpeakPhonemes) {
    spdlog::debug("Voice uses eSpeak phonemes ({})",
                  voice.phonemizeConfig.eSpeak.voice);

    if (runConfig.eSpeakDataPath) {
      // User provided path
      piperConfig.eSpeakDataPath = runConfig.eSpeakDataPath.value().string();
    } else {
      // Assume next to piper executable
      piperConfig.eSpeakDataPath =
          std::filesystem::absolute(
              exePath.parent_path().append("espeak-ng-data"))
              .string();

      spdlog::debug("espeak-ng-data directory is expected at {}",
                    piperConfig.eSpeakDataPath);
    }
  } else {
    // Not using eSpeak
    piperConfig.useESpeak = false;
  }

  // Enable libtashkeel for Arabic
  if (voice.phonemizeConfig.eSpeak.voice == "ar") {
    piperConfig.useTashkeel = true;
    if (runConfig.tashkeelModelPath) {
      // User provided path
      piperConfig.tashkeelModelPath =
          runConfig.tashkeelModelPath.value().string();
    } else {
      // Assume next to piper executable
      piperConfig.tashkeelModelPath =
          std::filesystem::absolute(
              exePath.parent_path().append("libtashkeel_model.ort"))
              .string();

      spdlog::debug("libtashkeel model is expected at {}",
                    piperConfig.tashkeelModelPath.value());
    }
  }

  piper::initialize(piperConfig);

  // Scales
  if (runConfig.noiseScale) {
    voice.synthesisConfig.noiseScale = runConfig.noiseScale.value();
  }

  if (runConfig.lengthScale) {
    voice.synthesisConfig.lengthScale = runConfig.lengthScale.value();
  }

  if (runConfig.noiseW) {
    voice.synthesisConfig.noiseW = runConfig.noiseW.value();
  }

  if (runConfig.sentenceSilenceSeconds) {
    voice.synthesisConfig.sentenceSilenceSeconds =
        runConfig.sentenceSilenceSeconds.value();
  }

  if (runConfig.phonemeSilenceSeconds) {
    if (!voice.synthesisConfig.phonemeSilenceSeconds) {
      // Overwrite
      voice.synthesisConfig.phonemeSilenceSeconds =
          runConfig.phonemeSilenceSeconds;
    } else {
      // Merge
      for (const auto &[phoneme, silenceSeconds] :
           *runConfig.phonemeSilenceSeconds) {
        voice.synthesisConfig.phonemeSilenceSeconds->try_emplace(
            phoneme, silenceSeconds);
      }
    }

  } // if phonemeSilenceSeconds

  if (runConfig.outputType == OUTPUT_DIRECTORY) {
    runConfig.outputPath = filesystem::absolute(runConfig.outputPath.value());
    spdlog::info("Output directory: {}", runConfig.outputPath.value().string());
  }

  string line;
  piper::SynthesisResult result;
  while (getline(cin, line)) {
    auto outputType = runConfig.outputType;
    auto speakerId = voice.synthesisConfig.speakerId;
    std::optional<filesystem::path> maybeOutputPath = runConfig.outputPath;

    if (runConfig.jsonInput) {
      // Each line is a JSON object
      json lineRoot = json::parse(line);

      // Text is required
      line = lineRoot["text"].get<std::string>();

      if (lineRoot.contains("output_file")) {
        // Override output WAV file path
        outputType = OUTPUT_FILE;
        maybeOutputPath =
            filesystem::path(lineRoot["output_file"].get<std::string>());
      }

      if (lineRoot.contains("speaker_id")) {
        // Override speaker id
        voice.synthesisConfig.speakerId =
            lineRoot["speaker_id"].get<piper::SpeakerId>();
      } else if (lineRoot.contains("speaker")) {
        // Resolve to id using speaker id map
        auto speakerName = lineRoot["speaker"].get<std::string>();
        if ((voice.modelConfig.speakerIdMap) &&
            (voice.modelConfig.speakerIdMap->count(speakerName) > 0)) {
          voice.synthesisConfig.speakerId =
              (*voice.modelConfig.speakerIdMap)[speakerName];
        } else {
          spdlog::warn("No speaker named: {}", speakerName);
        }
      }
    }

    // Timestamp is used for path to output WAV file
    const auto now = chrono::system_clock::now();
    const auto timestamp =
        chrono::duration_cast<chrono::nanoseconds>(now.time_since_epoch())
            .count();

    if (outputType == OUTPUT_DIRECTORY) {
      // Generate path using timestamp
      stringstream outputName;
      outputName << timestamp << ".wav";
      filesystem::path outputPath = runConfig.outputPath.value();
      outputPath.append(outputName.str());

      // Output audio to automatically-named WAV file in a directory
      ofstream audioFile(outputPath.string(), ios::binary);
      piper::textToWavFile(piperConfig, voice, line, audioFile, result);
      cout << outputPath.string() << endl;
    } else if (outputType == OUTPUT_FILE) {
      if (!maybeOutputPath || maybeOutputPath->empty()) {
        throw runtime_error("No output path provided");
      }

      filesystem::path outputPath = maybeOutputPath.value();

      if (!runConfig.jsonInput) {
        // Read all of standard input before synthesizing.
        // Otherwise, we would overwrite the output file for each line.
        stringstream text;
        text << line;
        while (getline(cin, line)) {
          text << " " << line;
        }

        line = text.str();
      }

      // Output audio to WAV file
      ofstream audioFile(outputPath.string(), ios::binary);
      piper::textToWavFile(piperConfig, voice, line, audioFile, result);
      cout << outputPath.string() << endl;
    } else if (outputType == OUTPUT_STDOUT) {
      // Output WAV to stdout
      piper::textToWavFile(piperConfig, voice, line, cout, result);
    } else if (outputType == OUTPUT_RAW) {
      // Raw output to stdout
      mutex mutAudio;
      condition_variable cvAudio;
      bool audioReady = false;
      bool audioFinished = false;
      vector<int16_t> audioBuffer;
      vector<int16_t> sharedAudioBuffer;

#ifdef _WIN32
      // Needed on Windows to avoid terminal conversions
      setmode(fileno(stdout), O_BINARY);
      setmode(fileno(stdin), O_BINARY);
#endif

      thread rawOutputThread(rawOutputProc, ref(sharedAudioBuffer),
                             ref(mutAudio), ref(cvAudio), ref(audioReady),
                             ref(audioFinished));
      auto audioCallback = [&audioBuffer, &sharedAudioBuffer, &mutAudio,
                            &cvAudio, &audioReady]() {
        // Signal thread that audio is ready
        {
          unique_lock lockAudio(mutAudio);
          copy(audioBuffer.begin(), audioBuffer.end(),
               back_inserter(sharedAudioBuffer));
          audioReady = true;
          cvAudio.notify_one();
        }
      };
      piper::textToAudio(piperConfig, voice, line, audioBuffer, result,
                         audioCallback);

      // Signal thread that there is no more audio
      {
        unique_lock lockAudio(mutAudio);
        audioReady = true;
        audioFinished = true;
        cvAudio.notify_one();
      }

      // Wait for audio output to finish
      spdlog::info("Waiting for audio to finish playing...");
      rawOutputThread.join();
    }

    spdlog::info("Real-time factor: {} (infer={} sec, audio={} sec)",
                 result.realTimeFactor, result.inferSeconds,
                 result.audioSeconds);

    // Restore config (--json-input)
    voice.synthesisConfig.speakerId = speakerId;

  } // for each line

  piper::terminate(piperConfig);

  return EXIT_SUCCESS;
}

// ----------------------------------------------------------------------------

void rawOutputProc(vector<int16_t> &sharedAudioBuffer, mutex &mutAudio,
                   condition_variable &cvAudio, bool &audioReady,
                   bool &audioFinished) {
  vector<int16_t> internalAudioBuffer;
  while (true) {
    {
      unique_lock lockAudio{mutAudio};
      cvAudio.wait(lockAudio, [&audioReady] { return audioReady; });

      if (sharedAudioBuffer.empty() && audioFinished) {
        break;
      }

      copy(sharedAudioBuffer.begin(), sharedAudioBuffer.end(),
           back_inserter(internalAudioBuffer));

      sharedAudioBuffer.clear();

      if (!audioFinished) {
        audioReady = false;
      }
    }

    cout.write((const char *)internalAudioBuffer.data(),
               sizeof(int16_t) * internalAudioBuffer.size());
    cout.flush();
    internalAudioBuffer.clear();
  }

} // rawOutputProc

// ----------------------------------------------------------------------------

void printUsage(char *argv[]) {
  cerr << endl;
  cerr << "usage: " << argv[0] << " [options]" << endl;
  cerr << endl;
  cerr << "options:" << endl;
  cerr << "   -h        --help              show this message and exit" << endl;
  cerr << "   -m  FILE  --model       FILE  path to onnx model file" << endl;
  cerr << "   -c  FILE  --config      FILE  path to model config file "
          "(default: model path + .json)"
       << endl;
  cerr << "   -f  FILE  --output_file FILE  path to output WAV file ('-' for "
          "stdout)"
       << endl;
  cerr << "   -d  DIR   --output_dir  DIR   path to output directory (default: "
          "cwd)"
       << endl;
  cerr << "   --output_raw                  output raw audio to stdout as it "
          "becomes available"
       << endl;
  cerr << "   -s  NUM   --speaker     NUM   id of speaker (default: 0)" << endl;
  cerr << "   --noise_scale           NUM   generator noise (default: 0.667)"
       << endl;
  cerr << "   --length_scale          NUM   phoneme length (default: 1.0)"
       << endl;
  cerr << "   --noise_w               NUM   phoneme width noise (default: 0.8)"
       << endl;
  cerr << "   --sentence_silence      NUM   seconds of silence after each "
          "sentence (default: 0.2)"
       << endl;
  cerr << "   --espeak_data           DIR   path to espeak-ng data directory"
       << endl;
  cerr << "   --tashkeel_model        FILE  path to libtashkeel onnx model "
          "(arabic)"
       << endl;
  cerr << "   --json-input                  stdin input is lines of JSON "
          "instead of plain text"
       << endl;
  cerr << "   --use-cuda                    use CUDA execution provider"
       << endl;
  cerr << "   --debug                       print DEBUG messages to the console"
       << endl;
  cerr << "   -q       --quiet              disable logging" << endl;
  cerr << endl;
}

void ensureArg(int argc, char *argv[], int argi) {
  if ((argi + 1) >= argc) {
    printUsage(argv);
    exit(0);
  }
}

// Parse command-line arguments
void parseArgs(int argc, char *argv[], RunConfig &runConfig) {
  optional<filesystem::path> modelConfigPath;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-m" || arg == "--model") {
      ensureArg(argc, argv, i);
      runConfig.modelPath = filesystem::path(argv[++i]);
    } else if (arg == "-c" || arg == "--config") {
      ensureArg(argc, argv, i);
      modelConfigPath = filesystem::path(argv[++i]);
    } else if (arg == "-f" || arg == "--output_file" ||
               arg == "--output-file") {
      ensureArg(argc, argv, i);
      std::string filePath = argv[++i];
      if (filePath == "-") {
        runConfig.outputType = OUTPUT_STDOUT;
        runConfig.outputPath = nullopt;
      } else {
        runConfig.outputType = OUTPUT_FILE;
        runConfig.outputPath = filesystem::path(filePath);
      }
    } else if (arg == "-d" || arg == "--output_dir" || arg == "output-dir") {
      ensureArg(argc, argv, i);
      runConfig.outputType = OUTPUT_DIRECTORY;
      runConfig.outputPath = filesystem::path(argv[++i]);
    } else if (arg == "--output_raw" || arg == "--output-raw") {
      runConfig.outputType = OUTPUT_RAW;
    } else if (arg == "-s" || arg == "--speaker") {
      ensureArg(argc, argv, i);
      runConfig.speakerId = (piper::SpeakerId)stol(argv[++i]);
    } else if (arg == "--noise_scale" || arg == "--noise-scale") {
      ensureArg(argc, argv, i);
      runConfig.noiseScale = stof(argv[++i]);
    } else if (arg == "--length_scale" || arg == "--length-scale") {
      ensureArg(argc, argv, i);
      runConfig.lengthScale = stof(argv[++i]);
    } else if (arg == "--noise_w" || arg == "--noise-w") {
      ensureArg(argc, argv, i);
      runConfig.noiseW = stof(argv[++i]);
    } else if (arg == "--sentence_silence" || arg == "--sentence-silence") {
      ensureArg(argc, argv, i);
      runConfig.sentenceSilenceSeconds = stof(argv[++i]);
    } else if (arg == "--phoneme_silence" || arg == "--phoneme-silence") {
      ensureArg(argc, argv, i);
      ensureArg(argc, argv, i + 1);
      auto phonemeStr = std::string(argv[++i]);
      if (!piper::isSingleCodepoint(phonemeStr)) {
        std::cerr << "Phoneme '" << phonemeStr
                  << "' is not a single codepoint (--phoneme_silence)"
                  << std::endl;
        exit(1);
      }

      if (!runConfig.phonemeSilenceSeconds) {
        runConfig.phonemeSilenceSeconds.emplace();
      }

      auto phoneme = piper::getCodepoint(phonemeStr);
      (*runConfig.phonemeSilenceSeconds)[phoneme] = stof(argv[++i]);
    } else if (arg == "--espeak_data" || arg == "--espeak-data") {
      ensureArg(argc, argv, i);
      runConfig.eSpeakDataPath = filesystem::path(argv[++i]);
    } else if (arg == "--tashkeel_model" || arg == "--tashkeel-model") {
      ensureArg(argc, argv, i);
      runConfig.tashkeelModelPath = filesystem::path(argv[++i]);
    } else if (arg == "--json_input" || arg == "--json-input") {
      runConfig.jsonInput = true;
    } else if (arg == "--use_cuda" || arg == "--use-cuda") {
      runConfig.useCuda = true;
    } else if (arg == "--version") {
      std::cout << piper::getVersion() << std::endl;
      exit(0);
    } else if (arg == "--debug") {
      // Set DEBUG logging
      spdlog::set_level(spdlog::level::debug);
    } else if (arg == "-q" || arg == "--quiet") {
      // diable logging
      spdlog::set_level(spdlog::level::off);
    } else if (arg == "-h" || arg == "--help") {
      printUsage(argv);
      exit(0);
    }
  }

  // Verify model file exists
  ifstream modelFile(runConfig.modelPath.c_str(), ios::binary);
  if (!modelFile.good()) {
    throw runtime_error("Model file doesn't exist");
  }

  if (!modelConfigPath) {
    runConfig.modelConfigPath =
        filesystem::path(runConfig.modelPath.string() + ".json");
  } else {
    runConfig.modelConfigPath = modelConfigPath.value();
  }

  // Verify model config exists
  ifstream modelConfigFile(runConfig.modelConfigPath.c_str());
  if (!modelConfigFile.good()) {
    throw runtime_error("Model config doesn't exist");
  }
}
```
