// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  site: "https://tderflinger.github.io/piper-docs",
  output: "static",
  base: "/piper-docs",
  integrations: [
    starlight({
      title: "Piper Documentation",
      social: {
        github: "https://github.com/withastro/starlight",
      },
      sidebar: [
        {
          label: "About",
          items: [
            { label: "Introduction", slug: "about/intro" },
            {
              label: "Voices",
              items: [
                { label: "Available Voices", slug: "about/voices" },
                { label: "Download Voices", slug: "about/voices/download" },
              ],
            },
            { label: "Dependencies", slug: "about/dependencies" },
            { label: "People Using Piper", slug: "about/project_use" },
          ],
        },
        {
          label: "Guides",
          items: [
            { label: "Installation Guide", slug: "guides/installation" },
            { label: "Usage Guide", slug: "guides/usage" },
            { label: "Training Guide", slug: "guides/training" },
          ],
        },
        {
          label: "Code Reading",
          items: [
            { label: "Introduction", slug: "codereading/intro" },
            {
              label: "cpp",
              items: [
                { label: "main.cpp", slug: "codereading/cpp/main" },
                { label: "piper.cpp", slug: "codereading/cpp/piper" },
              ],
            },

            {
              label: "python_run",
              items: [
                {
                  label: "__main__.py",
                  slug: "codereading/python_run/piper/main",
                },
                {
                  label: "config.py",
                  slug: "codereading/python_run/piper/config",
                },
                {
                  label: "download.py",
                  slug: "codereading/python_run/piper/download",
                },
                {
                  label: "file_hash.py",
                  slug: "codereading/python_run/piper/file_hash",
                },
                {
                  label: "http_server.py",
                  slug: "codereading/python_run/piper/http_server",
                },
                { label: "util.py", slug: "codereading/python_run/piper/util" },
                {
                  label: "voice.py",
                  slug: "codereading/python_run/piper/voice",
                },
              ],
            },
            {
              label: "piper_train",
              items: [
                {
                  label: "__main__.py",
                  slug: "codereading/python/piper_train/main",
                },
                {
                  label: "check_phonemes.py",
                  slug: "codereading/python/piper_train/check_phonemes",
                },
                {
                  label: "clean_cached_audio.py",
                  slug: "codereading/python/piper_train/clean_cached_audio",
                },
                {
                  label: "export_generator.py",
                  slug: "codereading/python/piper_train/export_generator",
                },
                {
                  label: "export_onnx_streaming.py",
                  slug: "codereading/python/piper_train/export_onnx_streaming",
                },
                {
                  label: "export_onnx.py",
                  slug: "codereading/python/piper_train/export_onnx",
                },
                {
                  label: "export_torchscript.py",
                  slug: "codereading/python/piper_train/export_torchscript",
                },
                {
                  label: "filter_utterances.py",
                  slug: "codereading/python/piper_train/filter_utterances",
                },
                {
                  label: "infer_generator.py",
                  slug: "codereading/python/piper_train/infer_generator",
                },
                {
                  label: "infer_onnx_streaming.py",
                  slug: "codereading/python/piper_train/infer_onnx_streaming",
                },
                {
                  label: "infer_onnx.py",
                  slug: "codereading/python/piper_train/infer_onnx",
                },
                {
                  label: "infer_torchscript.py",
                  slug: "codereading/python/piper_train/infer_torchscript",
                },
                {
                  label: "infer.py",
                  slug: "codereading/python/piper_train/infer",
                },
                {
                  label: "preprocess.py",
                  slug: "codereading/python/piper_train/preprocess",
                },
                {
                  label: "select_speaker.py",
                  slug: "codereading/python/piper_train/select_speaker",
                },
                {
                  label: "vits",
                  items: [
                    {
                      label: "attentions.py",
                      slug: "codereading/python/piper_train/vits/attentions",
                    },
                    {
                      label: "commmons.py",
                      slug: "codereading/python/piper_train/vits/commons",
                    },
                    {
                      label: "config.py",
                      slug: "codereading/python/piper_train/vits/config",
                    },
                    {
                      label: "dataset.py",
                      slug: "codereading/python/piper_train/vits/dataset",
                    },
                    {
                      label: "lightning.py",
                      slug: "codereading/python/piper_train/vits/lightning",
                    },
                    {
                      label: "losses.py",
                      slug: "codereading/python/piper_train/vits/losses",
                    },
                    {
                      label: "mel_processing.py",
                      slug: "codereading/python/piper_train/vits/mel_processing",
                    },
                    {
                      label: "models.py",
                      slug: "codereading/python/piper_train/vits/models",
                    },
                    {
                      label: "modules.py",
                      slug: "codereading/python/piper_train/vits/modules",
                    },
                    {
                      label: "transforms.py",
                      slug: "codereading/python/piper_train/vits/transforms",
                    },
                    {
                      label: "utils.py",
                      slug: "codereading/python/piper_train/vits/utils",
                    },
                  ],
                },
              ],
            },
          ],
        },
      ],
    }),
  ],
});
