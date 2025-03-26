---
title: Voices
description: Voices
---

Our goal is to support Home Assistant and the [Year of Voice](https://www.home-assistant.io/blog/2022/12/20/year-of-voice/).

[Download voices](VOICES.md) for the supported languages:

* Arabic (ar_JO)
* Catalan (ca_ES)
* Czech (cs_CZ)
* Welsh (cy_GB)
* Danish (da_DK)
* German (de_DE)
* Greek (el_GR)
* English (en_GB, en_US)
* Spanish (es_ES, es_MX)
* Finnish (fi_FI)
* French (fr_FR)
* Hungarian (hu_HU)
* Icelandic (is_IS)
* Italian (it_IT)
* Georgian (ka_GE)
* Kazakh (kk_KZ)
* Luxembourgish (lb_LU)
* Nepali (ne_NP)
* Dutch (nl_BE, nl_NL)
* Norwegian (no_NO)
* Polish (pl_PL)
* Portuguese (pt_BR, pt_PT)
* Romanian (ro_RO)
* Russian (ru_RU)
* Serbian (sr_RS)
* Swedish (sv_SE)
* Swahili (sw_CD)
* Turkish (tr_TR)
* Ukrainian (uk_UA)
* Vietnamese (vi_VN)
* Chinese (zh_CN)

You will need two files per voice:

1. A `.onnx` model file, such as [`en_US-lessac-medium.onnx`](https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx)
2. A `.onnx.json` config file, such as [`en_US-lessac-medium.onnx.json`](https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json)

The `MODEL_CARD` file for each voice contains important licensing information. Piper is intended for text to speech research, and does not impose any additional restrictions on voice models. Some voices may have restrictive licenses, however, so please review them carefully!
