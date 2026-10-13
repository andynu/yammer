# Third-Party Notices

Yammer includes or depends on the following third-party software. We are grateful
to their authors and contributors.

## Core Dependencies

### moshi
- **License**: MIT OR Apache-2.0
- **Copyright**: Copyright (c) The moshi authors (Kyutai)
- **Repository**: https://github.com/kyutai-labs/moshi

Rust implementation of Kyutai's streaming speech models, used for speech-to-text.

### candle
- **License**: MIT OR Apache-2.0
- **Copyright**: Copyright (c) The candle contributors
- **Repository**: https://github.com/huggingface/candle

Machine learning framework that runs the speech-to-text model, on CUDA when available.

### llama.cpp
- **License**: MIT
- **Copyright**: Copyright (c) 2023-2024 The ggml authors
- **Repository**: https://github.com/ggerganov/llama.cpp

C++ implementation of LLaMA and other large language models.

### Tauri
- **License**: Apache-2.0 OR MIT
- **Copyright**: Copyright (c) 2019-2024 Tauri Contributors
- **Repository**: https://github.com/tauri-apps/tauri

Framework for building lightweight desktop applications.

## Model Weights

Model weights are not bundled with Yammer. They are downloaded on first use.

### Kyutai STT (`kyutai/stt-1b-en_fr-candle`)
- **License**: CC-BY-4.0
- **Author**: Kyutai
- **Model card**: https://huggingface.co/kyutai/stt-1b-en_fr-candle

Streaming English and French speech-to-text model, fetched from Hugging Face.

## Rust Crates (Apache-2.0)

The following dependencies are licensed under Apache-2.0, which requires
attribution. When Apache-2.0 is offered alongside MIT, this project uses
the Apache-2.0 terms for these notices:

- **cpal** - Cross-platform audio library
  - Copyright (c) The cpal contributors
  - https://github.com/RustAudio/cpal

- **openssl** - OpenSSL bindings for Rust
  - Copyright (c) The openssl-rs authors
  - https://github.com/sfackler/rust-openssl

- **tao** - Cross-platform window management
  - Copyright (c) 2019-2024 Tauri Contributors
  - https://github.com/tauri-apps/tao

- **ring** - Safe, fast cryptography (Apache-2.0 AND ISC)
  - Copyright (c) 2015-2024 Brian Smith and contributors
  - https://github.com/briansmith/ring

## Mozilla Public License 2.0 (MPL-2.0) Dependencies

The following dependencies are licensed under MPL-2.0. Per MPL-2.0 Section 3.3,
these components are used unmodified. If you modify files from these crates,
your modifications to those specific files must be made available under MPL-2.0.

- **cssparser** - CSS parsing library
  - Copyright (c) The Servo Project Developers
  - https://github.com/servo/rust-cssparser

- **selectors** - CSS selector matching
  - Copyright (c) The Servo Project Developers
  - https://github.com/servo/servo/tree/main/components/selectors

- **dtoa-short** - Floating point formatting
  - Copyright (c) The dtoa-short authors

- **option-ext** - Option extension traits
  - Copyright (c) The option-ext authors

## GNU Lesser General Public License (LGPL-2.1-or-later OR GPL-3.0-or-later) Dependencies

- **librush** - zbus-based IBus engine/factory protocol implementation (connection
  setup, `org.freedesktop.IBus.Engine`/`Factory` D-Bus interfaces, GVariant
  serialization for `IBusText`/`IBusAttrList`), used by Yammer's IBus input-method
  engine component
  - Copyright (c) secext2022
  - https://github.com/fm-elpac/librush

Used unmodified, under LGPL-2.1-or-later. Yammer links against it dynamically
only — it is built into its own shared library, never statically compiled into
the main `yammer-app`/`yammer-cli` binaries — so Yammer's own MIT-licensed code
is not a combined work with it. See `CLAUDE.local.md` for the build-level
constraint this requires.

## Unicode License (Unicode-3.0) Dependencies

The following ICU and Unicode-related crates are licensed under Unicode-3.0:

- icu_collections, icu_locale_core, icu_normalizer, icu_properties, icu_provider
- litemap, potential_utf, tinystr, writeable, yoke, zerofrom, zerotrie, zerovec

Copyright (c) Unicode, Inc. See https://www.unicode.org/license.txt

## BSD-3-Clause Dependencies

- **alloc-no-stdlib, alloc-stdlib** - Allocator traits
- **bindgen** - Automatic Rust FFI bindings generation
- **encoding_rs** - Character encoding (includes BSD-3-Clause components)
- **subtle** - Constant-time operations for cryptography

## Other Notable Dependencies

### Rust Ecosystem (MIT OR Apache-2.0)
- **tokio** - Async runtime
- **serde** - Serialization framework
- **tracing** - Application-level tracing
- **reqwest** - HTTP client
- **clap** - Command-line argument parser

### NPM Packages
- **vite** (MIT) - Frontend build tool
- **rollup** (MIT) - JavaScript bundler
- **esbuild** (MIT) - JavaScript/TypeScript bundler
- **postcss** (MIT) - CSS processing
- **source-map-js** (BSD-3-Clause) - Source map library

---

For a complete list of dependencies and their licenses, run:
```bash
cargo license        # Rust dependencies
npx license-checker  # NPM dependencies
```
