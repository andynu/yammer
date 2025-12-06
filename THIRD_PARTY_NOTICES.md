# Third-Party Notices

Yammer includes or depends on the following third-party software. We are grateful
to their authors and contributors.

## Core Dependencies

### whisper.cpp
- **License**: MIT
- **Copyright**: Copyright (c) 2023-2024 The ggml authors
- **Repository**: https://github.com/ggerganov/whisper.cpp

High-performance C++ implementation of OpenAI's Whisper speech recognition model.

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
