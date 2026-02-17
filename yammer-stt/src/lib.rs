//! yammer-stt: streaming speech-to-text via Kyutai STT (moshi/candle)
//!
//! Uses the kyutai/stt-1b-en_fr-candle model, downloaded automatically from
//! HuggingFace on first use and cached in ~/.cache/huggingface/.

pub mod transcriber;

pub use transcriber::{KyutaiTranscriber, KYUTAI_CHUNK_SAMPLES, load_wav_24k};
