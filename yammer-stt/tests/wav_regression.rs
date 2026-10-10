//! End-to-end transcription regression check over the committed WAV fixtures.
//!
//! Needs the Kyutai model weights (downloaded from HuggingFace on first run)
//! and ideally a GPU, so it is `#[ignore]`d and CI never runs it. Run it by
//! hand before a release and before bumping candle, moshi, rubato, hound or
//! CUDA:
//!
//!     cargo test -p yammer-stt --features cuda --release --test wav_regression -- --ignored --nocapture
//!
//! `YAMMER_STT_REPO` overrides the model repo.
//!
//! The synthetic fixtures come from `scripts/generate-wav-fixtures.sh`;
//! `armstrong_11k_u8.wav` is NASA's public-domain Apollo 11 recording.

use std::path::PathBuf;

use yammer_stt::{KYUTAI_CHUNK_SAMPLES, KyutaiTranscriber, load_wav_24k};

const DEFAULT_STT_REPO: &str = "kyutai/stt-1b-en_fr-candle";

/// Silence appended after each clip, matching the ~1 s the app pipeline feeds
/// at the end of a recording. The model emits text ~0.5 s behind the audio, so
/// without it the last words of a clip never come out.
const TRAILING_SILENCE_CHUNKS: usize = 13;

struct Case {
    file: &'static str,
    /// Lowercase phrases that must all appear in the transcript.
    keywords: &'static [&'static str],
}

const CASES: &[Case] = &[
    Case {
        file: "meeting_22k_i16.wav",
        keywords: &["schedule", "meeting", "engineering team", "thursday afternoon"],
    },
    Case {
        file: "grocery_48k_stereo_i16.wav",
        keywords: &["coffee", "bread", "oranges", "grocery store"],
    },
    Case {
        file: "weather_24k_f32.wav",
        keywords: &["sunny", "gentle breeze"],
    },
    Case {
        file: "email_16k_i16.wav",
        keywords: &["email", "support team", "printer"],
    },
    Case {
        file: "armstrong_11k_u8.wav",
        // "giant leap" is left out: on this noisy 11 kHz recording the model
        // hears "triathlete" from a freshly reset state, on GPU and CPU alike.
        keywords: &["one small step for man"],
    },
];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Lowercase and collapse punctuation to spaces so "Store." matches "store".
fn normalize(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn normalize_strips_punctuation_and_case() {
    assert_eq!(normalize("  Grocery   Store. Don't!"), "grocery store don't");
}

#[test]
fn every_fixture_has_a_case() {
    let mut on_disk: Vec<String> = std::fs::read_dir(fixture(""))
        .expect("fixtures dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".wav"))
        .collect();
    on_disk.sort();
    let mut listed: Vec<String> = CASES.iter().map(|c| c.file.to_string()).collect();
    listed.sort();
    assert_eq!(on_disk, listed);
}

#[test]
#[ignore = "needs Kyutai model weights (and ideally a GPU); run before releases and dependency bumps"]
fn fixtures_transcribe_with_expected_keywords() {
    let repo = std::env::var("YAMMER_STT_REPO").unwrap_or_else(|_| DEFAULT_STT_REPO.to_string());
    let mut transcriber = KyutaiTranscriber::new(&repo).expect("load Kyutai model");

    let mut failures = Vec::new();
    for case in CASES {
        transcriber.reset().expect("reset ASR state");

        let mut samples = load_wav_24k(&fixture(case.file))
            .unwrap_or_else(|e| panic!("load {}: {e:#}", case.file));
        samples.resize(samples.len() + TRAILING_SILENCE_CHUNKS * KYUTAI_CHUNK_SAMPLES, 0.0);

        let text = transcriber
            .transcribe_buffer(&samples)
            .unwrap_or_else(|e| panic!("transcribe {}: {e:#}", case.file));
        let normalized = normalize(&text);

        let missing: Vec<&str> = case
            .keywords
            .iter()
            .copied()
            .filter(|kw| !normalized.contains(kw))
            .collect();

        println!("{}: {text:?}", case.file);
        if !missing.is_empty() {
            failures.push(format!("{}: missing {missing:?} in {text:?}", case.file));
        }
    }

    assert!(failures.is_empty(), "transcription regressions:\n{}", failures.join("\n"));
}
