//! Kyutai streaming STT transcriber (moshi/candle-based)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::Activation;
use rubato::{FftFixedIn, Resampler};
use tracing::{debug, info, warn};

/// PCM chunk size required by the Kyutai/Mimi codec: 80 ms at 24 kHz
pub const KYUTAI_CHUNK_SAMPLES: usize = 1920;

/// Sample rate expected by the Kyutai/Mimi codec
const KYUTAI_SAMPLE_RATE: u32 = 24_000;

/// Temperature for text token sampling (0 = greedy)
const ASR_TEMPERATURE: f64 = 0.0;

/// Serde-deserialisable subset of the model's config.json.
///
/// LM architecture fields (card, n_q, dim, etc.) are used to construct the
/// `moshi::lm::Config` dynamically, so we don't rely on hardcoded presets
/// that may not match the model weights.
#[derive(serde::Deserialize)]
struct ModelConfig {
    mimi_name: String,
    tokenizer_name: String,
    #[serde(default)]
    stt_config: SttConfig,

    // LM architecture
    card: usize,
    n_q: usize,
    text_card: usize,
    dim: usize,
    num_heads: usize,
    num_layers: usize,
    hidden_scale: f64,
    context: usize,
    max_period: f64,
    layer_scale: Option<f64>,
    #[serde(default)]
    extra_heads_num_heads: usize,
    #[serde(default)]
    extra_heads_dim: usize,
}

impl ModelConfig {
    fn build_lm_config(&self) -> moshi::lm::Config {
        let dim_feedforward = (self.dim as f64 * self.hidden_scale).round() as usize;

        let transformer = moshi::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: self.layer_scale,
            positional_embedding: moshi::transformer::PositionalEmbedding::Rope,
            use_conv_block: false,
            cross_attention: None,
            conv_kernel_size: 3,
            use_conv_bias: true,
            gating: Some(Activation::Silu),
            norm: moshi::NormType::RmsNorm,
            context: self.context,
            max_period: self.max_period as usize,
            max_seq_len: 4096,
            kv_repeat: 1,
            dim_feedforward,
            conv_layout: false,
            shared_cross_attn: false,
        };

        let extra_heads = if self.extra_heads_num_heads > 0 {
            Some(moshi::lm::ExtraHeadsConfig {
                num_heads: self.extra_heads_num_heads,
                dim: self.extra_heads_dim,
            })
        } else {
            None
        };

        moshi::lm::Config {
            transformer,
            depformer: None,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_vocab_size: self.card + 1,
            audio_codebooks: self.n_q,
            conditioners: Default::default(),
            extra_heads,
        }
    }
}

#[derive(serde::Deserialize, Default)]
struct SttConfig {
    #[serde(default)]
    audio_delay_seconds: f64,
}

/// Streaming Kyutai transcriber.
///
/// Call [`step`] with exactly [`KYUTAI_CHUNK_SAMPLES`] samples (80 ms @ 24 kHz)
/// per invocation; collect the returned word strings.
pub struct KyutaiTranscriber {
    state: moshi::asr::State,
    tokenizer: sentencepiece::SentencePieceProcessor,
    device: Device,
}

impl KyutaiTranscriber {
    /// Download (or use cached) model weights from HuggingFace and initialise
    /// the transcriber.  Automatically selects CUDA if available, otherwise CPU.
    pub fn new(model_repo: &str) -> Result<Self> {
        let device = Device::new_cuda(0).unwrap_or_else(|e| {
            warn!("CUDA unavailable ({}), falling back to CPU", e);
            Device::Cpu
        });
        Self::with_device(model_repo, device)
    }

    /// Like [`new`] but with an explicit compute device.
    pub fn with_device(model_repo: &str, device: Device) -> Result<Self> {
        info!("Loading Kyutai STT model from HF repo: {}", model_repo);

        let api = hf_hub::api::sync::Api::new()
            .context("Failed to create HuggingFace API client")?;
        let repo = api.model(model_repo.to_string());

        // Download config.json and parse it
        let config_path = repo.get("config.json")
            .context("Failed to download config.json")?;
        let config: ModelConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path).context("Failed to read config.json")?,
        )
        .context("Failed to parse config.json")?;

        debug!("mimi_name: {}", config.mimi_name);
        debug!("tokenizer_name: {}", config.tokenizer_name);

        // Download model weights
        let model_path = repo.get("model.safetensors")
            .context("Failed to download model.safetensors")?;

        // Download Mimi codec weights
        let mimi_path = repo.get(&config.mimi_name)
            .context("Failed to download Mimi codec weights")?;

        // Download SentencePiece tokenizer
        let tokenizer_path = repo.get(&config.tokenizer_name)
            .context("Failed to download tokenizer")?;

        info!("All model files downloaded/cached, loading into memory...");

        // Load Mimi audio tokenizer
        let audio_tokenizer = moshi::mimi::load(
            mimi_path.to_str().context("Non-UTF8 mimi path")?,
            Some(config.n_q),
            &device,
        )
        .context("Failed to load Mimi codec")?;

        // Build LM config from the model's own config.json
        let lm_cfg = config.build_lm_config();
        debug!(
            "LM config: d_model={}, heads={}, layers={}, ff={}, text_vocab={}, audio_codebooks={}",
            lm_cfg.transformer.d_model,
            lm_cfg.transformer.num_heads,
            lm_cfg.transformer.num_layers,
            lm_cfg.transformer.dim_feedforward,
            lm_cfg.text_in_vocab_size,
            lm_cfg.audio_codebooks,
        );

        // Load ASR language model with the config derived from config.json
        info!("Loading LM from {:?} with dtype BF16 on {:?}", model_path, device);
        let lm = moshi::lm::load_lm_model(lm_cfg, &model_path, DType::BF16, &device)
            .with_context(|| format!(
                "Failed to load LM model from {:?} (device={:?})",
                model_path, device
            ))?;

        // Compute ASR delay in tokens: 12.5 tokens/second from the Mimi codec
        let asr_delay_in_tokens =
            (config.stt_config.audio_delay_seconds * 12.5).round() as usize;

        let mut state =
            moshi::asr::State::new(1, asr_delay_in_tokens, ASR_TEMPERATURE, audio_tokenizer, lm)
                .context("Failed to create ASR state")?;

        // Load SentencePiece tokenizer
        let tokenizer =
            sentencepiece::SentencePieceProcessor::open(&tokenizer_path)
                .context("Failed to load SentencePiece tokenizer")?;

        // Warmup: feed a few zero chunks to prime GPU kernels
        info!("Warming up GPU kernels...");
        let zero_chunk = vec![0.0f32; KYUTAI_CHUNK_SAMPLES];
        for _ in 0..5 {
            let _ = Self::run_step(&mut state, &zero_chunk, &device);
        }
        info!("Kyutai STT model ready");

        Ok(Self {
            state,
            tokenizer,
            device,
        })
    }

    /// Feed exactly [`KYUTAI_CHUNK_SAMPLES`] PCM samples at 24 kHz.
    /// Returns any new words decoded in this step (empty if none yet).
    pub fn step(&mut self, pcm_24k: &[f32]) -> Result<Vec<String>> {
        debug_assert_eq!(
            pcm_24k.len(),
            KYUTAI_CHUNK_SAMPLES,
            "step() requires exactly {KYUTAI_CHUNK_SAMPLES} samples"
        );

        let msgs = Self::run_step(&mut self.state, pcm_24k, &self.device)?;
        let mut words = Vec::new();

        for msg in msgs {
            if let moshi::asr::AsrMsg::Word { tokens, .. } = msg {
                let text = self
                    .tokenizer
                    .decode_piece_ids(&tokens)
                    .unwrap_or_default();
                if !text.is_empty() {
                    debug!("Word: {:?}", text);
                    words.push(text);
                }
            }
        }

        Ok(words)
    }

    /// Transcribe a complete buffer of 24 kHz PCM samples in one call.
    ///
    /// Useful for batch/offline transcription (e.g. from a file).  The buffer is
    /// processed in [`KYUTAI_CHUNK_SAMPLES`]-sample chunks; any remaining samples
    /// shorter than a full chunk are zero-padded.
    pub fn transcribe_buffer(&mut self, pcm_24k: &[f32]) -> Result<String> {
        let mut words = Vec::new();

        let mut offset = 0;
        while offset + KYUTAI_CHUNK_SAMPLES <= pcm_24k.len() {
            let chunk = &pcm_24k[offset..offset + KYUTAI_CHUNK_SAMPLES];
            words.extend(self.step(chunk)?);
            offset += KYUTAI_CHUNK_SAMPLES;
        }

        // Pad and flush remaining samples
        if offset < pcm_24k.len() {
            let mut chunk = pcm_24k[offset..].to_vec();
            chunk.resize(KYUTAI_CHUNK_SAMPLES, 0.0);
            words.extend(self.step(&chunk)?);
        }

        Ok(words.join(" "))
    }

    fn run_step(
        state: &mut moshi::asr::State,
        pcm: &[f32],
        device: &Device,
    ) -> Result<Vec<moshi::asr::AsrMsg>> {
        let tensor = Tensor::new(pcm, device)
            .context("Failed to create PCM tensor")?
            .reshape((1, 1, ()))
            .context("Failed to reshape PCM tensor")?;

        state
            .step_pcm(tensor, None, &().into(), |_, _, _| ())
            .context("step_pcm failed")
    }
}

/// Load a WAV file and resample it to 24 kHz mono f32 suitable for
/// [`KyutaiTranscriber::transcribe_buffer`].
pub fn load_wav_24k(path: &std::path::Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV: {}", path.display()))?;
    let spec = reader.spec();

    // Decode to f32 mono
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
    };

    // Mix to mono if needed
    let mono: Vec<f32> = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    } else {
        samples
    };

    // Resample to 24 kHz if needed
    if spec.sample_rate == KYUTAI_SAMPLE_RATE {
        return Ok(mono);
    }

    let chunk_size = 1024usize;
    let mut resampler = FftFixedIn::<f64>::new(
        spec.sample_rate as usize,
        KYUTAI_SAMPLE_RATE as usize,
        chunk_size,
        2,
        1,
    )
    .context("Failed to create resampler")?;

    let mut output = Vec::new();
    for chunk in mono.chunks(chunk_size) {
        let mut padded = chunk.to_vec();
        padded.resize(chunk_size, 0.0);
        let input_f64: Vec<f64> = padded.iter().map(|&s| s as f64).collect();
        let out = resampler
            .process(&[input_f64], None)
            .context("Resampling failed")?;
        output.extend(out[0].iter().map(|&s| s as f32));
    }

    Ok(output)
}
