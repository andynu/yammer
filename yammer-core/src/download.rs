//! Model download manager

use crate::error::{Error, Result};
use crate::model::{ModelInfo, ModelStatus, ModelType};
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

/// Progress callback for downloads
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

/// Model download manager
pub struct DownloadManager {
    model_dir: PathBuf,
    client: reqwest::Client,
}

impl DownloadManager {
    /// Create a new download manager
    pub fn new(model_dir: PathBuf) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("yammer/0.1")
            .build()
            .expect("Failed to create HTTP client");

        Self { model_dir, client }
    }

    /// Get the default model directory
    pub fn default_model_dir() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("yammer")
            .join("models")
    }

    /// Ensure the model directory exists
    pub async fn ensure_model_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.model_dir).await?;
        Ok(())
    }

    /// Get path where a model would be stored
    pub fn model_path(&self, model: &ModelInfo) -> PathBuf {
        self.model_dir.join(&model.filename)
    }

    /// Check the status of a model
    pub async fn check_status(&self, model: &ModelInfo) -> ModelStatus {
        let path = self.model_path(model);
        if path.exists() {
            // Check file size matches expected
            match fs::metadata(&path).await {
                Ok(meta) if meta.len() == model.size_bytes => {
                    ModelStatus::Ready { path }
                }
                Ok(meta) => {
                    debug!(
                        "Model {} size mismatch: expected {}, got {}",
                        model.id, model.size_bytes, meta.len()
                    );
                    ModelStatus::Failed {
                        error: format!(
                            "Size mismatch: expected {}, got {}",
                            model.size_bytes,
                            meta.len()
                        ),
                    }
                }
                Err(e) => ModelStatus::Failed {
                    error: e.to_string(),
                },
            }
        } else {
            ModelStatus::NotDownloaded
        }
    }

    /// Download a model with optional progress callback
    pub async fn download(
        &self,
        model: &ModelInfo,
        progress: Option<ProgressCallback>,
    ) -> Result<PathBuf> {
        self.ensure_model_dir().await?;

        let dest_path = self.model_path(model);
        let temp_path = dest_path.with_extension("download");

        info!("Downloading {} from {}", model.id, model.url);

        // Start the download
        let response = self
            .client
            .get(&model.url)
            .send()
            .await
            .map_err(|e| Error::Model(format!("Failed to start download: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::Model(format!(
                "Download failed with status: {}",
                response.status()
            )));
        }

        let total_size = response.content_length().unwrap_or(model.size_bytes);

        // Create temp file
        let mut file = fs::File::create(&temp_path).await?;
        let mut downloaded: u64 = 0;
        let mut hasher = Sha256::new();

        // Stream the download
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| Error::Model(format!("Download error: {}", e)))?;

            file.write_all(&chunk).await?;
            hasher.update(&chunk);

            downloaded += chunk.len() as u64;
            if let Some(ref cb) = progress {
                cb(downloaded, total_size);
            }
        }

        file.flush().await?;
        drop(file);

        // Verify checksum if provided
        if let Some(expected_sha) = &model.sha256 {
            let actual_sha = format!("{:x}", hasher.finalize());
            if !actual_sha.starts_with(expected_sha) && !expected_sha.starts_with(&actual_sha) {
                fs::remove_file(&temp_path).await?;
                return Err(Error::Model(format!(
                    "Checksum mismatch: expected {}, got {}",
                    expected_sha, actual_sha
                )));
            }
            debug!("Checksum verified for {}", model.id);
        } else {
            let actual_sha = format!("{:x}", hasher.finalize());
            info!("Downloaded {} with SHA256: {}", model.id, actual_sha);
        }

        // Move temp file to final location
        fs::rename(&temp_path, &dest_path).await?;

        info!("Successfully downloaded {} to {:?}", model.id, dest_path);
        Ok(dest_path)
    }

    /// Download a model only if not already present
    pub async fn ensure_model(
        &self,
        model: &ModelInfo,
        progress: Option<ProgressCallback>,
    ) -> Result<PathBuf> {
        match self.check_status(model).await {
            ModelStatus::Ready { path } => {
                info!("Model {} already downloaded at {:?}", model.id, path);
                Ok(path)
            }
            ModelStatus::NotDownloaded => self.download(model, progress).await,
            ModelStatus::Failed { error } => {
                warn!("Model {} in failed state: {}. Re-downloading.", model.id, error);
                // Remove potentially corrupt file
                let path = self.model_path(model);
                if path.exists() {
                    fs::remove_file(&path).await?;
                }
                self.download(model, progress).await
            }
            ModelStatus::Downloading { .. } => {
                Err(Error::Model("Download already in progress".to_string()))
            }
        }
    }

    /// List all downloaded models
    pub async fn list_downloaded(&self) -> Result<Vec<PathBuf>> {
        let mut models = Vec::new();
        if !self.model_dir.exists() {
            return Ok(models);
        }

        let mut entries = fs::read_dir(&self.model_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                let ext = path.extension().and_then(|e| e.to_str());
                if matches!(ext, Some("bin") | Some("gguf") | Some("ggml")) {
                    models.push(path);
                }
            }
        }
        Ok(models)
    }

    /// Find a downloaded model by type
    pub async fn find_model_by_type(
        &self,
        model_type: ModelType,
        registry: &[ModelInfo],
    ) -> Option<PathBuf> {
        for model in registry.iter().filter(|m| m.model_type == model_type) {
            if let ModelStatus::Ready { path } = self.check_status(model).await {
                return Some(path);
            }
        }
        None
    }
}

/// Format bytes as human-readable size
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
    }
}
