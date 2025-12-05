//! Model download manager

use crate::error::{Error, Result};
use crate::model::{ModelInfo, ModelStatus, ModelType};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

/// Stores verified SHA256 hashes for downloaded models.
/// On first download, the hash is computed and stored.
/// On subsequent downloads, the hash is verified against the stored value.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerifiedHashes {
    /// Map of model ID to SHA256 hash (lowercase hex)
    pub hashes: HashMap<String, String>,
}

impl VerifiedHashes {
    /// Get the default path for the verified hashes file
    pub fn default_path() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("yammer")
            .join("verified_hashes.json")
    }

    /// Load verified hashes from the default path
    pub fn load() -> Self {
        Self::load_from(&Self::default_path())
    }

    /// Load verified hashes from a specific path
    pub fn load_from(path: &PathBuf) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(hashes) => {
                    debug!("Loaded verified hashes from {:?}", path);
                    hashes
                }
                Err(e) => {
                    warn!("Failed to parse verified hashes file {:?}: {}", path, e);
                    Self::default()
                }
            },
            Err(_) => {
                debug!("No verified hashes file at {:?}", path);
                Self::default()
            }
        }
    }

    /// Save verified hashes to the default path
    pub fn save(&self) -> Result<()> {
        self.save_to(&Self::default_path())
    }

    /// Save verified hashes to a specific path
    pub fn save_to(&self, path: &PathBuf) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("Failed to serialize hashes: {}", e)))?;
        std::fs::write(path, contents)?;
        debug!("Saved verified hashes to {:?}", path);
        Ok(())
    }

    /// Get the verified hash for a model
    pub fn get(&self, model_id: &str) -> Option<&String> {
        self.hashes.get(model_id)
    }

    /// Store a verified hash for a model
    pub fn set(&mut self, model_id: String, sha256: String) {
        self.hashes.insert(model_id, sha256);
    }

    /// Remove a verified hash for a model
    pub fn remove(&mut self, model_id: &str) -> Option<String> {
        self.hashes.remove(model_id)
    }

    /// Clear all verified hashes
    pub fn clear(&mut self) {
        self.hashes.clear();
    }
}

/// Progress callback for downloads
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

/// Model download manager
pub struct DownloadManager {
    model_dir: PathBuf,
    client: reqwest::Client,
    verified_hashes: VerifiedHashes,
}

impl DownloadManager {
    /// Create a new download manager
    pub fn new(model_dir: PathBuf) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("yammer/0.1")
            .build()
            .expect("Failed to create HTTP client");

        let verified_hashes = VerifiedHashes::load();

        Self {
            model_dir,
            client,
            verified_hashes,
        }
    }

    /// Get a reference to the verified hashes
    pub fn verified_hashes(&self) -> &VerifiedHashes {
        &self.verified_hashes
    }

    /// Get a mutable reference to the verified hashes
    pub fn verified_hashes_mut(&mut self) -> &mut VerifiedHashes {
        &mut self.verified_hashes
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
        &mut self,
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

        let actual_sha = format!("{:x}", hasher.finalize());

        // Verify checksum: check registry hash first, then verified hashes file
        let expected_sha = model
            .sha256
            .as_ref()
            .or_else(|| self.verified_hashes.get(&model.id));

        if let Some(expected) = expected_sha {
            if !actual_sha.starts_with(expected) && !expected.starts_with(&actual_sha) {
                fs::remove_file(&temp_path).await?;
                return Err(Error::Model(format!(
                    "Checksum mismatch: expected {}, got {}",
                    expected, actual_sha
                )));
            }
            info!(
                "Checksum verified for {} (SHA256: {}...)",
                model.id,
                &actual_sha[..16]
            );
        } else {
            // First download - save hash for future verification
            info!(
                "First download of {} - SHA256: {} (saved for future verification)",
                model.id, actual_sha
            );
            self.verified_hashes
                .set(model.id.clone(), actual_sha.clone());
            if let Err(e) = self.verified_hashes.save() {
                warn!("Failed to save verified hashes: {}", e);
            }
        }

        // Move temp file to final location
        fs::rename(&temp_path, &dest_path).await?;

        info!("Successfully downloaded {} to {:?}", model.id, dest_path);
        Ok(dest_path)
    }

    /// Download a model only if not already present
    pub async fn ensure_model(
        &mut self,
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

    #[test]
    fn test_verified_hashes() {
        let mut hashes = VerifiedHashes::default();
        assert!(hashes.hashes.is_empty());

        // Add a hash
        hashes.set("model1".to_string(), "abc123".to_string());
        assert_eq!(hashes.get("model1"), Some(&"abc123".to_string()));
        assert_eq!(hashes.get("model2"), None);

        // Update a hash
        hashes.set("model1".to_string(), "def456".to_string());
        assert_eq!(hashes.get("model1"), Some(&"def456".to_string()));

        // Remove a hash
        let removed = hashes.remove("model1");
        assert_eq!(removed, Some("def456".to_string()));
        assert_eq!(hashes.get("model1"), None);

        // Clear all
        hashes.set("a".to_string(), "1".to_string());
        hashes.set("b".to_string(), "2".to_string());
        assert_eq!(hashes.hashes.len(), 2);
        hashes.clear();
        assert!(hashes.hashes.is_empty());
    }
}
