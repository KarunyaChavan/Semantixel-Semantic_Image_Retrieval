use crate::errors::Result;
use rayon::prelude::*;
use std::path::Path;
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct ScanResult {
    pub paths: Vec<String>,
    pub total_files: usize,
    pub elapsed_ms: u128,
}

pub struct DirectoryScanner {
    directories: Vec<String>,
    exclude_dirs: Vec<String>,
    extensions: Vec<String>,
}

impl DirectoryScanner {
    pub fn new(directory: String, exclude_dirs: Vec<String>, extensions: Vec<String>) -> Self {
        Self {
            directories: vec![directory],
            exclude_dirs,
            extensions: extensions
                .iter()
                .map(|e| e.to_lowercase())
                .collect(),
        }
    }

    pub fn with_multiple_dirs(directories: Vec<String>, exclude_dirs: Vec<String>, extensions: Vec<String>) -> Self {
        Self {
            directories,
            exclude_dirs,
            extensions: extensions
                .iter()
                .map(|e| e.to_lowercase())
                .collect(),
        }
    }

    /// Scan directories and return image paths
    pub fn scan(&self) -> Result<ScanResult> {
        let start = std::time::Instant::now();

        let all_images: Vec<String> = self
            .directories
            .par_iter()
            .flat_map(|dir| self.scan_directory(dir))
            .collect();

        let total_files = all_images.len();
        let elapsed_ms = start.elapsed().as_millis();

        Ok(ScanResult {
            paths: all_images,
            total_files,
            elapsed_ms,
        })
    }

    /// Scan a single directory recursively
    fn scan_directory(&self, directory: &str) -> Vec<String> {
        WalkDir::new(directory)
            .into_iter()
            .par_bridge()
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();

                // Check if it's a file
                if !path.is_file() {
                    return None;
                }

                // Check if file should be excluded
                if self.is_excluded(&path) {
                    return None;
                }

                // Check file name patterns (e.g., skip hidden files)
                let file_name = path.file_name()?.to_str()?;
                if file_name.starts_with("._") {
                    return None;
                }

                // Check extension
                let extension = path
                    .extension()?
                    .to_str()?
                    .to_lowercase();

                if self.extensions.contains(&extension) {
                    path.to_str().map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if a path should be excluded
    fn is_excluded(&self, path: &Path) -> bool {
        for exclude_dir in &self.exclude_dirs {
            if path.starts_with(exclude_dir) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scanner_creation() {
        let scanner = DirectoryScanner::new(
            ".".to_string(),
            vec![],
            vec!["jpg".to_string(), "png".to_string()],
        );
        assert_eq!(scanner.extensions.len(), 2);
    }
}
