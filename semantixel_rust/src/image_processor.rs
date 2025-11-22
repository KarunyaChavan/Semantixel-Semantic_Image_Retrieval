use crate::errors::{Result, SemantiXelError};
use image::open;
use rayon::prelude::*;
use std::path::Path;

pub struct ImageProcessor;

impl ImageProcessor {
    pub fn new() -> Self {
        Self
    }

    /// Calculate average pixel value for a single image
    pub fn calculate_average(image_path: &str) -> Result<i32> {
        let img = open(image_path)
            .map_err(|e| SemantiXelError::ImageError(e))?;

        // Convert to grayscale for average calculation
        let gray = img.to_luma8();
        let pixels: Vec<u32> = gray.iter().map(|p| *p as u32).collect();

        if pixels.is_empty() {
            return Ok(0);
        }

        let sum: u32 = pixels.iter().sum();
        let average = (sum / pixels.len() as u32) as i32;

        Ok(average)
    }

    /// Calculate averages for multiple images in parallel (skip errors)
    pub fn calculate_averages(&self, image_paths: &[String]) -> Result<Vec<i32>> {
        let averages: Vec<i32> = image_paths
            .par_iter()
            .map(|path| Self::calculate_average(path).unwrap_or(0))
            .collect();

        Ok(averages)
    }

    /// Get image dimensions
    pub fn get_dimensions(image_path: &str) -> Result<(u32, u32)> {
        let img = open(image_path)
            .map_err(|e| SemantiXelError::ImageError(e))?;

        Ok((img.width(), img.height()))
    }

    /// Validate image file and return metadata
    pub fn validate_image(image_path: &str) -> Result<ImageMetadata> {
        if !Path::new(image_path).exists() {
            return Err(SemantiXelError::InvalidPath(image_path.to_string()));
        }

        let img = open(image_path)
            .map_err(|e| SemantiXelError::ImageError(e))?;

        let (width, height) = (img.width(), img.height());
        let file_size = std::fs::metadata(image_path)?.len();

        Ok(ImageMetadata {
            path: image_path.to_string(),
            width,
            height,
            file_size,
            format: format!("{:?}", img.color()),
        })
    }

    /// Process batch of images and return compressed/preprocessed data
    pub fn process_batch(&self, image_paths: &[String]) -> Result<Vec<Vec<u8>>> {
        let results: Result<Vec<Vec<u8>>> = image_paths
            .par_iter()
            .map(|path| self.process_single_image(path))
            .collect();

        results
    }

    /// Process a single image and return compressed format
    fn process_single_image(&self, image_path: &str) -> Result<Vec<u8>> {
        let img = open(image_path)
            .map_err(|e| SemantiXelError::ImageError(e))?;

        // Resize to smaller dimensions for faster processing (e.g., 224x224 for CLIP)
        let thumbnail = img.thumbnail(224, 224);

        // Convert to RGB8
        let rgb = thumbnail.to_rgb8();

        // Return raw RGB bytes
        Ok(rgb.to_vec())
    }

    /// Get statistics for a batch of images
    pub fn batch_statistics(&self, image_paths: &[String]) -> Result<BatchStatistics> {
        let mut total_files = 0;
        let mut valid_files = 0;
        let mut total_size: u64 = 0;
        let mut total_width = 0u64;
        let mut total_height = 0u64;

        for path in image_paths {
            total_files += 1;
            match Self::validate_image(path) {
                Ok(metadata) => {
                    valid_files += 1;
                    total_size += metadata.file_size;
                    total_width += metadata.width as u64;
                    total_height += metadata.height as u64;
                }
                Err(_) => {}
            }
        }

        let avg_width = if valid_files > 0 {
            (total_width / valid_files as u64) as u32
        } else {
            0
        };

        let avg_height = if valid_files > 0 {
            (total_height / valid_files as u64) as u32
        } else {
            0
        };

        Ok(BatchStatistics {
            total_files: total_files as u32,
            valid_files: valid_files as u32,
            total_size,
            avg_width,
            avg_height,
        })
    }
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ImageMetadata {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub file_size: u64,
    pub format: String,
}

#[derive(Debug, Clone)]
pub struct BatchStatistics {
    pub total_files: u32,
    pub valid_files: u32,
    pub total_size: u64,
    pub avg_width: u32,
    pub avg_height: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let processor = ImageProcessor::new();
        assert!(std::mem::size_of_val(&processor) >= 0);
    }
}
