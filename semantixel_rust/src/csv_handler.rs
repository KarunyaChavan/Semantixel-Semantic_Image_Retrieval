use crate::errors::{Result, SemantiXelError};
use csv::{Reader, Writer};
use std::fs::File;
use std::path::Path;

pub struct CsvHandler {
    file_path: String,
}

impl CsvHandler {
    pub fn new(file_path: String) -> Self {
        Self { file_path }
    }

    /// Read paths and averages from CSV file
    pub fn read(&self) -> Result<(Vec<String>, Vec<i32>)> {
        let file = File::open(&self.file_path)?;
        let mut reader = Reader::from_reader(file);

        let mut paths = Vec::new();
        let mut averages = Vec::new();

        for result in reader.records() {
            let record = result?;
            if record.len() < 2 {
                continue;
            }

            let path = record[0].to_string();
            let average: i32 = record[1]
                .parse()
                .unwrap_or(0);

            paths.push(path);
            averages.push(average);
        }

        Ok((paths, averages))
    }

    /// Write paths and averages to CSV file
    pub fn write(&self, paths: &[String], averages: &[i32]) -> Result<()> {
        if paths.len() != averages.len() {
            return Err(SemantiXelError::ProcessingError(
                "Paths and averages length mismatch".to_string(),
            ));
        }

        let file = File::create(&self.file_path)?;
        let mut writer = Writer::from_writer(file);

        // Write header
        writer.write_record(&["path", "average"])?;

        // Write records
        for (path, average) in paths.iter().zip(averages.iter()) {
            writer.write_record(&[path, &average.to_string()])?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Append records to existing CSV file
    pub fn append(&self, paths: &[String], averages: &[i32]) -> Result<()> {
        if paths.len() != averages.len() {
            return Err(SemantiXelError::ProcessingError(
                "Paths and averages length mismatch".to_string(),
            ));
        }

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)?;

        let mut writer = Writer::from_writer(file);

        for (path, average) in paths.iter().zip(averages.iter()) {
            writer.write_record(&[path, &average.to_string()])?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Check if CSV file exists
    pub fn exists(&self) -> bool {
        Path::new(&self.file_path).exists()
    }

    /// Get file size
    pub fn file_size(&self) -> Result<u64> {
        Ok(std::fs::metadata(&self.file_path)?.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_write_and_read() {
        let test_file = "test_paths.csv";
        
        // Clean up if exists
        let _ = fs::remove_file(test_file);

        let paths = vec!["image1.jpg".to_string(), "image2.png".to_string()];
        let averages = vec![128, 200];

        let handler = CsvHandler::new(test_file.to_string());
        handler.write(&paths, &averages).unwrap();

        let (read_paths, read_averages) = handler.read().unwrap();
        assert_eq!(read_paths.len(), 2);
        assert_eq!(read_averages, averages);

        // Clean up
        let _ = fs::remove_file(test_file);
    }
}
