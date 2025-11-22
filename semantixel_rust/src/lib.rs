mod scanner;
mod csv_handler;
mod image_processor;
mod errors;

use pyo3::prelude::*;

pub use scanner::{DirectoryScanner, ScanResult};
pub use csv_handler::CsvHandler;
pub use image_processor::ImageProcessor;

/// Python module initialization
#[pymodule]
fn semantixel_scanner(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_scan_directory, m)?)?;
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(write_csv, m)?)?;
    m.add_function(wrap_pyfunction!(process_images_batch, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_image_averages, m)?)?;
    
    Ok(())
}

/// Fast directory scanning - exposed to Python
#[pyfunction]
fn fast_scan_directory(
    directory: String,
    exclude_dirs: Vec<String>,
    extensions: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    let exts = extensions.unwrap_or_else(|| {
        vec!["jpg", "jpeg", "png", "gif", "bmp"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    });
    
    let scanner = DirectoryScanner::new(directory, exclude_dirs, exts);
    let results = scanner.scan().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Scan error: {}", e))
    })?;
    
    Ok(results.paths)
}

/// Read CSV file - exposed to Python
#[pyfunction]
fn read_csv(file_path: String) -> PyResult<(Vec<String>, Vec<i32>)> {
    let handler = CsvHandler::new(file_path);
    let (paths, averages) = handler.read().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("CSV read error: {}", e))
    })?;
    
    Ok((paths, averages))
}

/// Write CSV file - exposed to Python
#[pyfunction]
fn write_csv(file_path: String, paths: Vec<String>, averages: Vec<i32>) -> PyResult<()> {
    let handler = CsvHandler::new(file_path);
    handler.write(&paths, &averages).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("CSV write error: {}", e))
    })?;
    
    Ok(())
}

/// Process batch of images - exposed to Python
#[pyfunction]
fn process_images_batch(image_paths: Vec<String>) -> PyResult<Vec<Vec<u8>>> {
    let processor = ImageProcessor::new();
    let results = processor
        .process_batch(&image_paths)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image processing error: {}", e))
        })?;
    
    Ok(results)
}

/// Calculate image averages in parallel - exposed to Python
#[pyfunction]
fn calculate_image_averages(image_paths: Vec<String>) -> PyResult<Vec<i32>> {
    let processor = ImageProcessor::new();
    let averages = processor.calculate_averages(&image_paths).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Average calculation error: {}", e))
    })?;
    
    Ok(averages)
}
