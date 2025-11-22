"""
Integration layer for Rust modules.
This module provides Python bindings to call Rust functions.

To build:
    cd semantixel_rust
    maturin develop --release
"""

import sys
import time
from typing import List, Tuple, Optional
from pathlib import Path


# Try to import the compiled Rust module
try:
    import semantixel_scanner
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("WARNING: semantixel_scanner not available. Install with: cd semantixel_rust && maturin develop --release")


class RustScanner:
    """Wrapper for Rust directory scanner."""
    
    @staticmethod
    def scan_directory(
        directory: str,
        exclude_dirs: Optional[List[str]] = None,
        extensions: Optional[List[str]] = None
    ) -> Tuple[List[str], int]:
        """
        Scan directory for images using Rust backend.
        
        Args:
            directory: Directory to scan
            exclude_dirs: Directories to exclude
            extensions: File extensions to search for (default: jpg, jpeg, png, gif, bmp)
        
        Returns:
            Tuple of (image_paths, elapsed_ms)
        """
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust module not available. Build it first.")
        
        exclude_dirs = exclude_dirs or []
        extensions = extensions or ["jpg", "jpeg", "png", "gif", "bmp"]
        
        start = time.time()
        paths = semantixel_scanner.fast_scan_directory(directory, exclude_dirs, extensions)
        elapsed_ms = int((time.time() - start) * 1000)
        
        return paths, elapsed_ms


class RustCSVHandler:
    """Wrapper for Rust CSV operations."""
    
    @staticmethod
    def read_csv(file_path: str) -> Tuple[List[str], List[int]]:
        """
        Read paths and averages from CSV using Rust backend.
        
        Args:
            file_path: Path to CSV file
        
        Returns:
            Tuple of (paths, averages)
        """
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust module not available. Build it first.")
        
        return semantixel_scanner.read_csv(file_path)
    
    @staticmethod
    def write_csv(file_path: str, paths: List[str], averages: List[int]) -> None:
        """
        Write paths and averages to CSV using Rust backend.
        
        Args:
            file_path: Path to CSV file
            paths: List of image paths
            averages: List of average pixel values
        """
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust module not available. Build it first.")
        
        if len(paths) != len(averages):
            raise ValueError("Paths and averages must have same length")
        
        semantixel_scanner.write_csv(file_path, paths, averages)


class RustImageProcessor:
    """Wrapper for Rust image processing."""
    
    @staticmethod
    def calculate_averages(image_paths: List[str]) -> List[int]:
        """
        Calculate average pixel values for images using Rust backend.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of average pixel values
        """
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust module not available. Build it first.")
        
        return semantixel_scanner.calculate_image_averages(image_paths)
    
    @staticmethod
    def process_batch(image_paths: List[str]) -> List[bytes]:
        """
        Process batch of images using Rust backend.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of processed image data
        """
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust module not available. Build it first.")
        
        return semantixel_scanner.process_images_batch(image_paths)


# Drop-in replacement functions for existing Python code

def fast_scan_for_images_rust(
    directories: List[str],
    exclude_directories: Optional[List[str]] = None
) -> Tuple[List[str], float]:
    """
    Drop-in replacement for Index/scan_default.py::fast_scan_for_images
    
    Usage:
        # In Index/scan.py, replace:
        from Index.scan_default import fast_scan_for_images
        # With:
        from rust_integration import fast_scan_for_images_rust as fast_scan_for_images
    """
    if not RUST_AVAILABLE:
        # Fallback to Python implementation
        from Index.scan_default import fast_scan_for_images as python_impl
        return python_impl(directories, exclude_directories)
    
    exclude_dirs = exclude_directories or []
    all_images = []
    
    start = time.time()
    for directory in directories:
        paths, _ = RustScanner.scan_directory(directory, exclude_dirs)
        all_images.extend(paths)
    
    elapsed = time.time() - start
    return all_images, elapsed


if __name__ == "__main__":
    print(f"Rust module available: {RUST_AVAILABLE}")
    
    if RUST_AVAILABLE:
        # Test examples
        print("\n=== Testing Rust Integration ===\n")
        
        # Test 1: Scan current directory
        print("Test 1: Directory scanning")
        try:
            paths, elapsed_ms = RustScanner.scan_directory(".")
            print(f"  Found {len(paths)} images in {elapsed_ms}ms")
            if paths:
                print(f"  First few results: {paths[:3]}")
        except Exception as e:
            print(f"  Error: {e}")
        
        print("\nTest 2: CSV operations")
        try:
            test_paths = ["image1.jpg", "image2.png"]
            test_averages = [128, 200]
            RustCSVHandler.write_csv("test.csv", test_paths, test_averages)
            print("  CSV written")
            
            read_paths, read_averages = RustCSVHandler.read_csv("test.csv")
            print(f"  CSV read: {len(read_paths)} records")
            
            # Cleanup
            Path("test.csv").unlink(missing_ok=True)
        except Exception as e:
            print(f"  Error: {e}")
        
        print("\nRust integration tests completed!")
    else:
        print("\nTo use Rust integration:")
        print("  1. cd semantixel_rust")
        print("  2. maturin develop --release")
        print("  3. Rerun this script")
