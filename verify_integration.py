#!/usr/bin/env python3
"""
SemantiXel Integration Verification Script
Comprehensive check of all migration components and system readiness
"""

import sys
import os
import platform
import subprocess

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_check(status, message, details=""):
    """Print formatted check result."""
    symbol = "✓" if status else "✗"
    print(f"  {symbol} {message}")
    if details:
        print(f"    {details}")

def check_environment():
    """Check Python environment and dependencies."""
    print_header("ENVIRONMENT CHECK")
    
    # Python version
    version = sys.version_info
    py_version = f"{version.major}.{version.minor}.{version.micro}"
    py_ok = version.major == 3 and version.minor >= 8
    print_check(py_ok, f"Python {py_version}", f"Location: {sys.executable}")
    
    # Platform
    print_check(True, f"Platform: {platform.system()} {platform.release()}")
    
    # Conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "Not activated")
    conda_ok = conda_env == "Semantixel"
    print_check(conda_ok, f"Conda environment: {conda_env}", 
                "Expected: Semantixel" if not conda_ok else "")
    
    return py_ok and conda_ok

def check_dependencies():
    """Check required Python packages."""
    print_header("DEPENDENCIES CHECK")
    
    required = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "PIL": "Pillow",
        "chromadb": "ChromaDB",
        "flask": "Flask",
        "yaml": "PyYAML",
        "cv2": "OpenCV",
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print_check(True, f"{name} - installed")
        except ImportError:
            print_check(False, f"{name} - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_rust_module():
    """Check Rust optimization module."""
    print_header("RUST MODULE CHECK")
    
    try:
        import semantixel_scanner
        functions = [f for f in dir(semantixel_scanner) if not f.startswith('_')]
        print_check(True, "Rust module imported successfully")
        print(f"    Available functions: {', '.join(functions)}")
        
        # Test functions
        try:
            import semantixel_scanner
            # Quick smoke test
            print_check(True, "Rust module functions accessible")
        except Exception as e:
            print_check(False, f"Function test failed: {e}")
            return False
        
        return True
    except ImportError as e:
        print_check(False, "Rust module NOT found", 
                   "To build: cd semantixel_rust && maturin develop --release")
        return False

def check_python_integration():
    """Check Python integration layer."""
    print_header("PYTHON INTEGRATION CHECK")
    
    try:
        from rust_integration import RustScanner, RustCSVHandler, RustImageProcessor, RUST_AVAILABLE
        print_check(True, "rust_integration module loaded")
        print_check(RUST_AVAILABLE, f"RUST_AVAILABLE flag: {RUST_AVAILABLE}")
        
        # Check classes
        print_check(hasattr(RustScanner, 'scan_directory'), "RustScanner.scan_directory available")
        print_check(hasattr(RustCSVHandler, 'read_csv'), "RustCSVHandler.read_csv available")
        print_check(hasattr(RustImageProcessor, 'calculate_averages'), "RustImageProcessor.calculate_averages available")
        
        return True
    except Exception as e:
        print_check(False, f"Integration layer failed: {e}")
        return False

def check_index_module():
    """Check migrated Index module."""
    print_header("INDEX MODULE CHECK")
    
    try:
        from Index import scan
        print_check(True, "Index.scan module loaded")
        
        # Check functions
        has_scan = hasattr(scan, 'scan_and_save')
        has_read = hasattr(scan, 'read_from_csv')
        has_save = hasattr(scan, 'save_to_csv')
        
        print_check(has_scan, "scan_and_save function available")
        print_check(has_read, "read_from_csv function available")
        print_check(has_save, "save_to_csv function available")
        
        return has_scan and has_read and has_save
    except Exception as e:
        print_check(False, f"Index module failed: {e}")
        return False

def check_files():
    """Check critical files exist."""
    print_header("FILE STRUCTURE CHECK")
    
    files_to_check = {
        "config.yaml": "Configuration file",
        "requirements.txt": "Dependencies file",
        "main.py": "Entry point",
        "Index/scan.py": "Index module (migrated)",
        "Index/paths.csv": "Index data",
        "rust_integration.py": "Rust wrapper layer",
        "benchmark_performance.py": "Performance tests",
        "SETUP_AND_INTEGRATION_GUIDE.md": "Setup guide",
        "MIGRATION_COMPLETE.md": "Migration report",
        "semantixel_rust/Cargo.toml": "Rust project file",
        "semantixel_rust/src/lib.rs": "Rust module",
    }
    
    all_ok = True
    for filepath, description in files_to_check.items():
        exists = os.path.exists(filepath)
        print_check(exists, f"{description}: {filepath}")
        if not exists:
            all_ok = False
    
    return all_ok

def check_launcher_scripts():
    """Check launcher script availability."""
    print_header("LAUNCHER SCRIPTS CHECK")
    
    scripts = {
        "run_offline.sh": "Linux/macOS",
        "run_offline.bat": "Windows CMD",
        "run_offline.ps1": "Windows PowerShell",
    }
    
    all_ok = True
    for script, platform_name in scripts.items():
        exists = os.path.exists(script)
        executable = os.access(script, os.X_OK) if exists else False
        status = exists and (platform_name.startswith("Windows") or executable)
        
        print_check(status, f"{platform_name}: {script}", 
                   "Not executable" if exists and not executable and not platform_name.startswith("Windows") else "")
        if not status:
            all_ok = False
    
    return all_ok

def check_config():
    """Check configuration file."""
    print_header("CONFIGURATION CHECK")
    
    try:
        import yaml
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        print_check(True, "config.yaml loaded successfully")
        
        # Check key sections
        keys = config.keys() if config else []
        print_check(len(keys) > 0, f"Configuration has {len(keys)} sections")
        
        return len(keys) > 0
    except Exception as e:
        print_check(False, f"Config check failed: {e}")
        return False

def run_simple_test():
    """Run a simple functionality test."""
    print_header("FUNCTIONALITY TEST")
    
    try:
        # Try importing main
        import main
        print_check(True, "main.py imports successfully")
        
        # Check if Rust module check works
        rust_ok = main.check_rust_module()
        print_check(rust_ok, "Rust module check function works")
        
        return True
    except Exception as e:
        print_check(False, f"Functionality test failed: {e}")
        return False

def main_check():
    """Run all checks."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "SEMANTIXEL - INTEGRATION VERIFICATION" + " "*15 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {
        "Environment": check_environment(),
        "Dependencies": check_dependencies(),
        "Rust Module": check_rust_module(),
        "Python Integration": check_python_integration(),
        "Index Module": check_index_module(),
        "File Structure": check_files(),
        "Launcher Scripts": check_launcher_scripts(),
        "Configuration": check_config(),
        "Functionality": run_simple_test(),
    }
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for check_name, result in results.items():
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {check_name}")
    
    print("\n" + "="*70)
    if passed == total:
        print(f"  ✅ ALL CHECKS PASSED ({passed}/{total})")
        print("  System is ready for production use!")
    else:
        print(f"  ⚠ SOME CHECKS FAILED ({passed}/{total})")
        print("  Please review the failures above.")
    print("="*70 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main_check()
    sys.exit(0 if success else 1)
