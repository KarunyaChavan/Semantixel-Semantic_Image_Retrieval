import os
from typing import List
from urllib.parse import urlparse

def is_safe_path(path: str, allowed_dirs: List[str]) -> bool:
    """
    Check if a given path is within one of the allowed directories.
    Prevents path traversal attacks.
    """
    # Canonicalize the path
    abs_path = os.path.abspath(path)
    
    for allowed_dir in allowed_dirs:
        abs_allowed_dir = os.path.abspath(allowed_dir)
        if os.path.commonpath([abs_path, abs_allowed_dir]) == abs_allowed_dir:
            return True
            
    return False

def is_safe_url(url: str) -> bool:
    """
    Validation for external URLs to prevent SSRF.
    Basic implementation: allow only http/https and non-local addresses.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    
    # Simple check for localhost/internal IP ranges (optional, depends on use case)
    # For now, let's just ensure it's a valid remote URL structure
    if not parsed.netloc or parsed.netloc in ("localhost", "127.0.0.1", "0.0.0.0"):
        return False
        
    return True
