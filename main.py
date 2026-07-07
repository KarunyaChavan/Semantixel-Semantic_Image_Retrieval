"""CLI entry point for the Semantixel application.

Usage::

    python main.py                       # gRPC → Go scanner → Flask (default)
    python main.py --serve               # Flask only (skip indexing)
    python main.py --grpc                # gRPC server only
    python main.py --scan                # Python scanner only
    python main.py --settings            # Open settings UI
"""

import argparse
import os
import shutil
import platform
import socket
import subprocess
import sys
import threading
import time

from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.services.index_service import IndexService

DB_PATH = "db"
GO_SCANNER_PATH = os.path.join("cmd", "scanner", "scanner.exe")


def _start_grpc_server(port: int) -> threading.Thread:
    """Start gRPC inference server in a daemon thread."""
    from semantixel.grpc_server import serve_forever

    t = threading.Thread(
        target=serve_forever,
        kwargs={"port": port},
        daemon=True,
    )
    t.start()
    logger.info("gRPC server starting on port %d (background thread)", port)
    return t


def _wait_for_port(host: str, port: int, timeout: float = 60.0, interval: float = 0.5) -> bool:
    """Block until *port* is accepting TCP connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(interval)
    return False


def _run_go_scanner(dirs, addr="localhost:50051", concurrency=4, batch=32):
    """Run the Go scanner as a subprocess and wait for completion."""
    if not os.path.isfile(GO_SCANNER_PATH):
        logger.warning("Go scanner not found at %s — falling back to Python scanner", GO_SCANNER_PATH)
        return False

    dirs_str = ",".join(dirs) if isinstance(dirs, list) else dirs
    cmd = [
        GO_SCANNER_PATH,
        "-dirs", dirs_str,
        "-addr", addr,
        "-concurrency", str(concurrency),
        "-batch", str(batch),
    ]
    logger.info("Running Go scanner: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        logger.info("Go scanner completed successfully")
        return True
    else:
        logger.error("Go scanner failed (exit code %d)", result.returncode)
        return False


def _run_python_scanner():
    """Fall back to the Python-based scanner."""
    logger.info("Falling back to Python scanner")
    index_service = IndexService()
    index_service.run_full_scan()


def _start_flask():
    """Start the Flask web server (blocking)."""
    from wsgi import app

    logger.info("Starting Flask server on port %d", config.port)
    app.run(host="0.0.0.0", port=config.port, threaded=True)


def _handle_settings():
    subprocess.run([sys.executable, "settings.py"])


def _handle_delete_index():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        logger.info("Index deleted successfully")
    else:
        logger.warning("Index does not exist")


def _handle_get_index():
    db_path = os.path.abspath(DB_PATH)
    if os.path.exists(db_path):
        print(db_path)
    else:
        print("Index does not exist")


def _handle_open_config():
    config_path = "config.yaml"
    if platform.system() == "Windows":
        os.system("start %s" % config_path)
    elif platform.system() == "Darwin":
        os.system("open %s" % config_path)
    else:
        os.system("xdg-open %s" % config_path)


def main():
    """Parse CLI arguments and dispatch to the requested mode."""
    parser = argparse.ArgumentParser(
        description="Semantixel: Semantic Image Retrieval"
    )
    parser.add_argument(
        "--settings",
        action="store_true",
        help="Open Semantixel settings",
    )
    parser.add_argument(
        "--delete-index",
        action="store_true",
        help="Delete the index (Vector database)",
    )
    parser.add_argument(
        "--get-index",
        action="store_true",
        help="Get the index path",
    )
    parser.add_argument(
        "--open-config-file",
        action="store_true",
        help="Open config.yaml",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start only the Flask server (skip indexing)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Perform a full media scan and index update (Python scanner)",
    )
    parser.add_argument("--grpc", action="store_true", help="Start the gRPC Inference Server only")
    parser.add_argument("--grpc-port", type=int, default=50051, help="gRPC server port (default: 50051)")

    args = parser.parse_args()

    if args.grpc:
        from semantixel.grpc_server import serve_forever
        serve_forever(port=args.grpc_port)
        return

    if args.settings:
        _handle_settings()
        return

    if args.delete_index:
        _handle_delete_index()
        return

    if args.get_index:
        _handle_get_index()
        return

    if args.open_config_file:
        _handle_open_config()
        return

    if args.scan:
        _run_python_scanner()
        return

    if args.serve:
        _start_flask()
        return

    # ── Default mode: unified pipeline ───────────────────────────────
    from semantixel.unified_server import UnifiedServer

    logger.info("=== Semantixel: unified pipeline ===")

    # 1. Start unified server (gRPC + HTTP) in daemon thread
    unified = UnifiedServer(grpc_port=50051, http_port=config.port)
    server_thread = threading.Thread(target=unified.start, daemon=True)
    server_thread.start()

    # 2. Wait for gRPC to accept connections
    logger.info("Waiting for gRPC server to be ready…")
    if not _wait_for_port("localhost", 50051):
        logger.warning("gRPC server did not become ready — proceeding without Go scanner")
        server_thread.join()
        return

    logger.info("gRPC server is ready")

    # 3. Run Go scanner (with Python fallback)
    scanned_ok = _run_go_scanner(
        dirs=config.include_directories,
        addr="localhost:50051",
        concurrency=4,
        batch=config.batch_size,
    )
    if not scanned_ok:
        _run_python_scanner()

    # 4. Keep alive until user hits Ctrl+C
    logger.info("All services running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
