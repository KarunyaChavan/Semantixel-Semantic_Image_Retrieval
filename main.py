import argparse
import os
import shutil
import platform
import sys
from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.services.index_service import IndexService

def main():
    parser = argparse.ArgumentParser(description="Semantixel: Semantic Image Retrieval")
    parser.add_argument("--settings", action="store_true", help="Open Semantixel settings")
    parser.add_argument("--delete-index", action="store_true", help="Delete the index (Vector database)")
    parser.add_argument("--get-index", action="store_true", help="Get the index path")
    parser.add_argument("--open-config-file", action="store_true", help="Open config.yaml")
    parser.add_argument("--serve", action="store_true", help="Start the Flask server")
    parser.add_argument("--scan", action="store_true", help="Perform a full media scan and index update")
    
    args = parser.parse_args()
    
    # Initialize service only if needed to avoid loading too much
    index_service = None

    if args.settings:
        import subprocess
        subprocess.run([sys.executable, "settings.py"])
        return

    if args.delete_index:
        db_path = config.model_config.get("db_path", "db")
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            logger.info("Index deleted successfully")
        else:
            logger.warning("Index does not exist")
        return

    if args.get_index:
        db_path = os.path.abspath("db")
        if os.path.exists(db_path):
            print(db_path)
        else:
            print("Index does not exist")
        return

    if args.open_config_file:
        config_path = "config.yaml"
        if platform.system() == "Windows":
            os.system(f"start {config_path}")
        elif platform.system() == "Darwin":
            os.system(f"open {config_path}")
        else:
            os.system(f"xdg-open {config_path}")
        return

    if args.scan:
        index_service = IndexService()
        index_service.run_full_scan()
        return

    if args.serve:
        from wsgi import app
        app.run(host="0.0.0.0", port=config.port)
        return

    # Default behavior: Sync index and then serve
    logger.info("Starting Semantixel in default mode (Scan + Serve)")
    index_service = IndexService()
    index_service.run_full_scan()
    
    from wsgi import app
    app.run(host="0.0.0.0", port=config.port)

if __name__ == "__main__":
    main()
