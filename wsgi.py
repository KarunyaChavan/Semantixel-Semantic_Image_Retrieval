"""WSGI entry point for production deployment.

Usage::

    gunicorn wsgi:app
    waitress-serve wsgi:app
    python wsgi.py          # development server
"""

from semantixel.api import create_app
from semantixel.core.config import config

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.port, threaded=True)
