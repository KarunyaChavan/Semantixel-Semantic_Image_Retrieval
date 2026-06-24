"""Flask application factory for the Semantixel REST API.

Creates and configures the WSGI application, wires up services, and
registers blueprint routes.
"""

from flask import Flask
from flask_cors import CORS
from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.services.index_service import IndexService
from semantixel.services.search_service import SearchService
from semantixel.services.face_service import FaceService
from semantixel.services.model_manager import model_manager


def create_app() -> Flask:
    """Create and return a configured Flask application instance.

    Initialises all services (index, face, search) and attaches them to
    the app context so that route handlers can access them via
    ``current_app.<service>``.

    Returns:
        A ready-to-run :class:`Flask` application.
    """
    app = Flask(__name__, static_folder="../../UI/Semantixel WebUI")
    CORS(app)

    index_service = IndexService()
    face_service = FaceService()
    search_service = SearchService(index_service, face_service)

    app.index_service = index_service
    app.face_service = face_service
    app.search_service = search_service
    app.google_drive_source = index_service.google_drive_source

    # Warm up all models eagerly so the first search request is fast
    for name, loader in [
        ("CLIP", model_manager.clip),
        ("text_embed", model_manager.text_embed),
    ]:
        try:
            loader.load()
        except Exception as exc:
            logger.warning("%s warmup skipped: %s", name, exc)

    if config.audio.clap_enabled:
        try:
            model_manager.clap.load()
        except Exception as exc:
            logger.warning("CLAP warmup skipped: %s", exc)

    from semantixel.api.routes import main_bp

    app.register_blueprint(main_bp)

    logger.info("Semantixel Server initialized on port %d", config.port)
    return app
