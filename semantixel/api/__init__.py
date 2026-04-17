from flask import Flask
from flask_cors import CORS
from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.services.index_service import IndexService
from semantixel.services.search_service import SearchService
from semantixel.services.face_service import FaceService

def create_app():
    """
    Flask Application Factory.
    Initializes dependencies and registers blueprints.
    """
    app = Flask(__name__, static_folder="../../UI/Semantixel WebUI")
    CORS(app)

    # Initialize services
    index_service = IndexService()
    face_service = FaceService()
    search_service = SearchService(index_service, face_service)
    
    app.index_service = index_service
    app.face_service = face_service
    app.search_service = search_service
    app.google_drive_source = index_service.google_drive_source

    # Register Blueprints
    from semantixel.api.routes import main_bp
    app.register_blueprint(main_bp)

    logger.info(f"Semantixel Server initialized on port {config.port}")
    return app
