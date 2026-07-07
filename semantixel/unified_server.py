"""Single-process server combining gRPC and Flask (HTTP).

Eliminates the need for separate ``wsgi.py`` and ``grpc_server.py``
processes.  Services (index, face, search) are created once and shared
between both protocols, avoiding duplicate ChromaDB connections.

Usage::

    from semantixel.unified_server import UnifiedServer
    UnifiedServer().start()
"""

import asyncio
import threading

from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.services.index_service import IndexService
from semantixel.services.search_service import SearchService
from semantixel.services.face_service import FaceService
from semantixel.api import create_app


class UnifiedServer:
    """Serves gRPC and HTTP (Flask) in a single process.

    Services are initialised once and injected into both the gRPC
    servicer and the Flask application context.
    """

    def __init__(
        self,
        grpc_host: str = "0.0.0.0",
        grpc_port: int = 50051,
        http_port: int = 23107,
    ):
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.http_port = http_port

        # Shared services — created once, used by both protocols
        self.index_service = IndexService()
        self.face_service = FaceService()
        self.search_service = SearchService(self.index_service, self.face_service)

    def start(self) -> None:
        """Start both servers — HTTP in a daemon thread, gRPC (blocking)."""
        logger.info("Unified server starting (gRPC :%d, HTTP :%d)", self.grpc_port, self.http_port)

        # Index Google Drive in background (shared, runs once)
        self._index_drive()

        # Flask in daemon thread
        http_thread = threading.Thread(target=self._run_http, daemon=True)
        http_thread.start()

        # gRPC (blocks until shutdown)
        asyncio.run(self._run_grpc())

    def _run_http(self) -> None:
        """Build and run the Flask application (blocking in this thread)."""
        app = create_app(
            index_service=self.index_service,
            face_service=self.face_service,
            search_service=self.search_service,
        )
        logger.info("HTTP server starting on port %d", self.http_port)
        app.run(host="0.0.0.0", port=self.http_port, threaded=True)

    async def _run_grpc(self) -> None:
        """Build and run the gRPC server (async, blocks)."""
        from semantixel.grpc_server import create_grpc_server

        server = create_grpc_server(
            host=self.grpc_host,
            port=self.grpc_port,
            index_service=self.index_service,
        )
        logger.info("gRPC server starting on %s:%d", self.grpc_host, self.grpc_port)
        await server.serve()

    def _index_drive(self) -> None:
        """Index Google Drive items in a background daemon thread."""
        gdrive = self.index_service.google_drive_source
        if not gdrive.is_enabled():
            return

        def _run():
            try:
                items = gdrive.list_media()
                if items:
                    self.index_service.image_indexer.index_images(
                        items, google_drive_source=gdrive
                    )
                    self.index_service.bm25_service.rebuild_from_collection(
                        self.index_service.text_collection
                    )
                    logger.info("Google Drive indexing complete (%d items)", len(items))
            except Exception as exc:
                logger.warning("Google Drive indexing skipped: %s", exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
