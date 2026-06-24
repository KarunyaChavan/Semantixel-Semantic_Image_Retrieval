# Standalone gRPC Inference Service for Semantixel.
#
# Exposes CLIP image/text embedding and OCR extraction as RPCs,
# allowing the existing Flask app to continue operating unchanged
# while providing a dedicated gRPC endpoint for the Go scanner
# and Go GraphQL gateway.

import io
import signal
from concurrent import futures
from typing import List, Optional

import grpc
from PIL import Image

from semantixel.core.logging import logger
from semantixel import semantixel_inference_pb2
from semantixel import semantixel_inference_pb2_grpc
from semantixel.services.model_manager import model_manager


class InferenceServicer(semantixel_inference_pb2_grpc.SemantixelInferenceServicer):
    """gRPC servicer implementing the SemantixelInference service.

    Delegates all ML operations to the shared ModelManager singleton
    so that models are loaded at most once regardless of access layer
    (Flask REST or gRPC).
    """

    def __init__(self) -> None:
        """Initialise servicer with shared model providers."""
        self._clip = model_manager.clip
        self._ocr = model_manager.ocr
        self._text_embed = model_manager.text_embed

    def _decode_images(
        self,
        blobs: List[bytes],
        context: grpc.ServicerContext,
    ) -> List[Image.Image]:
        """Decode raw image bytes into PIL Images.

        Args:
            blobs: Raw image bytes (JPEG, PNG, WebP, etc.).
            context: gRPC context for aborting on decode failure.

        Returns:
            List of RGB PIL Images, one per input blob.
        """
        images = []
        for blob in blobs:
            try:
                images.append(Image.open(io.BytesIO(blob)).convert("RGB"))
            except Exception as exc:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Failed to decode image: {exc}",
                )
        return images

    def _clip_model_name(self) -> str:
        """Return the active CLIP model checkpoint name."""
        return getattr(self._clip, "checkpoint", "unknown")

    def _ocr_model_name(self) -> str:
        """Return the active OCR model architecture string."""
        det = getattr(self._ocr, "det_arch", "unknown")
        reco = getattr(self._ocr, "reco_arch", "unknown")
        return f"{det}+{reco}"

    def _text_embed_model_name(self) -> str:
        """Return the active text embed model checkpoint name."""
        return getattr(self._text_embed, "checkpoint", "unknown")

    #  EmbedImage 

    def EmbedImage(
        self,
        request: semantixel_inference_pb2.EmbedImageRequest,
        context: grpc.ServicerContext,
    ) -> semantixel_inference_pb2.EmbedImageResponse:
        """Produce L2-normalised CLIP embeddings for one or more images.

        Args:
            request: Contains raw image bytes to embed.
            context: gRPC context for error reporting.

        Returns:
            EmbedImageResponse with per-image embeddings, model name,
            and embedding dimensionality.
        """
        if not request.images:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No images provided")

        pil_images = self._decode_images(request.images, context)
        embeddings = self._clip.get_image_embeddings(pil_images)
        proto_embeddings = [
            semantixel_inference_pb2.Embedding(values=emb) for emb in embeddings
        ]

        dim = len(embeddings[0]) if embeddings else 0

        return semantixel_inference_pb2.EmbedImageResponse(
            embeddings=proto_embeddings,
            model_name=self._clip_model_name(),
            embedding_dim=dim,
        )

    #  EmbedText 

    def EmbedText(
        self,
        request: semantixel_inference_pb2.EmbedTextRequest,
        context: grpc.ServicerContext,
    ) -> semantixel_inference_pb2.EmbedTextResponse:
        """Produce L2-normalised CLIP embeddings for one or more texts.

        Args:
            request: Contains text strings to embed (batched).
            context: gRPC context for error reporting.

        Returns:
            EmbedTextResponse with per-text embeddings, model name,
            and embedding dimensionality.
        """
        if not request.texts:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No texts provided")

        embeddings = [
            semantixel_inference_pb2.Embedding(
                values=self._clip.get_text_embeddings(t),
            )
            for t in request.texts
        ]

        dim = len(embeddings[0].values) if embeddings else 0

        return semantixel_inference_pb2.EmbedTextResponse(
            embeddings=embeddings,
            model_name=self._clip_model_name(),
            embedding_dim=dim,
        )

    #  ExtractOCR 

    def ExtractOCR(
        self,
        request: semantixel_inference_pb2.ExtractOCRRequest,
        context: grpc.ServicerContext,
    ) -> semantixel_inference_pb2.ExtractOCRResponse:
        """Extract text from one or more images via OCR.

        Args:
            request: Contains raw image bytes and an optional confidence
                     threshold (0.0-1.0). When omitted, 0.4 is used.
            context: gRPC context for error reporting.

        Returns:
            ExtractOCRResponse with one OCRResult per input image.
        """
        if not request.images:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No images provided")

        if request.HasField("threshold"):
            thresh = request.threshold
            if thresh < 0 or thresh > 1:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "threshold must be in [0.0, 1.0]",
                )
        else:
            thresh = 0.4

        pil_images = self._decode_images(request.images, context)
        texts = self._ocr.apply_ocr(pil_images, threshold=thresh)
        cleaned = [t if t is not None else "" for t in texts]
        results = [
            semantixel_inference_pb2.OCRResult(text=t) for t in cleaned
        ]
        return semantixel_inference_pb2.ExtractOCRResponse(results=results)
    

    def TextEmbedMiniLM(
        self,
        request: semantixel_inference_pb2.TextEmbedMiniLMRequest,
        context: grpc.ServicerContext,
    ) -> semantixel_inference_pb2.TextEmbedMiniLMResponse:
        """Produce embeddings for one or more texts via the configured text embed provider.

        Used by the Go scanner to embed OCR text for the text_collection.
        Uses a dedicated text embed model (MiniLM by default), separate from
        the CLIP model used in EmbedText.

        Args:
            request: Contains text strings to embed (batched).
            context: gRPC context for error reporting.

        Returns:
            TextEmbedMiniLMResponse with per-text embeddings, model name,
            and embedding dimensionality.
        """
        if not request.texts:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No texts provided")

        embeddings = [
            semantixel_inference_pb2.Embedding(
                values=self._text_embed.get_embeddings(t),
            )
            for t in request.texts
        ]

        dim = len(embeddings[0].values) if embeddings else 0

        return semantixel_inference_pb2.TextEmbedMiniLMResponse(
            embeddings=embeddings,
            model_name=self._text_embed_model_name(),
            embedding_dim=dim,
        )

    #  HealthCheck 

    def HealthCheck(
        self,
        request: semantixel_inference_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> semantixel_inference_pb2.HealthCheckResponse:
        """Return server readiness and loaded model metadata.

        Args:
            request: Empty health check request.
            context: gRPC context for error reporting.

        Returns:
            HealthCheckResponse with ServingStatus enum, model load
            state, device, and active model names.
        """
        clip_loaded = self._clip.model is not None
        ocr_loaded = self._ocr.model is not None
        text_embed_loaded = self._text_embed.model is not None

        if clip_loaded and ocr_loaded and text_embed_loaded:
            status = semantixel_inference_pb2.SERVING
        elif clip_loaded or ocr_loaded or text_embed_loaded:
            status = semantixel_inference_pb2.LOADING
        else:
            status = semantixel_inference_pb2.NOT_SERVING

        device = getattr(self._clip, "device", "unknown")

        return semantixel_inference_pb2.HealthCheckResponse(
            status=status,
            clip_loaded=clip_loaded,
            ocr_loaded=ocr_loaded,
            text_embed_loaded=text_embed_loaded,
            device=device,
            clip_model_name=self._clip_model_name(),
            ocr_model_name=self._ocr_model_name(),
            text_embed_model_name=self._text_embed_model_name(),
        )


class GrpcInferenceServer:
    """Manages the lifecycle of the gRPC inference server.

    Usage::

        server = GrpcInferenceServer()
        await server.start()
        # ... or simply:
        await server.serve()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 10,
    ) -> None:
        """Initialise server with host, port, and thread pool size."""
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self._server: Optional[grpc.aio.Server] = None

    @property
    def address(self) -> str:
        """Return the ``host:port`` string for this server instance."""
        return f"{self.host}:{self.port}"

    async def start(self) -> None:
        """Start the gRPC async server and bind to the configured address."""
        self._server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
        )

        semantixel_inference_pb2_grpc.add_SemantixelInferenceServicer_to_server(
            InferenceServicer(),
            self._server,
        )

        self._server.add_insecure_port(self.address)
        logger.info("gRPC Inference Server starting on %s", self.address)
        await self._server.start()

    async def serve(self) -> None:
        """Start and await server termination with graceful shutdown."""
        await self.start()
        await self._wait_for_termination()

    async def _wait_for_termination(self) -> None:
        """Block until the server is shut down."""
        if self._server is None:
            return
        await self._server.wait_for_termination()

    async def stop(self, grace: float = 5.0) -> None:
        """Gracefully stop the server with a configurable timeout."""
        if self._server is None:
            return
        logger.info("Shutting down gRPC Inference Server...")
        await self._server.stop(grace)
        model_manager.unload_all()
        logger.info("gRPC Inference Server stopped")


def create_grpc_server(
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 10,
) -> GrpcInferenceServer:
    """Factory function for GrpcInferenceServer.

    Convenience wrapper that mirrors ``semantixel.api.create_app()``.
    """
    return GrpcInferenceServer(
        host=host,
        port=port,
        max_workers=max_workers,
    )


def serve_forever(
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 10,
) -> None:
    """Blocking entry point that starts the gRPC server and handles shutdown.

    Usage from CLI::

        python -c "from semantixel.grpc_server import serve_forever; serve_forever()"
    """
    import asyncio

    async def _run() -> None:
        """Run the gRPC server until a shutdown signal is received."""
        server = GrpcInferenceServer(
            host=host,
            port=port,
            max_workers=max_workers,
        )

        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            """Handle SIGINT/SIGTERM by signalling the server to stop."""
            logger.info("Received shutdown signal")
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                pass

        await server.start()
        msg = f"gRPC Inference Server is ready on {server.address}"
        logger.info(msg)
        print(f"[semantixel] {msg}", flush=True)

        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            logger.info("Server cancelled via Ctrl+C")
            print("[semantixel] Shutting down...", flush=True)
        finally:
            await server.stop()

    asyncio.run(_run())


if __name__ == "__main__":
    serve_forever()
