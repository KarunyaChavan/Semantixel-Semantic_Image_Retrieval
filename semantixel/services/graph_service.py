"""Semantic graph generation — builds a similarity graph from embeddings."""

import os
import time
from typing import Any, Dict, List
import torch
import torch.nn.functional as F
from semantixel.core.logging import logger
from semantixel.media import parse_media_id


class GraphService:
    """Generates a semantic similarity graph from a ChromaDB collection.

    Each node represents an indexed media item.  Edges connect the top-3
    nearest neighbours (cosine similarity > 0.5).

    To keep rendering fast and avoid OOM, the graph is limited to
    ``MAX_NODES`` (500).  If the collection has more items, the most
    recently indexed 500 are used.
    """

    TOP_K_NEIGHBORS = 3
    MIN_SIMILARITY = 0.5
    MAX_NODES = 500

    def __init__(self, image_collection):
        self.image_collection = image_collection

    def generate(self) -> Dict[str, Any]:
        """Build and return the full graph.

        Returns:
            A dict with ``"nodes"`` and ``"links"`` lists suitable for
            JSON serialisation.
        """
        t0 = time.time()
        data = self.image_collection.get(include=["embeddings", "metadatas"])
        ids = data["ids"]
        embeddings = data["embeddings"]

        if not ids:
            return {"nodes": [], "links": []}

        # Limit total nodes to keep the graph renderable
        if len(ids) > self.MAX_NODES:
            ids = ids[:self.MAX_NODES]
            embeddings = embeddings[:self.MAX_NODES]
            metadatas = (data.get("metadatas") or [])[:self.MAX_NODES]
        else:
            metadatas = data.get("metadatas") or []

        nodes = self._build_nodes(ids, metadatas)
        links = self._build_links(ids, embeddings)

        logger.info(
            "Generated Semantic Graph: %d nodes, %d edges in %.3fs",
            len(nodes),
            len(links),
            time.time() - t0,
        )
        return {"nodes": nodes, "links": links}

    def generate_for_ids(self, ids: List[str]) -> Dict[str, Any]:
        """Build a filtered graph containing only the specified IDs.

        Args:
            ids: Subset of ChromaDB IDs to include in the graph.

        Returns:
            A dict with ``"nodes"`` and ``"links"`` lists.
        """
        if not ids:
            return {"nodes": [], "links": []}

        t0 = time.time()
        data = self.image_collection.get(
            ids=ids, include=["embeddings", "metadatas"]
        )
        result_ids = data["ids"]
        if not result_ids:
            return {"nodes": [], "links": []}

        nodes = self._build_nodes(result_ids, data.get("metadatas") or [])
        links = self._build_links(result_ids, data["embeddings"])

        logger.info(
            "Generated Subgraph: %d nodes, %d edges in %.3fs",
            len(nodes),
            len(links),
            time.time() - t0,
        )
        return {"nodes": nodes, "links": links}

    # Internal

    @staticmethod
    def _build_nodes(ids: List[str], metadatas: List[Any]) -> List[Dict[str, Any]]:
        """Convert ChromaDB records into graph node dicts."""
        nodes = []
        for doc_id, metadata in zip(ids, metadatas):
            try:
                parsed = parse_media_id(doc_id).to_result()
            except ValueError:
                from semantixel.media import describe_local_media

                parsed = describe_local_media(doc_id).to_result()

            nodes.append({
                "id": doc_id,
                **parsed,
                "fileName": os.path.basename(parsed["path"]),
            })
        return nodes

    @staticmethod
    def _build_links(
        ids: List[str], embeddings: List[List[float]]
    ) -> List[Dict[str, Any]]:
        """Compute cosine-similarity edges between all node pairs."""
        if len(ids) < 2:
            return []

        embs_tensor = torch.tensor(embeddings)
        sim_matrix = F.cosine_similarity(
            embs_tensor.unsqueeze(1), embs_tensor.unsqueeze(0), dim=2
        )

        links = []
        seen_edges: set = set()

        sim_matrix.fill_diagonal_(-1.0)
        top_values, top_indices = torch.topk(
            sim_matrix,
            min(GraphService.TOP_K_NEIGHBORS, len(ids) - 1),
            dim=1,
        )

        for i, source_id in enumerate(ids):
            for j in range(top_indices.shape[1]):
                target_idx = top_indices[i, j].item()
                similarity = top_values[i, j].item()
                target_id = ids[target_idx]

                if similarity > GraphService.MIN_SIMILARITY:
                    edge_tuple = tuple(sorted([source_id, target_id]))
                    if edge_tuple not in seen_edges:
                        seen_edges.add(edge_tuple)
                        links.append({
                            "source": source_id,
                            "target": target_id,
                            "value": float(similarity),
                        })

        return links
