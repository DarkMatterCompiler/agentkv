import numpy as np
import agentkv_core
from typing import Dict, List, Optional, Tuple

from .context import ContextBuilder

# Re-export distance metric enum
COSINE = agentkv_core.DistanceMetric.COSINE
L2 = agentkv_core.DistanceMetric.L2
INNER_PRODUCT = agentkv_core.DistanceMetric.INNER_PRODUCT


class AgentKV:
    """
    AgentKV — v0.9
    Memory-Mapped Vector + Graph Database with:
      - HNSW index (SIMD-accelerated, multi-metric)
      - Metadata filtering (key=value tags)
      - Batch insert, delete/update, count/iteration
      - Crash recovery (CRC-32 checksum)
      - Semantic Lookaside Buffer (SLB) for context prediction
    """
    def __init__(self, db_path: str, size_mb: int = 100, dim: int = 1536,
                 metric: str = "cosine"):
        """
        Args:
            db_path: Path to the database file (created if not exists).
            size_mb:  Maximum database size in MB.
            dim:      Vector dimensionality.
            metric:   Distance metric: "cosine", "l2", or "ip" (inner product).
        """
        metric_map = {
            "cosine": agentkv_core.DistanceMetric.COSINE,
            "l2": agentkv_core.DistanceMetric.L2,
            "ip": agentkv_core.DistanceMetric.INNER_PRODUCT,
        }
        m = metric_map.get(metric)
        if m is None:
            raise ValueError(
                f"Unknown metric '{metric}'. Use 'cosine', 'l2', or 'ip'."
            )
        self.engine = agentkv_core.KVEngine(db_path, size_mb * 1024 * 1024, m)
        self.slb = agentkv_core.ContextManager(self.engine)
        self.ctx = ContextBuilder(self.engine)
        self.dim = dim

    # ── Insert ────────────────────────────────────────────────────────────

    def add(self, content: str, vector: np.ndarray,
            relations: Optional[List[int]] = None,
            metadata: Optional[Dict[str, str]] = None) -> int:
        """
        Insert a single node with text + vector.  Auto-indexes into HNSW.
        Optionally links semantic graph edges and sets metadata tags.
        Returns the node offset.
        """
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"Vector dim mismatch. Expected {self.dim}, got {vector.shape[0]}"
            )
        node_offset = self.engine.insert(
            hash(content) & 0xFFFFFFFFFFFFFFFF, vector, content
        )
        if relations:
            for target_id in relations:
                self.engine.add_edge(node_offset, target_id, 1.0)
        if metadata:
            for k, v in metadata.items():
                self.engine.set_metadata(node_offset, str(k), str(v))
        return node_offset

    def add_batch(self, contents: List[str], vectors: np.ndarray,
                  metadatas: Optional[List[Dict[str, str]]] = None
                  ) -> List[int]:
        """
        Batch insert N vectors in a single C++ call (one lock acquisition).
        Typically 3-5x faster than N sequential add() calls.

        Args:
            contents: List of N text strings.
            vectors:  numpy array of shape (N, dim), dtype float32.
            metadatas: Optional list of N dicts with string key=value pairs.
        Returns:
            List of N node offsets.
        """
        n = len(contents)
        if vectors.shape != (n, self.dim):
            raise ValueError(
                f"Expected vectors shape ({n}, {self.dim}), "
                f"got {vectors.shape}"
            )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = np.array(
            [hash(c) & 0xFFFFFFFFFFFFFFFF for c in contents],
            dtype=np.uint64
        )
        offsets = self.engine.insert_batch(ids, vectors, contents)
        if metadatas:
            for offset, meta in zip(offsets, metadatas):
                if meta:
                    for k, v in meta.items():
                        self.engine.set_metadata(offset, str(k), str(v))
        return offsets

    # ── Search ────────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, k: int = 5,
               ef_search: int = 50,
               where: Optional[Dict[str, str]] = None
               ) -> List[Tuple[int, float]]:
        """
        K-NN vector search via HNSW index.
        Optionally filter results by metadata: where={"key": "value"}.
        Returns list of (node_offset, distance) sorted ascending.
        """
        if where:
            filters = [(str(k_), str(v_)) for k_, v_ in where.items()]
            return self.engine.search_knn_filtered(
                query_vector, k, ef_search, filters
            )
        return self.engine.search_knn(query_vector, k, ef_search)

    # ── Delete / Update ───────────────────────────────────────────────────

    def delete(self, node_offset: int) -> None:
        """Tombstone a node. It will be excluded from search and iteration."""
        self.engine.delete_node(node_offset)

    def update(self, old_offset: int, content: str,
               vector: np.ndarray,
               metadata: Optional[Dict[str, str]] = None) -> int:
        """
        Update = tombstone old + insert new.  Returns the new node offset.
        The old offset becomes invalid.
        """
        new_id = hash(content) & 0xFFFFFFFFFFFFFFFF
        new_offset = self.engine.update_node(old_offset, new_id, vector, content)
        if metadata:
            for k, v in metadata.items():
                self.engine.set_metadata(new_offset, str(k), str(v))
        return new_offset

    # ── Metadata ──────────────────────────────────────────────────────────

    def set_metadata(self, node_offset: int, key: str, value: str) -> None:
        """Set a metadata tag on a node."""
        self.engine.set_metadata(node_offset, key, value)

    def get_metadata(self, node_offset: int, key: str) -> str:
        """Get a metadata value by key (empty string if not found)."""
        return self.engine.get_metadata(node_offset, key)

    def get_all_metadata(self, node_offset: int) -> Dict[str, str]:
        """Get all metadata as a dict."""
        return dict(self.engine.get_all_metadata(node_offset))

    # ── Count / Iteration ─────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of live (non-deleted) nodes."""
        return self.engine.count()

    def count(self) -> int:
        """Number of live (non-deleted) nodes."""
        return self.engine.count()

    def keys(self) -> List[int]:
        """Get all live node offsets."""
        return self.engine.get_all_node_offsets()

    def items(self) -> List[Tuple[int, str]]:
        """Get all live (offset, text) pairs."""
        offsets = self.engine.get_all_node_offsets()
        return [(off, self.engine.get_text(off)) for off in offsets]

    # ── Accessors ─────────────────────────────────────────────────────────

    def get_vector(self, node_offset: int) -> np.ndarray:
        """Get the vector embedding of a node (zero-copy from mmap)."""
        return self.engine.get_vector(node_offset)

    def get_text(self, node_offset: int) -> str:
        """Get the text content of a node."""
        return self.engine.get_text(node_offset)

    @property
    def metric(self) -> str:
        """The distance metric this DB uses."""
        m = self.engine.get_metric()
        return {0: "cosine", 1: "l2", 2: "ip"}.get(m, "unknown")

    # ── Context (SLB) ────────────────────────────────────────────────────

    def observe(self, node_id: int) -> List[int]:
        """Tell the SLB what the agent is looking at, get predicted context."""
        self.slb.observe(node_id)
        return self.slb.get_context()

    def build_context(self, node_offsets: List[int],
                      max_tokens: int = 2048) -> List[str]:
        """Assemble text from node offsets into a token-budgeted context."""
        return self.ctx.build(node_offsets, max_tokens)

    # ── Low-level HNSW (backward compat) ─────────────────────────────────

    def init_hnsw(self, node_offset: int) -> None:
        self.engine.init_hnsw(node_offset)

    def add_hnsw_link(self, node_offset: int, layer: int,
                      target_offset: int, dist: float) -> None:
        self.engine.add_hnsw_link(node_offset, layer, target_offset, dist)


# Export core classes
__version__ = "0.9.1"
__all__ = [
    "AgentKV", "ContextBuilder", "KVEngine", "ContextManager",
    "COSINE", "L2", "INNER_PRODUCT",
]

# Expose low-level API for advanced users
KVEngine = agentkv_core.KVEngine
ContextManager = agentkv_core.ContextManager
DistanceMetric = agentkv_core.DistanceMetric