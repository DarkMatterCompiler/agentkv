import numpy as np
import agentkv_core
from typing import List, Optional, Tuple

from .context import ContextBuilder


class AgentKV:
    """
    The Main Cockpit — v0.6
    Manages the Memory-Mapped Engine, Semantic Lookaside Buffer,
    HNSW Vector Index (dynamic insert), String Arena, and Context Assembly.
    """
    def __init__(self, db_path: str, size_mb: int = 100, dim: int = 1536):
        # Initialize the C++ Core
        self.engine = agentkv_core.KVEngine(db_path, size_mb * 1024 * 1024)
        self.slb = agentkv_core.ContextManager(self.engine)
        self.ctx = ContextBuilder(self.engine)
        self.dim = dim

    def add(self, content: str, vector: np.ndarray,
            relations: Optional[List[int]] = None) -> int:
        """
        Stores a memory node with text + vector and auto-indexes into HNSW.
        Optionally links semantic graph edges to related nodes.
        Returns the Node Offset.
        """
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"Vector dim mismatch. Expected {self.dim}, got {vector.shape[0]}"
            )

        # Dynamic HNSW insertion: creates node + wires bidirectional links
        node_offset = self.engine.insert(
            hash(content) & 0xFFFFFFFFFFFFFFFF, vector, content
        )

        # Link semantic graph edges (separate from HNSW)
        if relations:
            for target_id in relations:
                self.engine.add_edge(node_offset, target_id, 1.0)

        return node_offset

    def search(self, query_vector: np.ndarray, k: int = 5,
               ef_search: int = 50) -> List[Tuple[int, float]]:
        """
        K-NN vector search via HNSW index.
        Returns list of (node_offset, distance) sorted ascending by distance.
        Requires nodes to have been indexed with init_hnsw / add_hnsw_link.
        """
        return self.engine.search_knn(query_vector, k, ef_search)

    def observe(self, node_id: int) -> List[int]:
        """
        The "Active" component.
        Tells the SLB what the agent is looking at, returns relevant context.
        """
        self.slb.observe(node_id)
        return self.slb.get_context()

    def get_vector(self, node_id: int) -> np.ndarray:
        """Get the vector embedding of a node (zero-copy from mmap)."""
        return self.engine.get_vector(node_id)

    def get_text(self, node_id: int) -> str:
        """Get the text content of a node."""
        return self.engine.get_text(node_id)

    def build_context(self, node_offsets: List[int],
                      max_tokens: int = 2048) -> List[str]:
        """
        Assemble text from node offsets into a token-budgeted context window.
        Nodes are processed in order; stops when budget is exhausted.
        """
        return self.ctx.build(node_offsets, max_tokens)

    # --- HNSW manual wiring (Phase 2 will auto-insert) ---
    def init_hnsw(self, node_offset: int) -> None:
        """Initialize HNSW index data for a node (assigns a random level)."""
        self.engine.init_hnsw(node_offset)

    def add_hnsw_link(self, node_offset: int, layer: int,
                      target_offset: int, dist: float) -> None:
        """Add an HNSW connection at the given layer (with M_max pruning)."""
        self.engine.add_hnsw_link(node_offset, layer, target_offset, dist)


# Export core classes
__version__ = "0.7.1"
__all__ = [
    "AgentKV", "ContextBuilder", "KVEngine", "ContextManager",
]

# Expose low-level API for advanced users
KVEngine = agentkv_core.KVEngine
ContextManager = agentkv_core.ContextManager