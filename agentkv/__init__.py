import numpy as np
import agentkv_core
from typing import List, Optional, Tuple

class AgentKV:
    """
    The Main Cockpit.
    Manages the Memory-Mapped Engine and the Semantic Lookaside Buffer.
    """
    def __init__(self, db_path: str, size_mb: int = 100):
        # Initialize the C++ Core
        self.engine = agentkv_core.KVEngine(db_path, size_mb * 1024 * 1024)
        self.slb = agentkv_core.ContextManager(self.engine)
        self.dim = 1536 # Default for now

    def add(self, content: str, vector: np.ndarray, relations: List[int] = []) -> int:
        """
        Stores a memory node and links it to related nodes.
        Returns the Node Offset (ID).
        """
        # 1. Create Node
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dim mismatch. Expected {self.dim}, got {vector.shape[0]}")
        
        # hash() can be negative in Python 3; mask to uint64
        node_id = self.engine.create_node(hash(content) & 0xFFFFFFFFFFFFFFFF, vector)
        
        # 2. Link Relations (Graph Edges)
        for target_id in relations:
            # Weight 1.0 for manual links
            self.engine.add_edge(node_id, target_id, 1.0)
            
        return node_id

    def observe(self, node_id: int) -> List[int]:
        """
        The "Active" component. 
        Tells the SLB what the agent is looking at, returns relevant context.
        """
        self.slb.observe(node_id)
        return self.slb.get_context()

    def get_vector(self, node_id: int) -> np.ndarray:
        return self.engine.get_vector(node_id)


# Export core classes
__version__ = "0.1.0"
__all__ = ["AgentKV", "KVEngine", "ContextManager"]

# Expose low-level API for advanced users
KVEngine = agentkv_core.KVEngine
ContextManager = agentkv_core.ContextManager