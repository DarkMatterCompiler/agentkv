from typing import Any, Dict, List, Optional, Sequence
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore, Op, Result
from agentkv import AgentKV
import numpy as np

class AgentKVStore(BaseStore):
    """
    A Zero-Copy Semantic Memory Store for LangGraph.
    """
    def __init__(self, db_path: str):
        self.kv = AgentKV(db_path)
        # In a real app, we'd persist this mapping. 
        # For v0.1, we assume offsets are stable.
        self.namespace_roots = {} 

    def batch(self, ops: Sequence[Op]) -> List[Result]:
        """
        Execute a batch of memory operations (Put, Get, Delete).
        """
        results = []
        for op in ops:
            if op.type == "put":
                # namespace = ["user_id", "memory_type"]
                # key = "memory_id"
                # value = {"text": "...", "vector": [...]}
                
                # 1. Get/Create Namespace Root
                ns_key = tuple(op.namespace)
                if ns_key not in self.namespace_roots:
                    # Create a dummy root for this user/thread
                    dummy_vec = np.zeros(1536, dtype=np.float32)
                    self.namespace_roots[ns_key] = self.kv.add("ROOT", dummy_vec)

                root_id = self.namespace_roots[ns_key]

                # 2. Extract Data
                data = op.value
                if not data or "vector" not in data:
                    results.append(None) # Skip non-vector data
                    continue

                vector = np.array(data["vector"], dtype=np.float32)
                
                # 3. Store in C++ Engine
                # We link it to the Root (temporal) AND any specific relations
                node_id = self.kv.add(data.get("text", ""), vector, relations=[root_id])
                
                # 4. Trigger SLB (Update Context)
                self.kv.observe(node_id)
                
                results.append(None) # Put returns None

            elif op.type == "search":
                # For v0.1, "Search" returns the SLB's Active Context
                # This is a deviation from standard vector search, 
                # but aligns with our "Active Memory" philosophy.
                
                ns_key = tuple(op.namespace)
                root_id = self.namespace_roots.get(ns_key)
                
                if not root_id:
                    results.append([])
                    continue

                # Get context relevant to the LAST observed node in this namespace
                # (In v0.1 we simplify: we just ask SLB for current global context)
                # Ideally, we'd store 'last_accessed' per namespace.
                
                context_offsets = self.kv.slb.get_context()
                
                # Format as LangGraph Results
                search_results = []
                for offset in context_offsets:
                    search_results.append({
                        "key": str(offset),
                        "value": {"offset": offset},
                        "score": 1.0 # Placeholder
                    })
                results.append(search_results)
                
        return results