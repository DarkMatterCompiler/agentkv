"""
Context Assembler — Packs node text into a token-budgeted context window.

Token estimation: chars / 4 (lightweight, no external dependencies).
"""
from typing import List


class ContextBuilder:
    """
    Assembles text from a list of node offsets into a token-budgeted context.

    Usage:
        builder = ContextBuilder(engine)
        texts = builder.build(offsets, max_tokens=2048)
    """
    def __init__(self, engine):
        """
        Args:
            engine: A KVEngine instance (agentkv_core.KVEngine)
        """
        self.engine = engine

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Approximate token count: ~4 chars per token (GPT-family heuristic)."""
        return max(1, len(text) // 4)

    def build(self, node_offsets: List[int], max_tokens: int = 2048) -> List[str]:
        """
        Fetch text from nodes and pack within a token budget.

        Nodes are processed in the order given. Once the budget is exhausted,
        remaining nodes are skipped.

        Args:
            node_offsets: List of node offsets to assemble text from.
            max_tokens: Maximum token budget.

        Returns:
            List of text strings that fit within the budget.
        """
        result: List[str] = []
        tokens_used = 0

        for offset in node_offsets:
            text = self.engine.get_text(offset)
            if not text:
                continue

            cost = self.estimate_tokens(text)
            if tokens_used + cost > max_tokens:
                break  # Budget exhausted

            result.append(text)
            tokens_used += cost

        return result
