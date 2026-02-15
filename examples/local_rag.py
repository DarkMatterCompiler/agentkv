#!/usr/bin/env python3
"""
Example: Local RAG with AgentKV + Ollama

Stores facts with real embeddings, then answers questions via semantic search.
Requires: pip install agentkv[ollama]
          ollama pull nomic-embed-text
"""
import sys
import os

# Ensure the local package is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentkv import AgentKV
from agentkv.ollama import get_embedding, is_available

DB_PATH = "rag_example.db"
DIM = 768  # nomic-embed-text

FACTS = [
    "The speed of light is approximately 299,792 km/s.",
    "Python was created by Guido van Rossum and first released in 1991.",
    "The human brain contains roughly 86 billion neurons.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Great Wall of China is over 13,000 miles long.",
    "DNA stands for deoxyribonucleic acid.",
    "Jupiter is the largest planet in our solar system.",
    "The Eiffel Tower was completed in 1889.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The Amazon River is the largest river by volume in the world.",
]

QUERIES = [
    "How fast does light travel?",
    "Tell me about the history of Python programming.",
    "What is the biggest planet?",
]


def main():
    print("=" * 50)
    print("  AgentKV Local RAG Example")
    print("=" * 50)

    if not is_available():
        print("ERROR: Ollama is not running. Start with: ollama serve")
        sys.exit(1)

    # Clean start
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    db = AgentKV(DB_PATH, size_mb=50, dim=DIM)

    # 1. Embed and store facts
    print(f"\nStoring {len(FACTS)} facts...")
    for i, fact in enumerate(FACTS):
        vec = get_embedding(fact)
        db.add(fact, vec)
        print(f"  [{i+1}/{len(FACTS)}] {fact[:60]}...")

    # 2. Query
    print(f"\n{'─' * 50}")
    print("Querying...\n")
    for query in QUERIES:
        q_vec = get_embedding(query)
        results = db.search(q_vec, k=3)
        print(f"Q: {query}")
        for rank, (offset, dist) in enumerate(results, 1):
            text = db.get_text(offset)
            print(f"  #{rank} [{dist:.4f}] {text}")
        print()

    # Cleanup
    os.remove(DB_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
