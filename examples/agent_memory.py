#!/usr/bin/env python3
"""
Example: Persistent Agent Memory with AgentKV

Demonstrates that memories survive process restarts (mmap persistence).
No Ollama required — uses random vectors for simplicity.

Usage: python examples/agent_memory.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentkv import AgentKV

DB_PATH = "agent_memory_example.db"
DIM = 128  # Small dim for this demo


def make_vec(seed: int) -> np.ndarray:
    """Create a deterministic, normalized vector from a seed."""
    rng = np.random.RandomState(seed)
    v = rng.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def main():
    print("=" * 50)
    print("  AgentKV Agent Memory Example")
    print("=" * 50)

    # Clean start
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # --- SESSION 1: Store memories ---
    print("\n[Session 1] Creating agent and storing memories...")
    db = AgentKV(DB_PATH, size_mb=10, dim=DIM)

    memories = [
        ("User prefers dark mode", 42),
        ("User's name is Alice", 43),
        ("User works at Acme Corp", 44),
        ("User likes Python and Rust", 45),
        ("Last meeting was about Q4 roadmap", 46),
    ]

    offsets = []
    for text, seed in memories:
        vec = make_vec(seed)
        off = db.add(text, vec)
        offsets.append(off)
        print(f"  Stored: \"{text}\" -> offset {off}")

    # Validate header
    print(f"\n  Header valid: {db.engine.is_valid()}")

    # Explicitly delete to trigger clean shutdown
    del db
    print("  [Session 1 closed]")

    # --- SESSION 2: Reopen and recall ---
    print("\n[Session 2] Reopening database (simulating restart)...")
    db2 = AgentKV(DB_PATH, size_mb=10, dim=DIM)

    print(f"  Header valid after reopen: {db2.engine.is_valid()}")

    # Search for "user preferences"
    query_vec = make_vec(42)  # Same seed as "User prefers dark mode"
    results = db2.search(query_vec, k=3)

    print(f"\n  Searching for memories similar to 'dark mode preference':")
    for rank, (offset, dist) in enumerate(results, 1):
        text = db2.get_text(offset)
        print(f"    #{rank} [{dist:.4f}] {text}")

    # Verify the top result is correct
    top_text = db2.get_text(results[0][0])
    assert "dark mode" in top_text, f"Expected dark mode memory, got: {top_text}"
    print("\n  PASS: Correct memory recalled after restart!")

    # --- SESSION 3: Add more and verify graph edges ---
    print("\n[Session 3] Adding more memories with relations...")
    new_vec = make_vec(100)
    new_off = db2.add("Q4 roadmap includes AgentKV launch", new_vec,
                       relations=[offsets[4]])  # Link to "Q4 roadmap" memory
    print(f"  Stored new memory -> offset {new_off}")
    print(f"  Linked to offset {offsets[4]} (Q4 roadmap)")

    # Observe and predict context
    context = db2.observe(new_off)
    print(f"  SLB predicted context: {len(context)} nodes")

    del db2

    # Cleanup
    os.remove(DB_PATH)
    print("\nAll sessions completed successfully!")


if __name__ == "__main__":
    main()
