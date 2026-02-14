"""
v0.3 Integration Test — Dynamic HNSW Insertion (The Autonomy Upgrade)

Inserts 500 nodes using ONLY db.add() — no manual init_hnsw / add_hnsw_link.
Verifies Recall@5 > 0.9 against brute-force ground truth.
"""
import numpy as np
import time
from agentkv import AgentKV


def brute_force_knn(query: np.ndarray, vectors: list, k: int):
    """Brute-force ground truth: returns list of (index, distance) sorted ascending."""
    dists = []
    for i, (offset, vec) in enumerate(vectors):
        d = 1.0 - float(np.dot(query, vec))
        dists.append((d, offset))
    dists.sort()
    return dists[:k]


def main():
    print("=== AgentKV v0.3 — Dynamic HNSW Insert Test ===\n")
    
    NUM_NODES = 500
    DIM = 128
    K = 5
    NUM_QUERIES = 50
    
    db = AgentKV("test_v03.db", size_mb=20, dim=DIM)
    rng = np.random.RandomState(42)

    # --- 1. Insert 500 nodes using ONLY db.add() ---
    print(f"[1] Inserting {NUM_NODES} nodes via db.add() (dynamic HNSW)...")
    all_nodes = []  # (offset, vector)
    
    t_start = time.time()
    for i in range(NUM_NODES):
        vec = rng.rand(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)  # Normalize for cosine distance
        text = f"Memory chunk {i}: context data for testing"
        
        offset = db.add(text, vec)
        all_nodes.append((offset, vec))
    
    t_insert = time.time() - t_start
    avg_insert_us = (t_insert / NUM_NODES) * 1e6
    print(f"    Done. Total: {t_insert*1000:.1f} ms, Avg: {avg_insert_us:.1f} μs/insert")

    # --- 2. Verify text roundtrip ---
    print(f"\n[2] Text roundtrip check...")
    sample_offset = all_nodes[0][0]
    text_back = db.get_text(sample_offset)
    assert text_back == "Memory chunk 0: context data for testing", f"Text mismatch: {text_back}"
    print(f"    PASS — \"{text_back}\"")

    # --- 3. K-NN Search: Recall@5 over 50 random queries ---
    print(f"\n[3] Recall@{K} over {NUM_QUERIES} queries...")
    total_recall = 0.0
    total_search_us = 0.0

    for q_idx in range(NUM_QUERIES):
        query = rng.rand(DIM).astype(np.float32)
        query /= np.linalg.norm(query)

        # HNSW search
        t0 = time.time()
        hnsw_results = db.search(query, k=K, ef_search=100)
        t1 = time.time()
        total_search_us += (t1 - t0) * 1e6

        hnsw_offsets = set(off for off, _ in hnsw_results)

        # Brute-force ground truth
        gt = brute_force_knn(query, all_nodes, K)
        gt_offsets = set(off for _, off in gt)

        recall = len(hnsw_offsets & gt_offsets) / K
        total_recall += recall

    avg_recall = total_recall / NUM_QUERIES
    avg_search_us = total_search_us / NUM_QUERIES
    
    print(f"    Avg Recall@{K}: {avg_recall*100:.1f}%")
    print(f"    Avg Search Latency: {avg_search_us:.1f} μs")
    
    assert avg_recall >= 0.9, f"FAIL: Recall {avg_recall*100:.1f}% < 90% threshold"
    print(f"    PASS — Recall@{K} = {avg_recall*100:.1f}% (≥ 90%)")

    # --- 4. Verify results are sorted ---
    print(f"\n[4] Sort order check...")
    query = rng.rand(DIM).astype(np.float32)
    query /= np.linalg.norm(query)
    results = db.search(query, k=10, ef_search=100)
    distances = [d for _, d in results]
    assert distances == sorted(distances), "Results not sorted ascending!"
    print(f"    PASS — {len(results)} results sorted correctly")

    # --- 5. Context assembly from search results ---
    print(f"\n[5] Context assembly from search results...")
    offsets = [off for off, _ in results]
    context = db.build_context(offsets, max_tokens=50)
    print(f"    Assembled {len(context)} texts within 50-token budget:")
    for t in context[:3]:
        print(f"      \"{t}\"")
    if len(context) > 3:
        print(f"      ... and {len(context)-3} more")
    assert len(context) > 0, "Context should not be empty"
    print(f"    PASS")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"  ALL v0.3 TESTS PASSED")
    print(f"  Nodes: {NUM_NODES} | Recall@{K}: {avg_recall*100:.1f}%")
    print(f"  Insert: {avg_insert_us:.0f} μs/op | Search: {avg_search_us:.0f} μs/op")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
