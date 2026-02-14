"""
v0.4 Integration Test — Zero-Alloc Pruning, Persistence, Scale Stress

1. Insert 2,000 nodes via db.add() (dynamic HNSW with smart pruning)
2. Close the DB (munmap + close)
3. Re-open the DB on the same file
4. Verify text integrity and search recall after restart
"""
import numpy as np
import time
import gc
import os
from agentkv import AgentKV

DB_PATH = "test_v04.db"
NUM_NODES = 2000
DIM = 128
K = 5
NUM_QUERIES = 100


def brute_force_knn(query: np.ndarray, vectors: list, k: int):
    """Brute-force ground truth: returns list of (offset, distance) sorted ascending."""
    dists = []
    for offset, vec in vectors:
        d = 1.0 - float(np.dot(query, vec))
        dists.append((offset, d))
    dists.sort(key=lambda x: x[1])
    return dists[:k]


def main():
    print("=== AgentKV v0.4 — Durability & Quality Test ===\n")

    # Clean up any previous test file
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    rng = np.random.RandomState(42)

    # =========================================================================
    # Phase 1: Insert 2,000 nodes
    # =========================================================================
    print(f"[1] Inserting {NUM_NODES} nodes via db.add()...")
    db = AgentKV(DB_PATH, size_mb=50, dim=DIM)
    all_nodes = []  # (offset, vector) — kept in Python for ground truth

    t_start = time.time()
    for i in range(NUM_NODES):
        vec = rng.rand(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        text = f"Node-{i:04d}: persistent memory block"

        offset = db.add(text, vec)
        all_nodes.append((offset, vec))
    t_insert = time.time() - t_start

    avg_us = (t_insert / NUM_NODES) * 1e6
    print(f"    Done. Total: {t_insert*1000:.0f} ms, Avg: {avg_us:.0f} μs/insert")

    # =========================================================================
    # Phase 2: Pre-restart recall baseline
    # =========================================================================
    print(f"\n[2] Pre-restart Recall@{K} over {NUM_QUERIES} queries...")
    total_recall_pre = 0.0
    queries = []  # save for post-restart comparison

    for _ in range(NUM_QUERIES):
        query = rng.rand(DIM).astype(np.float32)
        query /= np.linalg.norm(query)
        queries.append(query)

        hnsw_results = db.search(query, k=K, ef_search=100)
        hnsw_offsets = set(off for off, _ in hnsw_results)

        gt = brute_force_knn(query, all_nodes, K)
        gt_offsets = set(off for off, _ in gt)

        total_recall_pre += len(hnsw_offsets & gt_offsets) / K

    avg_recall_pre = total_recall_pre / NUM_QUERIES
    print(f"    Recall@{K}: {avg_recall_pre*100:.1f}%")
    assert avg_recall_pre >= 0.9, f"FAIL: Pre-restart recall {avg_recall_pre*100:.1f}% < 90%"
    print(f"    PASS")

    # =========================================================================
    # Phase 3: Close DB (simulate process exit)
    # =========================================================================
    print(f"\n[3] Closing DB (munmap + close)...")
    del db
    gc.collect()
    print(f"    DB closed. File persisted at: {DB_PATH}")
    print(f"    File size: {os.path.getsize(DB_PATH) / (1024*1024):.1f} MB")

    # =========================================================================
    # Phase 4: Re-open DB on the same file
    # =========================================================================
    print(f"\n[4] Re-opening DB from disk...")
    db2 = AgentKV(DB_PATH, size_mb=50, dim=DIM)
    print(f"    DB re-opened.")

    # =========================================================================
    # Phase 5: Text integrity after restart
    # =========================================================================
    print(f"\n[5] Text integrity check (sample 20 nodes)...")
    sample_indices = rng.choice(NUM_NODES, size=20, replace=False)
    for idx in sample_indices:
        offset = all_nodes[idx][0]
        expected = f"Node-{idx:04d}: persistent memory block"
        actual = db2.get_text(offset)
        assert actual == expected, f"Text mismatch at node {idx}: '{actual}' != '{expected}'"
    print(f"    PASS — All 20 sampled texts match after restart")

    # =========================================================================
    # Phase 6: Post-restart recall (same queries as Phase 2)
    # =========================================================================
    print(f"\n[6] Post-restart Recall@{K} over {NUM_QUERIES} queries...")
    total_recall_post = 0.0
    total_search_us = 0.0

    for query in queries:
        t0 = time.time()
        hnsw_results = db2.search(query, k=K, ef_search=100)
        t1 = time.time()
        total_search_us += (t1 - t0) * 1e6

        hnsw_offsets = set(off for off, _ in hnsw_results)

        gt = brute_force_knn(query, all_nodes, K)
        gt_offsets = set(off for off, _ in gt)

        total_recall_post += len(hnsw_offsets & gt_offsets) / K

    avg_recall_post = total_recall_post / NUM_QUERIES
    avg_search_us = total_search_us / NUM_QUERIES

    print(f"    Recall@{K}: {avg_recall_post*100:.1f}%")
    print(f"    Avg Search Latency: {avg_search_us:.0f} μs")
    assert avg_recall_post >= 0.9, f"FAIL: Post-restart recall {avg_recall_post*100:.1f}% < 90%"
    print(f"    PASS")

    # =========================================================================
    # Phase 7: Verify recall didn't degrade after restart
    # =========================================================================
    print(f"\n[7] Recall stability check...")
    recall_delta = abs(avg_recall_post - avg_recall_pre)
    print(f"    Pre-restart:  {avg_recall_pre*100:.1f}%")
    print(f"    Post-restart: {avg_recall_post*100:.1f}%")
    print(f"    Delta: {recall_delta*100:.2f}%")
    assert recall_delta < 0.01, f"FAIL: Recall changed by {recall_delta*100:.2f}% after restart"
    print(f"    PASS — Recall identical across restart")

    # Cleanup
    del db2
    gc.collect()

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*55}")
    print(f"  ALL v0.4 TESTS PASSED")
    print(f"  Nodes: {NUM_NODES} | Recall@{K}: {avg_recall_post*100:.1f}%")
    print(f"  Insert: {avg_us:.0f} μs/op | Search: {avg_search_us:.0f} μs/op")
    print(f"  Persistence: VERIFIED (close/reopen, zero data loss)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
