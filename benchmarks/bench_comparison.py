#!/usr/bin/env python3
"""
Benchmark: AgentKV vs FAISS vs ChromaDB

Measures insert throughput, search latency, and recall@5 at 10K, 50K, 100K vectors.
All engines use the same random dataset for fair comparison.

Install deps:  pip install agentkv faiss-cpu chromadb numpy
Usage:          python benchmarks/bench_comparison.py
"""
import sys
import os
import time
import gc
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------- Configurable ----------
DIMS = 768
SCALES = [10_000, 50_000, 100_000]
K = 5
EF_SEARCH = 50
NUM_QUERIES = 100
SEED = 42
# ----------------------------------


def generate_data(n, dim, seed):
    """Generate n normalized random vectors and some query vectors."""
    rng = np.random.RandomState(seed)
    data = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data /= norms
    return data


def brute_force_topk(data, queries, k):
    """Ground truth via brute-force dot product."""
    # data: (N, D), queries: (Q, D) -> similarities (Q, N)
    sims = queries @ data.T  # dot product
    # For each query, get top-k indices (highest similarity)
    topk = np.argsort(-sims, axis=1)[:, :k]
    return topk


# ===================== AgentKV =====================
def bench_agentkv(data, queries, ground_truth, k):
    from agentkv import AgentKV

    db_path = "_bench_agentkv.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    n = len(data)
    size_mb = max(100, n * DIMS * 4 // (1024 * 1024) * 4)  # ~4x data size
    db = AgentKV(db_path, size_mb=size_mb, dim=DIMS)

    # Insert
    t0 = time.perf_counter()
    offsets = []
    for i in range(n):
        off = db.add(f"vec_{i}", data[i])
        offsets.append(off)
    t_insert = time.perf_counter() - t0

    # Search
    latencies = []
    recalls = []
    for qi in range(len(queries)):
        t_start = time.perf_counter()
        results = db.search(queries[qi], k=k, ef_search=EF_SEARCH)
        latencies.append(time.perf_counter() - t_start)

        # Map offsets back to indices for recall calculation
        found_offsets = set(r[0] for r in results)
        gt_offsets = set(offsets[idx] for idx in ground_truth[qi])
        recall = len(found_offsets & gt_offsets) / k
        recalls.append(recall)

    del db
    if os.path.exists(db_path):
        os.remove(db_path)

    return {
        "insert_total_s": t_insert,
        "insert_per_vec_us": (t_insert / n) * 1e6,
        "search_avg_us": np.mean(latencies) * 1e6,
        "search_p99_us": np.percentile(latencies, 99) * 1e6,
        "recall_at_k": np.mean(recalls),
    }


# ===================== FAISS =====================
def bench_faiss(data, queries, ground_truth, k):
    try:
        import faiss
    except ImportError:
        return None

    n = len(data)
    dim = data.shape[1]

    # Build HNSW index (M=16 to match AgentKV)
    index = faiss.IndexHNSWFlat(dim, 16)
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = EF_SEARCH

    # Insert
    t0 = time.perf_counter()
    index.add(data)
    t_insert = time.perf_counter() - t0

    # Search
    latencies = []
    recalls = []
    for qi in range(len(queries)):
        t_start = time.perf_counter()
        dists, idxs = index.search(queries[qi:qi+1], k)
        latencies.append(time.perf_counter() - t_start)

        found = set(idxs[0].tolist())
        gt = set(ground_truth[qi].tolist())
        recalls.append(len(found & gt) / k)

    return {
        "insert_total_s": t_insert,
        "insert_per_vec_us": (t_insert / n) * 1e6,
        "search_avg_us": np.mean(latencies) * 1e6,
        "search_p99_us": np.percentile(latencies, 99) * 1e6,
        "recall_at_k": np.mean(recalls),
    }


# ===================== ChromaDB =====================
def bench_chroma(data, queries, ground_truth, k):
    try:
        import chromadb
    except ImportError:
        return None

    chroma_dir = "_bench_chroma"
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.create_collection(
        name="bench",
        metadata={"hnsw:space": "ip"}  # inner product
    )

    n = len(data)

    # Insert (Chroma needs batches)
    BATCH = 5000
    t0 = time.perf_counter()
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        ids = [str(i) for i in range(start, end)]
        embeddings = data[start:end].tolist()
        collection.add(ids=ids, embeddings=embeddings)
    t_insert = time.perf_counter() - t0

    # Search
    latencies = []
    recalls = []
    for qi in range(len(queries)):
        t_start = time.perf_counter()
        results = collection.query(
            query_embeddings=[queries[qi].tolist()],
            n_results=k
        )
        latencies.append(time.perf_counter() - t_start)

        found = set(int(x) for x in results["ids"][0])
        gt = set(ground_truth[qi].tolist())
        recalls.append(len(found & gt) / k)

    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)

    return {
        "insert_total_s": t_insert,
        "insert_per_vec_us": (t_insert / n) * 1e6,
        "search_avg_us": np.mean(latencies) * 1e6,
        "search_p99_us": np.percentile(latencies, 99) * 1e6,
        "recall_at_k": np.mean(recalls),
    }


# ===================== Main =====================
def print_row(engine, stats):
    if stats is None:
        print(f"  {engine:<12} {'(not installed)':>10}")
        return
    print(f"  {engine:<12} "
          f"  {stats['insert_total_s']:>8.2f}s"
          f"  {stats['insert_per_vec_us']:>8.1f} us/vec"
          f"  {stats['search_avg_us']:>8.1f} us"
          f"  {stats['search_p99_us']:>10.1f} us"
          f"  {stats['recall_at_k']:>8.1%}")


def main():
    print("=" * 80)
    print("  AgentKV Benchmark: AgentKV vs FAISS vs ChromaDB")
    print(f"  Dimensions: {DIMS}, K: {K}, Queries: {NUM_QUERIES}")
    print("=" * 80)

    for n in SCALES:
        print(f"\n{'─' * 80}")
        print(f"  N = {n:,} vectors")
        print(f"{'─' * 80}")
        print(f"  {'Engine':<12} {'Insert':>10} {'Insert/vec':>14} "
              f"{'Search avg':>12} {'Search p99':>12} {'Recall@{}'.format(K):>10}")

        # Generate data
        data = generate_data(n, DIMS, SEED)
        query_data = generate_data(NUM_QUERIES, DIMS, SEED + 1)

        # Ground truth
        print("  Computing ground truth (brute force)...")
        gt = brute_force_topk(data, query_data, K)

        gc.collect()

        # Benchmark each engine
        print("  Running AgentKV...")
        akv = bench_agentkv(data, query_data, gt, K)
        print_row("AgentKV", akv)

        gc.collect()

        print("  Running FAISS...")
        faiss_res = bench_faiss(data, query_data, gt, K)
        print_row("FAISS", faiss_res)

        gc.collect()

        # Skip Chroma for 100K (very slow)
        if n <= 50_000:
            print("  Running ChromaDB...")
            chroma_res = bench_chroma(data, query_data, gt, K)
            print_row("ChromaDB", chroma_res)
        else:
            print("  ChromaDB   (skipped at 100K — too slow for batch insert)")

        gc.collect()

    print(f"\n{'=' * 80}")
    print("  Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
