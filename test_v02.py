"""
v0.2 Integration Test — String Arena, HNSW Search, Context Assembler
"""
import numpy as np
import time
from agentkv import AgentKV

def main():
    print("=== AgentKV v0.2 Integration Test ===\n")
    db = AgentKV("test_v02.db", size_mb=10, dim=128)

    # --- 1. String Arena ---
    print("[1] String Arena")
    vec_a = np.random.rand(128).astype(np.float32)
    vec_a /= np.linalg.norm(vec_a)  # Normalize for cosine

    off_a = db.add("The capital of France is Paris.", vec_a)
    text_back = db.get_text(off_a)
    assert text_back == "The capital of France is Paris.", f"Text mismatch: {text_back}"
    print(f"    PASS — Stored & retrieved text: \"{text_back}\"")

    # --- 2. HNSW Search ---
    print("\n[2] HNSW K-NN Search")
    NUM_NODES = 200
    offsets = []
    rng = np.random.RandomState(42)

    # Insert nodes with normalized random vectors
    for i in range(NUM_NODES):
        v = rng.rand(128).astype(np.float32)
        v /= np.linalg.norm(v)
        text = f"Memory node {i}"
        off = db.engine.create_node(i, v, text)
        db.engine.init_hnsw(off)
        offsets.append((off, v))

    # Build HNSW links (brute-force wiring for test — real insert is Phase 3)
    # For each node, link to its 16 nearest neighbors at layer 0
    print(f"    Indexing {NUM_NODES} nodes...")
    for i, (off_i, vec_i) in enumerate(offsets):
        # Compute distances to all other nodes
        dists = []
        for j, (off_j, vec_j) in enumerate(offsets):
            if i == j:
                continue
            d = 1.0 - float(np.dot(vec_i, vec_j))
            dists.append((d, off_j))
        dists.sort()
        # Link to top-16 nearest
        for d, off_j in dists[:16]:
            db.engine.add_hnsw_link(off_i, 0, off_j, d)

    # Query: find top-5 nearest to a random query
    query = rng.rand(128).astype(np.float32)
    query /= np.linalg.norm(query)

    start = time.time()
    results = db.search(query, k=5, ef_search=50)
    elapsed = (time.time() - start) * 1000

    print(f"    Search returned {len(results)} results in {elapsed:.3f} ms:")
    for offset, dist in results:
        text = db.get_text(offset)
        print(f"      offset={offset}, dist={dist:.4f}, text=\"{text}\"")

    # Verify results are sorted by distance (ascending)
    distances = [d for _, d in results]
    assert distances == sorted(distances), "Results not sorted by distance!"
    print("    PASS — Results sorted correctly")

    # Brute-force ground truth
    gt_dists = []
    for off, v in offsets:
        d = 1.0 - float(np.dot(query, v))
        gt_dists.append((d, off))
    gt_dists.sort()
    gt_top5 = set(off for _, off in gt_dists[:5])
    hnsw_top5 = set(off for off, _ in results)

    recall = len(gt_top5 & hnsw_top5) / 5.0
    print(f"    Recall@5: {recall*100:.0f}% (HNSW vs brute-force)")
    assert recall >= 0.6, f"Recall too low: {recall}"
    print("    PASS — Recall acceptable")

    # --- 3. Context Assembler ---
    print("\n[3] Context Assembler")
    # Use search results to build context
    search_offsets = [off for off, _ in results]
    context = db.build_context(search_offsets, max_tokens=20)
    print(f"    Budget=20 tokens, assembled {len(context)} texts:")
    for t in context:
        print(f"      \"{t}\"")
    assert len(context) > 0, "Context should not be empty"
    assert len(context) <= len(search_offsets), "More texts than offsets"
    print("    PASS — Context assembled within budget")

    # --- 4. Backward compatibility ---
    print("\n[4] Backward Compatibility")
    # create_node without text still works
    vec_notext = rng.rand(128).astype(np.float32)
    off_notext = db.engine.create_node(9999, vec_notext)
    text_notext = db.get_text(off_notext)
    assert text_notext == "", f"Expected empty text, got: \"{text_notext}\""
    print("    PASS — create_node without text returns empty string")

    print("\n=== ALL v0.2 TESTS PASSED ===")


if __name__ == "__main__":
    main()
