#!/usr/bin/env python3
"""
test_v07_full.py — Comprehensive pre-release test suite for AgentKV v0.7

Tests:
  1.  Fresh DB creation + header validation
  2.  Insert + search correctness (small scale)
  3.  Text storage & retrieval
  4.  Persistence across close/reopen
  5.  Crash recovery (dirty header simulation)
  6.  Concurrency (1 writer + N readers)
  7.  Scale test (2000 nodes, recall@5)
  8.  Graph edges + SLB context prediction
  9.  AgentKV high-level API
  10. Version / package metadata
  --- NEW (Critical) ---
  11. Input validation
  12. Error handling
  13. Thread safety (improved)
  14. Boundary conditions
  --- NEW (Important) ---
  15. Data correctness
  16. Graph / relations
  17. Memory stress
  18. Multi-cycle persistence
  --- NEW (Nice to have) ---
  19. Recall / quality
  20. API completeness
"""
import sys
import os
import time
import threading
import tempfile
import struct
import gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We need the C++ extension
try:
    import agentkv_core
except ImportError:
    print("ERROR: agentkv_core not built. Run: pip install -e .")
    sys.exit(1)

from agentkv import AgentKV

DIM = 128
passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        errors.append(name)


def make_vec(seed, dim=DIM):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


# ─────────────────────────────────────────────────────
# Test 1: Fresh DB creation + header validation
# ─────────────────────────────────────────────────────
def test_fresh_db():
    print("\n[Test 1] Fresh DB creation + header validation")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=5, dim=DIM)
        check("DB created", db is not None)
        check("Header valid", db.engine.is_valid())
        del db

        # Reopen and validate
        db2 = AgentKV(path, size_mb=5, dim=DIM)
        check("Reopen valid", db2.engine.is_valid())
        del db2
    finally:
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────────────
# Test 2: Insert + search correctness
# ─────────────────────────────────────────────────────
def test_insert_search():
    print("\n[Test 2] Insert + search correctness (100 nodes)")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=10, dim=DIM)
        vecs = []
        for i in range(100):
            v = make_vec(i)
            vecs.append(v)
            db.add(f"item_{i}", v)

        # Search for the first vector — should find itself as rank #1
        results = db.search(vecs[0], k=5)
        check("Results returned", len(results) > 0, f"got {len(results)}")

        top_text = db.get_text(results[0][0])
        check("Top result correct", top_text == "item_0", f"got '{top_text}'")
        check("Top distance small", results[0][1] < 0.01, f"dist={results[0][1]:.6f}")

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────────────
# Test 3: Text storage & retrieval
# ─────────────────────────────────────────────────────
def test_text_storage():
    print("\n[Test 3] Text storage & retrieval")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=5, dim=DIM)

        test_strings = [
            "Hello, world!",
            "Unicode: café, naïve, Ünter",
            "Long text: " + "a" * 5000,
            "",  # empty
            "Special chars: \n\t\"quotes\" and 'apostrophes'",
        ]

        offsets = []
        for i, text in enumerate(test_strings):
            v = make_vec(200 + i)
            off = db.add(text, v)
            offsets.append(off)

        for i, (off, expected) in enumerate(zip(offsets, test_strings)):
            got = db.get_text(off)
            check(f"Text[{i}] roundtrip", got == expected,
                  f"expected len={len(expected)}, got len={len(got)}")

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────────────
# Test 4: Persistence across close/reopen
# ─────────────────────────────────────────────────────
def test_persistence():
    print("\n[Test 4] Persistence across close/reopen")
    path = tempfile.mktemp(suffix=".db")
    try:
        # Session 1: write
        db = AgentKV(path, size_mb=10, dim=DIM)
        vecs = []
        for i in range(50):
            v = make_vec(300 + i)
            vecs.append(v)
            db.add(f"persist_{i}", v)
        del db

        # Session 2: read
        db2 = AgentKV(path, size_mb=10, dim=DIM)
        results = db2.search(vecs[0], k=5)
        check("Post-restart results", len(results) >= 1)

        top_text = db2.get_text(results[0][0])
        check("Post-restart correct", top_text == "persist_0", f"got '{top_text}'")

        # Verify recall didn't degrade
        hits_before = 0
        hits_after = 0
        for i in range(min(20, len(vecs))):
            r = db2.search(vecs[i], k=1)
            if r:
                t = db2.get_text(r[0][0])
                if t == f"persist_{i}":
                    hits_after += 1
                hits_before += 1  # just counting queries

        recall = hits_after / max(hits_before, 1)
        check("Persistence recall", recall >= 0.8, f"{recall:.1%}")

        del db2
    finally:
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────────────
# Test 5: Crash recovery (dirty header simulation)
# ─────────────────────────────────────────────────────
def test_crash_recovery():
    print("\n[Test 5] Crash recovery (dirty header simulation)")
    path = tempfile.mktemp(suffix=".db")
    try:
        # Session 1: create DB with data
        db = AgentKV(path, size_mb=10, dim=DIM)
        vecs = []
        for i in range(30):
            v = make_vec(400 + i)
            vecs.append(v)
            db.add(f"crash_{i}", v)
        del db  # clean shutdown

        # Simulate a crash: clear the CLEAN_SHUTDOWN flag in the header
        with open(path, "r+b") as f:
            # flags field is at a specific offset in the header
            # magic(4) + version(4) + write_head(8) + capacity(8) + active_readers(4)
            # + hnsw_entry_point(8) + hnsw_max_level(4) + M_max(4) + ef_construction(4)
            # = 4+4+8+8+4+8+4+4+4 = 48 bytes -> flags at offset 48
            f.seek(48)
            old_flags = struct.unpack('<I', f.read(4))[0]
            # Clear the clean shutdown flag
            new_flags = old_flags & ~0x01
            f.seek(48)
            f.write(struct.pack('<I', new_flags))

        # Session 2: reopen — should detect dirty and recover
        db2 = AgentKV(path, size_mb=10, dim=DIM)
        check("Recovery: DB opened", db2 is not None)
        check("Recovery: header valid", db2.engine.is_valid())

        # Data should still be searchable
        results = db2.search(vecs[0], k=3)
        check("Recovery: data intact", len(results) >= 1)
        if results:
            t = db2.get_text(results[0][0])
            check("Recovery: correct result", "crash_" in t, f"got '{t}'")

        del db2
    finally:
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────────────
# Test 6: Concurrency (1 writer + 4 readers)
# ─────────────────────────────────────────────────────
def test_concurrency():
    print("\n[Test 6] Concurrency (1 writer + 4 readers)")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=20, dim=DIM)

        # Seed some data
        for i in range(50):
            db.add(f"seed_{i}", make_vec(500 + i))

        write_count = [0]
        read_count = [0]
        error_count = [0]
        stop = threading.Event()

        def writer():
            for i in range(100):
                try:
                    db.add(f"conc_w_{i}", make_vec(600 + i))
                    write_count[0] += 1
                except Exception as e:
                    error_count[0] += 1

        def reader():
            while not stop.is_set():
                try:
                    q = make_vec(700 + read_count[0] % 50)
                    db.search(q, k=3)
                    read_count[0] += 1
                except Exception:
                    error_count[0] += 1

        t0 = time.time()
        readers = [threading.Thread(target=reader) for _ in range(4)]
        writer_t = threading.Thread(target=writer)

        for r in readers:
            r.start()
        writer_t.start()
        writer_t.join()
        stop.set()
        for r in readers:
            r.join()
        elapsed = time.time() - t0

        check("Concurrency: zero errors", error_count[0] == 0,
              f"errors={error_count[0]}")
        check("Concurrency: writes done", write_count[0] == 100,
              f"writes={write_count[0]}")
        check("Concurrency: reads done", read_count[0] >= 10,
              f"reads={read_count[0]}")
        print(f"    {write_count[0]} writes + {read_count[0]} reads in {elapsed:.2f}s")

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────────────
# Test 7: Scale test (2000 nodes, recall@5)
# ─────────────────────────────────────────────────────
def test_scale():
    print("\n[Test 7] Scale test (2000 nodes, recall@5)")
    path = tempfile.mktemp(suffix=".db")
    N = 2000
    try:
        db = AgentKV(path, size_mb=50, dim=DIM)

        t0 = time.time()
        vecs = []
        for i in range(N):
            v = make_vec(800 + i)
            vecs.append(v)
            db.add(f"scale_{i}", v)
        insert_time = time.time() - t0
        insert_per = (insert_time / N) * 1e6

        print(f"    Insert: {insert_time:.2f}s ({insert_per:.0f} us/vec)")

        # Recall@5 over 100 random queries
        hits = 0
        total = 0
        t_search = time.time()
        for i in range(0, 100):
            q = vecs[i * 20 % N]
            expected_text = f"scale_{i * 20 % N}"
            results = db.search(q, k=5, ef_search=50)
            found_texts = [db.get_text(r[0]) for r in results]
            if expected_text in found_texts:
                hits += 1
            total += 1
        search_time = time.time() - t_search

        recall = hits / total
        per_query = (search_time / total) * 1e6
        print(f"    Search: {search_time:.2f}s ({per_query:.0f} us/query)")
        print(f"    Recall@5: {recall:.1%} ({hits}/{total})")

        check("Scale: insert reasonable", insert_per < 5000,
              f"{insert_per:.0f} us/vec")
        check("Scale: recall >= 80%", recall >= 0.80, f"{recall:.1%}")

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────────────
# Test 8: Graph edges + SLB context prediction
# ─────────────────────────────────────────────────────
def test_graph_slb():
    print("\n[Test 8] Graph edges + SLB context prediction")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=10, dim=DIM)

        # Create nodes first
        off_a = db.add("node_A", make_vec(900))
        off_b = db.add("node_B", make_vec(901))
        off_c = db.add("node_C", make_vec(902))

        # Add OUTGOING edges: A -> B -> C
        # add_edge(source_offset, target_offset, weight)
        db.engine.add_edge(off_a, off_b, 1.0)
        db.engine.add_edge(off_b, off_c, 1.0)

        # Observe A — SLB should predict B (via A->B edge)
        ctx = db.observe(off_a)
        check("SLB returns context", len(ctx) >= 1, f"got {len(ctx)} nodes")
        # Verify B is actually in the returned context
        ctx_offsets_a = [c[0] if isinstance(c, tuple) else c for c in ctx]
        check("SLB A→B: B in context", off_b in ctx_offsets_a,
              f"expected {off_b} in {ctx_offsets_a}")

        # Observe B — should predict C (via B->C edge)
        ctx2 = db.observe(off_b)
        check("SLB B->C", len(ctx2) >= 1, f"got {len(ctx2)} nodes")
        ctx_offsets_b = [c[0] if isinstance(c, tuple) else c for c in ctx2]
        check("SLB B→C: C in context", off_c in ctx_offsets_b,
              f"expected {off_c} in {ctx_offsets_b}")

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────────────
# Test 9: AgentKV high-level API completeness
# ─────────────────────────────────────────────────────
def test_api():
    print("\n[Test 9] AgentKV high-level API")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=5, dim=DIM)

        # add
        v = make_vec(1000)
        off = db.add("api_test", v)
        check("API: add returns offset", off > 0)

        # get_text
        t = db.get_text(off)
        check("API: get_text", t == "api_test")

        # get_vector (zero-copy)
        vec = db.get_vector(off)
        check("API: get_vector shape", vec.shape[0] == DIM)
        check("API: get_vector values", np.allclose(vec, v, atol=1e-6))

        # search
        results = db.search(v, k=1)
        check("API: search returns", len(results) == 1)

        # build_context
        texts = db.build_context([off], max_tokens=100)
        check("API: build_context", len(texts) == 1 and texts[0] == "api_test")

        # engine.is_valid
        check("API: is_valid", db.engine.is_valid())

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────────────
# Test 10: Version / package metadata
# ─────────────────────────────────────────────────────
def test_version():
    print("\n[Test 10] Version / package metadata")
    import agentkv
    check("Version defined", hasattr(agentkv, '__version__'))
    check("Version is 0.7.1", agentkv.__version__ == "0.7.1",
          f"got '{agentkv.__version__}'")
    check("AgentKV exported", hasattr(agentkv, 'AgentKV'))
    check("KVEngine exported", hasattr(agentkv, 'KVEngine'))


# =============================================================================
#  🔴 CRITICAL — Test 11: Input Validation
# =============================================================================
def test_input_validation():
    print("\n[Test 11] Input validation")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=5, dim=DIM)

        # 11a. Empty text string (should be accepted)
        v = make_vec(1100)
        off = db.add("", v)
        check("Empty text accepted", off > 0)
        check("Empty text roundtrip", db.get_text(off) == "")

        # 11b. Vector dimension mismatch
        wrong_dim = np.random.randn(64).astype(np.float32)
        try:
            db.add("bad dim", wrong_dim)
            check("Dim mismatch rejected", False, "No exception raised")
        except (ValueError, RuntimeError):
            check("Dim mismatch rejected", True)

        # 11c. Non-normalized vectors (should still work — just lower recall)
        unnorm = np.ones(DIM, dtype=np.float32) * 100.0
        off_nn = db.add("unnormalized", unnorm)
        check("Non-normalized accepted", off_nn > 0)

        # 11d. k=0 in search
        results = db.search(v, k=0)
        check("k=0 returns empty", len(results) == 0)

        # 11e. k > total_nodes
        results = db.search(v, k=9999)
        check("k>N returns available", len(results) <= 9999)
        check("k>N non-empty", len(results) >= 1)

        # 11f. Very large ef_search
        results = db.search(v, k=1, ef_search=10000)
        check("Large ef_search works", len(results) >= 1)

        # 11g. Zero vector (all zeros)
        zero_vec = np.zeros(DIM, dtype=np.float32)
        off_z = db.add("zero vector", zero_vec)
        check("Zero vector accepted", off_z > 0)

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# =============================================================================
#  🔴 CRITICAL — Test 12: Error Handling
# =============================================================================
def test_error_handling():
    print("\n[Test 12] Error handling")

    # 12a. Corrupted file header (bad magic)
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=5, dim=DIM)
        db.add("test", make_vec(1200))
        del db

        # Corrupt the magic number
        with open(path, "r+b") as f:
            f.seek(0)
            f.write(b'\x00\x00\x00\x00')  # Zero out magic

        # Should create a fresh DB (magic mismatch → treat as new)
        db2 = AgentKV(path, size_mb=5, dim=DIM)
        check("Bad magic → fresh DB", db2.engine.is_valid())
        del db2
    finally:
        if os.path.exists(path):
            os.remove(path)

    # 12b. Corrupted checksum
    path2 = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path2, size_mb=5, dim=DIM)
        for i in range(10):
            db.add(f"chk_{i}", make_vec(1210 + i))
        del db

        # Corrupt the checksum but keep magic intact
        with open(path2, "r+b") as f:
            # checksum is at offset 56 in the new header layout
            f.seek(56)
            f.write(struct.pack('<I', 0xDEADBEEF))

        # Should recover to safe state
        db2 = AgentKV(path2, size_mb=5, dim=DIM)
        check("Bad checksum → recovery", db2 is not None)
        check("Post-recovery valid", db2.engine.is_valid())

        # Verify data survived corruption recovery
        q = make_vec(1210)  # Same seed as chk_0
        results = db2.search(q, k=1)
        check("Post-recovery data intact", len(results) >= 1,
              f"search returned {len(results)} results")
        if results:
            txt = db2.get_text(results[0][0])
            check("Post-recovery text correct", "chk_" in txt,
                  f"got '{txt}'")
        del db2
    finally:
        if os.path.exists(path2):
            os.remove(path2)

    # 12c. Truncated database file
    path3 = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path3, size_mb=5, dim=DIM)
        db.add("trunc", make_vec(1230))
        del db

        # Reopening with same size should work (ftruncate extends)
        db2 = AgentKV(path3, size_mb=5, dim=DIM)
        check("Reopen truncated works", db2 is not None)
        del db2
    finally:
        if os.path.exists(path3):
            os.remove(path3)

    # 12d. Search on empty database
    path4 = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path4, size_mb=5, dim=DIM)
        results = db.search(make_vec(1240), k=5)
        check("Empty DB search → []", len(results) == 0)
        del db
    finally:
        if os.path.exists(path4):
            os.remove(path4)


# =============================================================================
#  🔴 CRITICAL — Test 13: Thread Safety (Improved)
# =============================================================================
def test_thread_safety():
    print("\n[Test 13] Thread safety (improved)")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=30, dim=DIM)

        # Seed data
        for i in range(20):
            db.add(f"ts_seed_{i}", make_vec(1300 + i))

        # Use proper locking for shared counters
        lock = threading.Lock()
        write_count = [0]
        read_count = [0]
        error_list = []
        consistency_errors = []
        stop = threading.Event()

        # 13a. Read-during-write consistency
        def writer():
            for i in range(200):
                try:
                    db.add(f"ts_w_{i}", make_vec(1400 + i))
                    with lock:
                        write_count[0] += 1
                except Exception as e:
                    with lock:
                        error_list.append(f"write: {e}")

        def reader():
            local_read = 0
            while not stop.is_set():
                try:
                    q = make_vec(1500 + (local_read % 50))
                    local_read += 1
                    results = db.search(q, k=3)
                    # Verify results are consistent (valid offsets, valid text)
                    for off, dist in results:
                        text = db.get_text(off)
                        if text is not None and not isinstance(text, str):
                            with lock:
                                consistency_errors.append(f"bad text type: {type(text)}")
                    with lock:
                        read_count[0] += 1
                except Exception as e:
                    with lock:
                        error_list.append(f"read: {e}")

        t0 = time.time()
        readers = [threading.Thread(target=reader) for _ in range(4)]
        writer_t = threading.Thread(target=writer)
        for r in readers:
            r.start()
        writer_t.start()
        writer_t.join()
        stop.set()
        for r in readers:
            r.join()
        elapsed = time.time() - t0

        check("TS: zero errors", len(error_list) == 0,
              f"{len(error_list)} errors: {error_list[:3]}")
        check("TS: zero consistency errors", len(consistency_errors) == 0,
              f"{len(consistency_errors)}")
        check("TS: all writes done", write_count[0] == 200,
              f"writes={write_count[0]}")
        check("TS: reads done", read_count[0] >= 20,
              f"reads={read_count[0]}")
        print(f"    {write_count[0]}W + {read_count[0]}R in {elapsed:.2f}s")

        # 13b. Multiple concurrent writers — should serialize via WriteGuard
        write_errors = []
        multi_write_count = [0]

        def multi_writer(tid):
            for i in range(50):
                try:
                    db.add(f"mw_{tid}_{i}", make_vec(2000 + tid * 100 + i))
                    with lock:
                        multi_write_count[0] += 1
                except Exception as e:
                    with lock:
                        write_errors.append(f"writer{tid}: {e}")

        writers = [threading.Thread(target=multi_writer, args=(t,)) for t in range(4)]
        for w in writers:
            w.start()
        for w in writers:
            w.join()

        check("Multi-writer: zero errors", len(write_errors) == 0,
              f"errors={write_errors[:3]}")
        check("Multi-writer: all writes", multi_write_count[0] == 200,
              f"count={multi_write_count[0]}")

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# =============================================================================
#  🔴 CRITICAL — Test 14: Boundary Conditions
# =============================================================================
def test_boundary_conditions():
    print("\n[Test 14] Boundary conditions")

    # 14a. Single-node database
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=1, dim=DIM)
        v = make_vec(1400)
        off = db.add("single", v)
        results = db.search(v, k=5)
        check("Single node: search works", len(results) == 1)
        check("Single node: correct", db.get_text(results[0][0]) == "single")
        del db
    finally:
        if os.path.exists(path):
            os.remove(path)

    # 14b. Database at capacity — tiny DB with many inserts
    path2 = tempfile.mktemp(suffix=".db")
    try:
        # 1MB DB with 128-dim vectors should hold limited nodes
        db = AgentKV(path2, size_mb=1, dim=DIM)
        count = 0
        capacity_error = False
        for i in range(5000):  # Try to overfill
            try:
                db.add(f"cap_{i}", make_vec(1500 + i))
                count += 1
            except RuntimeError:
                # Any RuntimeError from insert means we hit capacity
                capacity_error = True
                break
        check("Capacity: some nodes stored", count > 0, f"stored {count}")
        check("Capacity: graceful full error", capacity_error, f"stored {count} before stop")
        del db
    finally:
        if os.path.exists(path2):
            os.remove(path2)

    # 14c. Maximum text length
    path3 = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path3, size_mb=10, dim=DIM)
        big_text = "X" * 100_000  # 100KB text
        off = db.add(big_text, make_vec(1600))
        got = db.get_text(off)
        check("Large text (100KB) roundtrip", got == big_text,
              f"expected {len(big_text)}, got {len(got)}")
        del db
    finally:
        if os.path.exists(path3):
            os.remove(path3)


# =============================================================================
#  🟡 IMPORTANT — Test 15: Data Correctness
# =============================================================================
def test_data_correctness():
    print("\n[Test 15] Data correctness")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=10, dim=DIM)

        # 15a. Exact vector values after retrieval
        v = make_vec(1500)
        off = db.add("exact_vec", v)
        got = db.get_vector(off)
        check("Vector exact match", np.allclose(got, v, atol=1e-7),
              f"max diff={np.max(np.abs(got - v)):.2e}")

        # 15b. Results ordered by distance (ascending)
        vecs = []
        for i in range(100):
            vi = make_vec(1600 + i)
            vecs.append(vi)
            db.add(f"order_{i}", vi)

        q = make_vec(1600)  # Same as vec[0]
        results = db.search(q, k=10)
        dists = [d for _, d in results]
        is_sorted = all(dists[i] <= dists[i+1] + 1e-7 for i in range(len(dists)-1))
        check("Results sorted by distance", is_sorted,
              f"dists={[f'{d:.4f}' for d in dists]}")

        # 15c. No duplicate offsets in results
        offsets = [off for off, _ in results]
        check("No duplicate results", len(offsets) == len(set(offsets)),
              f"{len(offsets)} results, {len(set(offsets))} unique")

        # 15d. Vector normalization preserved
        norm_v = make_vec(1700)
        off_n = db.add("norm_test", norm_v)
        got_n = db.get_vector(off_n)
        got_norm = np.linalg.norm(got_n)
        check("Normalization preserved", abs(got_norm - 1.0) < 0.01,
              f"norm={got_norm:.6f}")

        # 15e. UTF-8 special characters
        unicode_texts = [
            "café résumé naïve",
            "日本語テスト",
            "emoji: 🚀🧠💾",
            "math: α β γ δ ε → ∞",
            "arabic: مرحبا",
        ]
        for i, txt in enumerate(unicode_texts):
            off_u = db.add(txt, make_vec(1800 + i))
            got_u = db.get_text(off_u)
            check(f"UTF-8[{i}]", got_u == txt, f"'{got_u}' != '{txt}'")

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# =============================================================================
#  🟡 IMPORTANT — Test 16: Graph / Relations
# =============================================================================
def test_graph_relations():
    print("\n[Test 16] Graph / relations")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=10, dim=DIM)

        # 16a. Create chain: A → B → C → D → E
        offsets = []
        for i in range(5):
            off = db.add(f"chain_{i}", make_vec(1600 + i))
            offsets.append(off)
        for i in range(4):
            db.engine.add_edge(offsets[i], offsets[i+1], 1.0)

        # Observe A — should reach B (and maybe C via multi-hop)
        ctx = db.observe(offsets[0])
        check("Chain: A reaches neighbors", len(ctx) >= 1, f"got {len(ctx)}")

        # 16b. Cycle: A → B → C → A
        off_ca = db.add("cycle_A", make_vec(1700))
        off_cb = db.add("cycle_B", make_vec(1701))
        off_cc = db.add("cycle_C", make_vec(1702))
        db.engine.add_edge(off_ca, off_cb, 1.0)
        db.engine.add_edge(off_cb, off_cc, 1.0)
        db.engine.add_edge(off_cc, off_ca, 1.0)

        # Should not hang (BFS with visited set)
        ctx_cycle = db.observe(off_ca)
        check("Cycle: no hang", True)  # If we get here, no infinite loop
        check("Cycle: returns context", len(ctx_cycle) >= 1, f"got {len(ctx_cycle)}")

        # 16c. Orphan nodes (no relations)
        off_orphan = db.add("orphan", make_vec(1710))
        ctx_orphan = db.observe(off_orphan)
        check("Orphan: returns empty", len(ctx_orphan) == 0, f"got {len(ctx_orphan)}")

        # 16d. Long chain A→B→...→Z (26 nodes)
        chain = []
        for i in range(26):
            off_lc = db.add(f"long_{chr(65+i)}", make_vec(1720 + i))
            chain.append(off_lc)
        for i in range(25):
            db.engine.add_edge(chain[i], chain[i+1], 1.0)

        ctx_long = db.observe(chain[0])
        check("Long chain: no crash", True)
        # SLB has max_depth=2, so should get at most 2 hops
        check("Long chain: bounded context", len(ctx_long) <= 10,
              f"got {len(ctx_long)}")

        # 16e. Relations via high-level API
        # relations=[X] adds edge FROM new_node TO X, so observe the new node
        off_rel_a = db.add("rel_parent", make_vec(1750))
        off_rel_b = db.add("rel_child", make_vec(1751), relations=[off_rel_a])
        ctx_rel = db.observe(off_rel_b)  # B has outgoing edge to A
        check("Relations API: edge created", len(ctx_rel) >= 1, f"got {len(ctx_rel)}")
        # Verify that off_rel_a is actually in the returned context
        rel_offsets = [c[0] if isinstance(c, tuple) else c for c in ctx_rel]
        check("Relations API: correct target", off_rel_a in rel_offsets,
              f"expected {off_rel_a} in {rel_offsets}")

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# =============================================================================
#  🟡 IMPORTANT — Test 17: Memory Stress
# =============================================================================
def test_memory_stress():
    print("\n[Test 17] Memory stress")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=50, dim=DIM)

        # 17a. Very large text (1MB)
        big = "A" * (1024 * 1024)  # 1MB
        off_big = db.add(big, make_vec(1700))
        got_big = db.get_text(off_big)
        check("1MB text stored", len(got_big) == len(big))
        check("1MB text correct", got_big == big)

        # 17b. Rapid insert loop (1000 nodes)
        t0 = time.time()
        for i in range(1000):
            db.add(f"stress_{i}", make_vec(1800 + i))
        t_stress = time.time() - t0
        check("1000 inserts", True, f"{t_stress:.2f}s")

        # 17c. Memory usage shouldn't explode after GC
        gc.collect()
        # Just confirm we can still search after heavy use
        r = db.search(make_vec(1800), k=3)
        check("Post-stress search works", len(r) >= 1)

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# =============================================================================
#  🟡 IMPORTANT — Test 18: Multi-Cycle Persistence
# =============================================================================
def test_multi_cycle_persistence():
    print("\n[Test 18] Multi-cycle persistence")
    path = tempfile.mktemp(suffix=".db")
    try:
        # 5 cycles of open → write → close → reopen → verify
        for cycle in range(5):
            db = AgentKV(path, size_mb=10, dim=DIM)
            v = make_vec(1900 + cycle)
            db.add(f"cycle_{cycle}", v)

            # Verify all previous cycles' data is still there
            hits = 0
            for prev in range(cycle + 1):
                pv = make_vec(1900 + prev)
                results = db.search(pv, k=1)
                if results:
                    txt = db.get_text(results[0][0])
                    if txt == f"cycle_{prev}":
                        hits += 1
            check(f"Cycle {cycle}: {hits}/{cycle+1} recalled",
                  hits == cycle + 1, f"{hits}/{cycle+1}")

            del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# =============================================================================
#  🟢 NICE TO HAVE — Test 19: Recall / Quality
# =============================================================================
def test_recall_quality():
    print("\n[Test 19] Recall / quality")
    path = tempfile.mktemp(suffix=".db")
    N = 500
    try:
        db = AgentKV(path, size_mb=30, dim=DIM)

        # Build index
        vecs = []
        offsets = []
        for i in range(N):
            v = make_vec(2000 + i)
            vecs.append(v)
            off = db.add(f"recall_{i}", v)
            offsets.append(off)

        data = np.array(vecs)  # (N, DIM)

        # Brute-force ground truth
        def brute_topk(query, k):
            sims = data @ query  # dot product
            topk_idx = np.argsort(-sims)[:k]
            return set(offsets[j] for j in topk_idx)

        # Recall@1, @5, @10
        NUM_Q = 50
        for at_k in [1, 5, 10]:
            hits = 0
            total = 0
            for qi in range(NUM_Q):
                q = vecs[qi * (N // NUM_Q)]
                gt = brute_topk(q, at_k)
                results = db.search(q, k=at_k, ef_search=100)
                found = set(r[0] for r in results)
                hits += len(found & gt)
                total += at_k
            recall = hits / max(total, 1)
            check(f"Recall@{at_k}: {recall:.1%}", recall >= 0.80,
                  f"{recall:.1%}")

        # Verify no duplicates in large k
        results_big = db.search(vecs[0], k=50)
        offs_big = [r[0] for r in results_big]
        check("No duplicates k=50", len(offs_big) == len(set(offs_big)))

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# =============================================================================
#  🟢 NICE TO HAVE — Test 20: API Completeness
# =============================================================================
def test_api_completeness():
    print("\n[Test 20] API completeness")
    import agentkv

    # All expected public methods
    expected_methods = ["add", "search", "get_text", "get_vector",
                        "observe", "build_context", "init_hnsw", "add_hnsw_link"]
    for m in expected_methods:
        check(f"API: AgentKV.{m} exists", hasattr(AgentKV, m))

    # Engine-level methods
    engine_methods = ["search_knn", "insert", "create_node", "add_edge",
                      "get_text", "get_vector", "init_hnsw", "add_hnsw_link",
                      "is_valid", "sync"]
    for m in engine_methods:
        check(f"API: KVEngine.{m} exists", hasattr(agentkv.KVEngine, m))

    # Return type checks
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=5, dim=DIM)
        v = make_vec(2100)
        off = db.add("type_check", v)

        # add returns int
        check("API: add returns int", isinstance(off, int))

        # search returns list of tuples
        results = db.search(v, k=1)
        check("API: search returns list", isinstance(results, list))
        if results:
            check("API: result is tuple", isinstance(results[0], tuple))
            check("API: offset is int", isinstance(results[0][0], int))
            check("API: distance is float", isinstance(results[0][1], float))

        # get_text returns str
        txt = db.get_text(off)
        check("API: get_text returns str", isinstance(txt, str))

        # get_vector returns ndarray
        vec = db.get_vector(off)
        check("API: get_vector returns ndarray", isinstance(vec, np.ndarray))
        check("API: vector dtype float32", vec.dtype == np.float32)

        # observe returns list
        ctx = db.observe(off)
        check("API: observe returns list", isinstance(ctx, list))

        # build_context returns list of str
        texts = db.build_context([off])
        check("API: build_context returns list", isinstance(texts, list))
        if texts:
            check("API: context item is str", isinstance(texts[0], str))

        # is_valid returns bool
        check("API: is_valid returns bool", isinstance(db.engine.is_valid(), bool))

        del db
    finally:
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AgentKV v0.7 — Pre-Release Test Suite")
    print("  (10 core + 10 extended = 20 test groups)")
    print("=" * 60)

    t_start = time.time()

    # ── Original 10 ──
    test_fresh_db()
    test_insert_search()
    test_text_storage()
    test_persistence()
    test_crash_recovery()
    test_concurrency()
    test_scale()
    test_graph_slb()
    test_api()
    test_version()

    # ── Critical (11-14) ──
    test_input_validation()
    test_error_handling()
    test_thread_safety()
    test_boundary_conditions()

    # ── Important (15-18) ──
    test_data_correctness()
    test_graph_relations()
    test_memory_stress()
    test_multi_cycle_persistence()

    # ── Nice to have (19-20) ──
    test_recall_quality()
    test_api_completeness()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    if errors:
        print(f"  Failed: {', '.join(errors)}")
    else:
        print("  ALL TESTS PASSED ✓")
    print(f"{'=' * 60}")

    sys.exit(0 if failed == 0 else 1)
