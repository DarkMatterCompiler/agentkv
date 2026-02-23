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

# Windows console: force UTF-8 output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

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


def safe_remove(path):
    """Remove a DB file.  On Windows the mmap handle may linger until GC runs,
    so we force collection + retry before giving up."""
    gc.collect()
    for _ in range(5):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.1)
    # Last resort — ignore on Windows
    if os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            pass


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
            safe_remove(path)


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
            safe_remove(path)


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
            safe_remove(path)


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
            safe_remove(path)


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
            safe_remove(path)


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
            safe_remove(path)


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

        insert_limit = 10000 if sys.platform == "win32" else 5000
        check("Scale: insert reasonable", insert_per < insert_limit,
              f"{insert_per:.0f} us/vec (limit {insert_limit})")
        check("Scale: recall >= 80%", recall >= 0.80, f"{recall:.1%}")

        del db
    finally:
        if os.path.exists(path):
            safe_remove(path)


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
        check("SLB A->B: B in context", off_b in ctx_offsets_a,
              f"expected {off_b} in {ctx_offsets_a}")

        # Observe B — should predict C (via B->C edge)
        ctx2 = db.observe(off_b)
        check("SLB B->C", len(ctx2) >= 1, f"got {len(ctx2)} nodes")
        ctx_offsets_b = [c[0] if isinstance(c, tuple) else c for c in ctx2]
        check("SLB B->C: C in context", off_c in ctx_offsets_b,
              f"expected {off_c} in {ctx_offsets_b}")

        del db
    finally:
        if os.path.exists(path):
            safe_remove(path)


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
            safe_remove(path)


# ─────────────────────────────────────────────────────
# Test 10: Version / package metadata
# ─────────────────────────────────────────────────────
def test_version():
    print("\n[Test 10] Version / package metadata")
    import agentkv
    check("Version defined", hasattr(agentkv, '__version__'))
    check("Version is 0.9.0", agentkv.__version__ == "0.9.0",
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
            safe_remove(path)


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
        check("Bad magic -> fresh DB", db2.engine.is_valid())
        del db2
    finally:
        if os.path.exists(path):
            safe_remove(path)

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
        check("Bad checksum -> recovery", db2 is not None)
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
            safe_remove(path2)

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
            safe_remove(path3)

    # 12d. Search on empty database
    path4 = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path4, size_mb=5, dim=DIM)
        results = db.search(make_vec(1240), k=5)
        check("Empty DB search -> []", len(results) == 0)
        del db
    finally:
        if os.path.exists(path4):
            safe_remove(path4)


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
            safe_remove(path)


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
            safe_remove(path)

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
            safe_remove(path2)

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
            safe_remove(path3)


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
            safe_remove(path)


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
            safe_remove(path)


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
            safe_remove(path)


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
            safe_remove(path)


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
            safe_remove(path)


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
            safe_remove(path)


# =============================================================================
#  v0.9 — Test 21: Batch Insert
# =============================================================================
def test_batch_insert():
    print("\n[Test 21] Batch insert")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=10, dim=DIM)

        N = 200
        vecs = np.random.randn(N, DIM).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        texts = [f"batch_{i}" for i in range(N)]

        t0 = time.time()
        offsets = db.add_batch(texts, vecs)
        batch_time = time.time() - t0

        check("Batch: correct count", len(offsets) == N, f"got {len(offsets)}")
        check("Batch: db.count", len(db) == N, f"got {len(db)}")

        # Verify round-trip
        check("Batch: text[0]", db.get_text(offsets[0]) == "batch_0")
        check("Batch: text[-1]", db.get_text(offsets[-1]) == f"batch_{N-1}")

        # Search should find the exact vector
        results = db.search(vecs[42], k=1)
        check("Batch: search finds exact", db.get_text(results[0][0]) == "batch_42",
              f"got {db.get_text(results[0][0])}")

        # Compare batch vs sequential speed
        path2 = tempfile.mktemp(suffix=".db")
        db2 = AgentKV(path2, size_mb=10, dim=DIM)
        t1 = time.time()
        for i in range(N):
            db2.add(texts[i], vecs[i])
        seq_time = time.time() - t1
        del db2
        safe_remove(path2)

        speedup = seq_time / max(batch_time, 1e-9)
        print(f"    Batch: {batch_time:.3f}s  Seq: {seq_time:.3f}s  Speedup: {speedup:.1f}x")
        check("Batch: faster than sequential", speedup > 1.0,
              f"speedup {speedup:.1f}x")

        # Batch with metadata
        metas = [{"src": "batch"} for _ in range(N)]
        path3 = tempfile.mktemp(suffix=".db")
        db3 = AgentKV(path3, size_mb=10, dim=DIM)
        offsets3 = db3.add_batch(texts, vecs, metas)
        check("Batch+meta: metadata set", db3.get_metadata(offsets3[0], "src") == "batch")
        del db3
        safe_remove(path3)

        del db
    finally:
        if os.path.exists(path):
            safe_remove(path)


# =============================================================================
#  v0.9 — Test 22: Delete / Update
# =============================================================================
def test_delete_update():
    print("\n[Test 22] Delete / Update")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=10, dim=DIM)

        v1 = make_vec(9001)
        v2 = make_vec(9002)
        v3 = make_vec(9003)

        o1 = db.add("node_alpha", v1)
        o2 = db.add("node_beta", v2)
        o3 = db.add("node_gamma", v3)

        check("Del: initial count", len(db) == 3)

        # Delete one
        db.delete(o2)
        check("Del: count after delete", len(db) == 2)

        # Search should not return deleted node
        results = db.search(v2, k=3)
        found_texts = [db.get_text(r[0]) for r in results]
        check("Del: deleted node not in search", "node_beta" not in found_texts,
              f"found: {found_texts}")

        # Double-delete should be safe
        db.delete(o2)
        check("Del: double-delete safe", len(db) == 2)

        # Update = tombstone + re-insert
        v_new = make_vec(9999)
        o_new = db.update(o1, "node_alpha_v2", v_new)
        check("Update: new offset differs", o_new != o1)
        check("Update: old text gone from search",
              "node_alpha" not in [db.get_text(r[0]) for r in db.search(v1, k=5)])
        check("Update: new text found",
              db.get_text(o_new) == "node_alpha_v2")
        check("Update: count unchanged", len(db) == 2)

        del db
    finally:
        if os.path.exists(path):
            safe_remove(path)


# =============================================================================
#  v0.9 — Test 23: Metadata Filtering
# =============================================================================
def test_metadata_filtering():
    print("\n[Test 23] Metadata filtering")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=10, dim=DIM)

        # Insert nodes with different categories
        categories = ["science", "history", "science", "art", "history",
                       "science", "art", "science", "history", "art"]
        offsets = []
        for i, cat in enumerate(categories):
            v = make_vec(5000 + i)
            o = db.add(f"doc_{i}_{cat}", v, metadata={"category": cat})
            offsets.append(o)

        # Set additional metadata
        for i, o in enumerate(offsets):
            db.set_metadata(o, "idx", str(i))

        # Verify metadata retrieval
        check("Meta: get single key", db.get_metadata(offsets[0], "category") == "science")
        check("Meta: get idx", db.get_metadata(offsets[3], "idx") == "3")
        check("Meta: missing key = empty", db.get_metadata(offsets[0], "nonexistent") == "")

        # get_all_metadata
        all_meta = db.get_all_metadata(offsets[0])
        check("Meta: get_all has category", "category" in all_meta)
        check("Meta: get_all has idx", "idx" in all_meta)

        # Filtered search: only science
        q = make_vec(5000)  # should match doc_0_science
        results_all = db.search(q, k=10)
        results_science = db.search(q, k=10, where={"category": "science"})
        results_art = db.search(q, k=10, where={"category": "art"})

        check("Filter: all results >= science results",
              len(results_all) >= len(results_science))

        # All science results should have category=science
        for offset, dist in results_science:
            cat = db.get_metadata(offset, "category")
            if cat != "science":
                check("Filter: science only", False, f"got category={cat}")
                break
        else:
            check("Filter: science only", True)

        # Art results should all be art
        for offset, dist in results_art:
            cat = db.get_metadata(offset, "category")
            if cat != "art":
                check("Filter: art only", False, f"got category={cat}")
                break
        else:
            check("Filter: art only", True)

        # Count science = 4, art = 3
        check("Filter: science count", len(results_science) == 4,
              f"got {len(results_science)}")
        check("Filter: art count", len(results_art) == 3,
              f"got {len(results_art)}")

        del db
    finally:
        if os.path.exists(path):
            safe_remove(path)


# =============================================================================
#  v0.9 — Test 24: Distance Metrics
# =============================================================================
def test_distance_metrics():
    print("\n[Test 24] Distance metrics")

    # Test all 3 metrics
    for metric_name in ["cosine", "l2", "ip"]:
        path = tempfile.mktemp(suffix=".db")
        try:
            db = AgentKV(path, size_mb=5, dim=DIM, metric=metric_name)
            check(f"Metric {metric_name}: created", db.metric == metric_name)

            # Insert
            v1 = make_vec(7001)
            v2 = make_vec(7002)
            o1 = db.add("m1", v1)
            o2 = db.add("m2", v2)

            # Self-search should return distance ~ 0
            results = db.search(v1, k=1)
            check(f"Metric {metric_name}: self-search",
                  db.get_text(results[0][0]) == "m1")
            if metric_name in ("cosine", "l2"):
                check(f"Metric {metric_name}: self-dist ~0",
                      results[0][1] < 0.01, f"got {results[0][1]:.6f}")
            else:  # ip: distance = -dot, self should be most negative
                check(f"Metric {metric_name}: self-dist most negative",
                      results[0][1] < -0.5, f"got {results[0][1]:.6f}")

            del db
        finally:
            if os.path.exists(path):
                safe_remove(path)

    # Invalid metric should raise
    try:
        db = AgentKV("_bad_metric.db", size_mb=1, dim=4, metric="hamming")
        check("Metric: invalid raises", False, "no exception")
        del db
        safe_remove("_bad_metric.db")
    except ValueError:
        check("Metric: invalid raises", True)


# =============================================================================
#  v0.9 — Test 25: Count / Iteration
# =============================================================================
def test_count_iteration():
    print("\n[Test 25] Count / Iteration")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=10, dim=DIM)

        # Empty DB
        check("Count: empty db", len(db) == 0)
        check("Keys: empty db", len(db.keys()) == 0)

        # Insert some nodes
        N = 50
        offsets = []
        for i in range(N):
            o = db.add(f"iter_{i}", make_vec(8000 + i))
            offsets.append(o)

        check("Count: after insert", len(db) == N, f"got {len(db)}")
        check("Count: count() method", db.count() == N)

        all_keys = db.keys()
        check("Keys: length", len(all_keys) == N, f"got {len(all_keys)}")

        # All inserted offsets should be in keys
        keys_set = set(all_keys)
        all_found = all(o in keys_set for o in offsets)
        check("Keys: all offsets present", all_found)

        # items() returns (offset, text)
        items = db.items()
        check("Items: length", len(items) == N)
        texts = [t for _, t in items]
        check("Items: all texts present",
              all(f"iter_{i}" in texts for i in range(N)))

        # Delete some and verify
        for i in range(10):
            db.delete(offsets[i])
        check("Count: after delete 10", len(db) == N - 10,
              f"got {len(db)}")
        check("Keys: after delete 10", len(db.keys()) == N - 10)

        del db
    finally:
        if os.path.exists(path):
            safe_remove(path)


# =============================================================================
#  v0.9 — Test 26: SIMD Correctness
# =============================================================================
def test_simd_correctness():
    print("\n[Test 26] SIMD correctness")
    path = tempfile.mktemp(suffix=".db")
    try:
        # Test with various dimensions to exercise SIMD edge cases
        for dim in [1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 128, 256, 384, 768, 1536]:
            db = AgentKV(path, size_mb=50, dim=dim, metric="cosine")

            v = np.random.randn(dim).astype(np.float32)
            v /= np.linalg.norm(v)
            o = db.add("simd_test", v)

            results = db.search(v, k=1)
            dist = results[0][1]
            # Self-distance for normalized cosine should be ~0
            if dist > 0.001:
                check(f"SIMD dim={dim} self-dist", False,
                      f"expected ~0, got {dist:.6f}")
                del db
                safe_remove(path)
                break
            del db
            safe_remove(path)
        else:
            check("SIMD: all dims correct", True)

        # L2 cross-check: compute expected L2 in numpy, compare with DB
        path2 = tempfile.mktemp(suffix=".db")
        db = AgentKV(path2, size_mb=10, dim=DIM, metric="l2")
        v_a = np.random.randn(DIM).astype(np.float32)
        v_b = np.random.randn(DIM).astype(np.float32)
        db.add("a", v_a)
        db.add("b", v_b)
        results = db.search(v_a, k=2)
        # First result should be "a" with dist ~0 (self)
        db_dist_self = results[0][1]
        db_dist_other = results[1][1]
        np_dist = float(np.sum((v_a - v_b) ** 2))
        check("SIMD L2: self-dist ~0", db_dist_self < 0.001,
              f"got {db_dist_self:.6f}")
        check("SIMD L2: matches numpy",
              abs(db_dist_other - np_dist) < 0.01,
              f"db={db_dist_other:.6f} np={np_dist:.6f}")
        del db
        safe_remove(path2)

    finally:
        if os.path.exists(path):
            safe_remove(path)


# =============================================================================
#  v0.9 — Test 27: SIMD Edge Cases & Alignment
# =============================================================================
def test_simd_edge_cases():
    print("\n[Test 27] SIMD edge cases & alignment")
    path = tempfile.mktemp(suffix=".db")
    try:
        # 27a. Dimensions that are NOT multiples of SIMD lane widths
        #      SSE=4, AVX2=8, unrolled=16. Test the tail-loop paths.
        odd_dims = [1, 2, 3, 5, 6, 7, 9, 10, 13, 15, 17, 23, 25, 31, 33,
                    63, 65, 127, 129, 255, 257]
        all_ok = True
        for dim in odd_dims:
            db = AgentKV(path, size_mb=5, dim=dim, metric="cosine")
            v = np.random.randn(dim).astype(np.float32)
            v /= np.linalg.norm(v)
            db.add("odd", v)
            results = db.search(v, k=1)
            if results[0][1] > 0.002:
                check(f"SIMD odd dim={dim}", False,
                      f"self-dist={results[0][1]:.6f}")
                all_ok = False
            del db
            safe_remove(path)
        check("SIMD: odd dimensions all correct", all_ok)

        # 27b. Identical vectors — distance must be exactly 0 (or very close)
        db = AgentKV(path, size_mb=5, dim=DIM, metric="l2")
        v_same = np.ones(DIM, dtype=np.float32)
        db.add("dup1", v_same)
        db.add("dup2", v_same.copy())
        results = db.search(v_same, k=2)
        check("SIMD: identical L2 dist=0", results[0][1] < 1e-6,
              f"got {results[0][1]:.9f}")
        del db
        safe_remove(path)

        # 27c. Orthogonal vectors — cosine distance = 1.0
        db = AgentKV(path, size_mb=5, dim=4, metric="cosine")
        v_a = np.array([1, 0, 0, 0], dtype=np.float32)
        v_b = np.array([0, 1, 0, 0], dtype=np.float32)
        db.add("ortho_a", v_a)
        db.add("ortho_b", v_b)
        results = db.search(v_a, k=2)
        # Self should be ~0, other should be ~1.0
        check("SIMD: ortho self ~0", results[0][1] < 0.001)
        check("SIMD: ortho other ~1.0",
              abs(results[1][1] - 1.0) < 0.001,
              f"got {results[1][1]:.6f}")
        del db
        safe_remove(path)

        # 27d. Opposite vectors — cosine distance = 2.0
        db = AgentKV(path, size_mb=5, dim=DIM, metric="cosine")
        v_pos = make_vec(4001)
        v_neg = -v_pos.copy()
        db.add("pos", v_pos)
        db.add("neg", v_neg)
        results = db.search(v_pos, k=2)
        check("SIMD: opposite dist ~2.0",
              abs(results[1][1] - 2.0) < 0.01,
              f"got {results[1][1]:.6f}")
        del db
        safe_remove(path)

        # 27e. Zero vector handling (L2 metric — shouldn't crash)
        db = AgentKV(path, size_mb=5, dim=DIM, metric="l2")
        v_zero = np.zeros(DIM, dtype=np.float32)
        v_one = np.ones(DIM, dtype=np.float32)
        db.add("zero", v_zero)
        db.add("one", v_one)
        results = db.search(v_zero, k=2)
        expected_l2 = float(DIM)  # sum(1^2) * DIM times
        check("SIMD: zero vs ones L2",
              abs(results[1][1] - expected_l2) < 0.1,
              f"expected {expected_l2}, got {results[1][1]:.2f}")
        del db
        safe_remove(path)

        # 27f. Very large values — no overflow in float32 accumulation
        db = AgentKV(path, size_mb=5, dim=DIM, metric="l2")
        v_big = np.full(DIM, 1000.0, dtype=np.float32)
        v_big2 = np.full(DIM, 1001.0, dtype=np.float32)
        db.add("big", v_big)
        db.add("big2", v_big2)
        results = db.search(v_big, k=2)
        expected = float(DIM)  # sum((1001-1000)^2) = DIM
        check("SIMD: large values L2",
              abs(results[1][1] - expected) < 1.0,
              f"expected {expected}, got {results[1][1]:.2f}")
        del db
        safe_remove(path)

        # 27g. Inner product metric cross-check
        db = AgentKV(path, size_mb=5, dim=DIM, metric="ip")
        v_ip = make_vec(4444)
        db.add("ip_a", v_ip)
        db.add("ip_b", -v_ip)
        results = db.search(v_ip, k=2)
        # ip distance = -dot. Self: -dot(v,v) = -1.0 (normalized)
        check("SIMD: ip self dist ~-1.0",
              abs(results[0][1] - (-1.0)) < 0.01,
              f"got {results[0][1]:.6f}")
        # Opposite: -dot(v,-v) = +1.0
        check("SIMD: ip opposite dist ~+1.0",
              abs(results[1][1] - 1.0) < 0.01,
              f"got {results[1][1]:.6f}")
        del db
        safe_remove(path)

    finally:
        if os.path.exists(path):
            safe_remove(path)


# =============================================================================
#  v0.9 — Test 28: Metadata Stress & Edge Cases
# =============================================================================
def test_metadata_stress():
    print("\n[Test 28] Metadata stress & edge cases")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=20, dim=DIM)

        v = make_vec(6001)
        o = db.add("meta_node", v)

        # 28a. Many keys on one node
        N_KEYS = 100
        for i in range(N_KEYS):
            db.set_metadata(o, f"key_{i}", f"val_{i}")
        all_meta = db.get_all_metadata(o)
        check("MetaStress: many keys count", len(all_meta) == N_KEYS,
              f"got {len(all_meta)}")
        check("MetaStress: many keys read",
              db.get_metadata(o, "key_50") == "val_50")

        # 28b. Overwrite a key (prepend semantics — first match wins)
        db.set_metadata(o, "key_0", "UPDATED")
        check("MetaStress: overwrite", db.get_metadata(o, "key_0") == "UPDATED")

        # 28c. Empty key and value
        db.set_metadata(o, "", "empty_key_val")
        check("MetaStress: empty key", db.get_metadata(o, "") == "empty_key_val")

        db.set_metadata(o, "empty_val", "")
        check("MetaStress: empty value", db.get_metadata(o, "empty_val") == "")

        # 28d. Long key/value (1KB each)
        long_key = "K" * 1024
        long_val = "V" * 1024
        db.set_metadata(o, long_key, long_val)
        check("MetaStress: long key/val", db.get_metadata(o, long_key) == long_val)

        # 28e. Unicode in metadata
        db.set_metadata(o, "lang", "日本語テスト")
        check("MetaStress: unicode meta",
              db.get_metadata(o, "lang") == "日本語テスト")

        # 28f. Special characters
        db.set_metadata(o, "special", "a=b&c=d\nnewline\ttab")
        check("MetaStress: special chars",
              db.get_metadata(o, "special") == "a=b&c=d\nnewline\ttab")

        # 28g. Metadata filtering with many nodes
        N_FILTER = 500
        colors = ["red", "green", "blue", "yellow", "purple"]
        vecs = np.random.randn(N_FILTER, DIM).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        texts = [f"filt_{i}" for i in range(N_FILTER)]
        metas = [{"color": colors[i % len(colors)]} for i in range(N_FILTER)]
        filter_offsets = db.add_batch(texts, vecs, metas)

        q = vecs[0]  # Should match filt_0 (red)
        red_results = db.search(q, k=N_FILTER, where={"color": "red"})
        expected_red = N_FILTER // len(colors)  # 100
        check("MetaStress: filter count",
              len(red_results) == expected_red,
              f"expected {expected_red}, got {len(red_results)}")

        # All red results should actually be red
        all_red = all(db.get_metadata(off, "color") == "red"
                      for off, _ in red_results)
        check("MetaStress: filter correctness", all_red)

        # 28h. Multi-key filter
        for off in filter_offsets[:50]:
            db.set_metadata(off, "priority", "high")
        for off in filter_offsets[50:]:
            db.set_metadata(off, "priority", "low")

        # Search red + high priority
        red_high = db.search(q, k=N_FILTER,
                             where={"color": "red", "priority": "high"})
        # Among first 50 (high), red ones are those at indices 0,5,10,...,45 = 10
        check("MetaStress: multi-key filter",
              len(red_high) == 10,
              f"got {len(red_high)}")

        del db
    finally:
        if os.path.exists(path):
            safe_remove(path)


# =============================================================================
#  v0.9 — Test 29: Delete/Update Edge Cases
# =============================================================================
def test_delete_update_edge_cases():
    print("\n[Test 29] Delete/Update edge cases")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=10, dim=DIM)

        # 29a. Delete all nodes — DB should be empty but functional
        offsets = []
        for i in range(20):
            offsets.append(db.add(f"del_all_{i}", make_vec(3000 + i)))
        for o in offsets:
            db.delete(o)
        check("DelEdge: all deleted count=0", len(db) == 0)
        check("DelEdge: keys empty", len(db.keys()) == 0)

        # Can still insert after deleting everything
        o_new = db.add("after_nuke", make_vec(3999))
        check("DelEdge: insert after full delete", len(db) == 1)
        results = db.search(make_vec(3999), k=1)
        check("DelEdge: search after full delete",
              len(results) > 0 and db.get_text(results[0][0]) == "after_nuke",
              f"got {len(results)} results")

        # 29b. Search returns fewer than k when nodes are deleted
        db2_path = tempfile.mktemp(suffix=".db")
        db2 = AgentKV(db2_path, size_mb=5, dim=DIM)
        o_keep = []
        for i in range(10):
            o = db2.add(f"sparse_{i}", make_vec(3100 + i))
            if i >= 7:
                db2.delete(o)
            else:
                o_keep.append(o)
        results = db2.search(make_vec(3100), k=10)
        check("DelEdge: search respects k vs live",
              len(results) <= 7,
              f"got {len(results)} results (expected <=7)")
        del db2
        safe_remove(db2_path)

        # 29c. Update preserves metadata
        o_meta = db.add("has_meta", make_vec(3200),
                        metadata={"color": "blue", "priority": "high"})
        o_updated = db.update(o_meta, "has_meta_v2", make_vec(3201),
                              metadata={"color": "green", "priority": "high"})
        check("DelEdge: update meta color",
              db.get_metadata(o_updated, "color") == "green")
        check("DelEdge: update meta priority",
              db.get_metadata(o_updated, "priority") == "high")
        check("DelEdge: old offset is deleted",
              db.engine.is_deleted(o_meta))

        # 29d. Rapid insert-delete cycles
        for cycle in range(50):
            o = db.add(f"cycle_{cycle}", make_vec(3300 + cycle))
            db.delete(o)
        live_count = len(db)
        check("DelEdge: rapid cycles stable",
              live_count >= 1, f"count={live_count}")

        # 29e. Delete should not affect other nodes' search results
        db3_path = tempfile.mktemp(suffix=".db")
        db3 = AgentKV(db3_path, size_mb=5, dim=DIM)
        v_target = make_vec(3400)
        o_target = db3.add("target", v_target)
        neighbors = []
        for i in range(20):
            neighbors.append(db3.add(f"neighbor_{i}", make_vec(3401 + i)))
        # Delete half the neighbors
        for o in neighbors[:10]:
            db3.delete(o)
        results = db3.search(v_target, k=1)
        check("DelEdge: target still found",
              db3.get_text(results[0][0]) == "target")
        del db3
        safe_remove(db3_path)

        del db
    finally:
        if os.path.exists(path):
            safe_remove(path)


# =============================================================================
#  v0.9 — Test 30: Batch Operations Edge Cases
# =============================================================================
def test_batch_edge_cases():
    print("\n[Test 30] Batch operations edge cases")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=20, dim=DIM)

        # 30a. Batch of size 1
        vecs1 = np.random.randn(1, DIM).astype(np.float32)
        vecs1 /= np.linalg.norm(vecs1, axis=1, keepdims=True)
        offsets1 = db.add_batch(["single"], vecs1)
        check("BatchEdge: batch of 1", len(offsets1) == 1)
        check("BatchEdge: text of 1", db.get_text(offsets1[0]) == "single")

        # 30b. Batch with metadata — all nodes get correct tags
        N = 100
        vecs = np.random.randn(N, DIM).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        texts = [f"batch_m_{i}" for i in range(N)]
        metas = [{"idx": str(i), "group": "A" if i < 50 else "B"} for i in range(N)]
        offsets = db.add_batch(texts, vecs, metas)

        check("BatchEdge: meta[0].idx", db.get_metadata(offsets[0], "idx") == "0")
        check("BatchEdge: meta[99].idx", db.get_metadata(offsets[99], "idx") == "99")
        check("BatchEdge: meta[0].group", db.get_metadata(offsets[0], "group") == "A")
        check("BatchEdge: meta[50].group", db.get_metadata(offsets[50], "group") == "B")

        # 30c. Large batch
        N_LARGE = 1000
        large_vecs = np.random.randn(N_LARGE, DIM).astype(np.float32)
        large_vecs /= np.linalg.norm(large_vecs, axis=1, keepdims=True)
        large_texts = [f"large_{i}" for i in range(N_LARGE)]
        t0 = time.time()
        large_offsets = db.add_batch(large_texts, large_vecs)
        batch_time = time.time() - t0
        check("BatchEdge: large batch count",
              len(large_offsets) == N_LARGE)
        print(f"    1000 batch insert: {batch_time:.2f}s")

        # 30d. Search after batch should be accurate
        q = large_vecs[500]
        results = db.search(q, k=1)
        check("BatchEdge: search finds exact after batch",
              db.get_text(results[0][0]) == "large_500",
              f"got {db.get_text(results[0][0])}")

        # 30e. Total count is consistent
        expected_total = 1 + N + N_LARGE
        check("BatchEdge: total count",
              len(db) == expected_total,
              f"expected {expected_total}, got {len(db)}")

        # 30f. Batch with empty texts
        empty_vecs = np.random.randn(5, DIM).astype(np.float32)
        empty_vecs /= np.linalg.norm(empty_vecs, axis=1, keepdims=True)
        empty_texts = ["", "has_text", "", "", "also_text"]
        empty_offsets = db.add_batch(empty_texts, empty_vecs)
        check("BatchEdge: empty text stored",
              db.get_text(empty_offsets[0]) == "")
        check("BatchEdge: non-empty text stored",
              db.get_text(empty_offsets[1]) == "has_text")

        del db
    finally:
        if os.path.exists(path):
            safe_remove(path)


# =============================================================================
#  v0.9 — Test 31: Concurrent Operations on New Features
# =============================================================================
def test_concurrent_new_features():
    print("\n[Test 31] Concurrent operations on new features")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=20, dim=DIM)

        # Pre-populate
        N = 200
        vecs = np.random.randn(N, DIM).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        texts = [f"conc_{i}" for i in range(N)]
        offsets = db.add_batch(texts, vecs)

        conc_errors = []

        # 31a. Concurrent reads (search + metadata) while writer deletes
        def reader_search(n_iters):
            try:
                for _ in range(n_iters):
                    q = np.random.randn(DIM).astype(np.float32)
                    q /= np.linalg.norm(q)
                    results = db.search(q, k=5)
                    # Results should only contain non-deleted nodes
                    for off, dist in results:
                        if db.engine.is_deleted(off):
                            conc_errors.append("Got deleted node in search")
            except Exception as e:
                conc_errors.append(f"reader_search: {e}")

        def reader_metadata(n_iters):
            try:
                for _ in range(n_iters):
                    for off in offsets[:20]:
                        try:
                            db.get_text(off)
                        except Exception:
                            pass  # Node might be deleted
            except Exception as e:
                conc_errors.append(f"reader_metadata: {e}")

        def writer_delete(indices):
            try:
                for i in indices:
                    db.delete(offsets[i])
                    time.sleep(0.001)
            except Exception as e:
                conc_errors.append(f"writer_delete: {e}")

        def writer_insert(n):
            try:
                for i in range(n):
                    v = np.random.randn(DIM).astype(np.float32)
                    v /= np.linalg.norm(v)
                    db.add(f"conc_new_{i}", v)
            except Exception as e:
                conc_errors.append(f"writer_insert: {e}")

        threads = [
            threading.Thread(target=reader_search, args=(50,)),
            threading.Thread(target=reader_search, args=(50,)),
            threading.Thread(target=reader_metadata, args=(20,)),
            threading.Thread(target=writer_delete, args=(list(range(50, 100)),)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        check("Conc: no errors in mixed ops", len(conc_errors) == 0,
              "; ".join(conc_errors[:3]))

        # 31b. Concurrent filtered search + insert
        for off in offsets[:50]:
            db.set_metadata(off, "tag", "searchable")

        conc_errors2 = []

        def filtered_reader(n_iters):
            try:
                for _ in range(n_iters):
                    q = np.random.randn(DIM).astype(np.float32)
                    q /= np.linalg.norm(q)
                    db.search(q, k=5, where={"tag": "searchable"})
            except Exception as e:
                conc_errors2.append(f"filtered_reader: {e}")

        threads2 = [
            threading.Thread(target=filtered_reader, args=(30,)),
            threading.Thread(target=filtered_reader, args=(30,)),
            threading.Thread(target=writer_insert, args=(20,)),
        ]
        for t in threads2:
            t.start()
        for t in threads2:
            t.join(timeout=30)

        check("Conc: filtered search + insert", len(conc_errors2) == 0,
              "; ".join(conc_errors2[:3]))

        # 31c. Concurrent count/keys consistency
        count_before = len(db)
        keys_before = len(db.keys())
        check("Conc: count == keys length",
              count_before == keys_before,
              f"count={count_before}, keys={keys_before}")

        del db
    finally:
        if os.path.exists(path):
            safe_remove(path)


# =============================================================================
#  v0.9 — Test 32: Cross-Platform Consistency
# =============================================================================
def test_cross_platform():
    print("\n[Test 32] Cross-platform consistency")
    path = tempfile.mktemp(suffix=".db")
    try:
        db = AgentKV(path, size_mb=10, dim=DIM)

        # 32a. Deterministic vectors: same seed must produce same results
        v1 = make_vec(42)
        v2 = make_vec(42)
        check("XPlat: deterministic vectors", np.allclose(v1, v2))

        # 32b. Insert deterministic data, verify exact byte-level results
        np.random.seed(12345)
        N = 100
        for i in range(N):
            v = make_vec(12345 + i)
            db.add(f"xplat_{i}", v)

        # Search must return exact same ordering with same query
        q = make_vec(12345)
        results = db.search(q, k=5)
        check("XPlat: search returns results", len(results) == 5)
        check("XPlat: self is first",
              db.get_text(results[0][0]) == "xplat_0")

        # 32c. Text encoding: full Unicode range survives roundtrip
        test_strings = [
            "ASCII only",
            "Accents: cafe\u0301 re\u0301sume\u0301",
            "CJK: \u4f60\u597d\u4e16\u754c",
            "Emoji: \U0001f680\U0001f525\u2764\ufe0f",
            "Arabic: \u0645\u0631\u062d\u0628\u0627",
            "Math: \u03b1 \u03b2 \u03b3 \u2192 \u221e",
            "Null-adjacent: abc\x01\x02\x03def",
            "Mixed: Hello \u4e16\u754c \U0001f600 caf\u00e9",
        ]
        for i, s in enumerate(test_strings):
            v = make_vec(20000 + i)
            o = db.add(s, v)
            got = db.get_text(o)
            if got != s:
                check(f"XPlat: unicode[{i}]", False,
                      f"expected {repr(s)}, got {repr(got)}")
                break
        else:
            check("XPlat: all unicode roundtrips", True)

        # 32d. File size is OS-independent (same inputs = same write_head)
        count = len(db)
        check("XPlat: count matches", count == N + len(test_strings),
              f"got {count}")

        # 32e. Metadata with unicode keys/values
        o_uni = db.add("uni_meta", make_vec(20100))
        db.set_metadata(o_uni, "\u30ad\u30fc", "\u5024")
        check("XPlat: unicode metadata",
              db.get_metadata(o_uni, "\u30ad\u30fc") == "\u5024")

        del db
    finally:
        if os.path.exists(path):
            safe_remove(path)


# =============================================================================
#  v0.9 — Test 33: Performance Regression Baseline
# =============================================================================
def test_performance_baseline():
    print("\n[Test 33] Performance regression baseline")
    path = tempfile.mktemp(suffix=".db")
    is_win = sys.platform == "win32"
    try:
        db = AgentKV(path, size_mb=50, dim=DIM)

        # 33a. Single-insert throughput
        N_INSERT = 500
        vecs = np.random.randn(N_INSERT, DIM).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        t0 = time.time()
        offsets = []
        for i in range(N_INSERT):
            offsets.append(db.add(f"perf_{i}", vecs[i]))
        seq_time = time.time() - t0
        seq_per = (seq_time / N_INSERT) * 1e6
        insert_limit = 10000 if is_win else 5000
        print(f"    Sequential insert: {seq_per:.0f} us/vec ({N_INSERT} vecs)")
        check("Perf: seq insert < limit",
              seq_per < insert_limit,
              f"{seq_per:.0f} us/vec (limit {insert_limit})")

        # 33b. Batch-insert throughput
        N_BATCH = 1000
        batch_vecs = np.random.randn(N_BATCH, DIM).astype(np.float32)
        batch_vecs /= np.linalg.norm(batch_vecs, axis=1, keepdims=True)
        batch_texts = [f"bperf_{i}" for i in range(N_BATCH)]
        t0 = time.time()
        db.add_batch(batch_texts, batch_vecs)
        batch_time = time.time() - t0
        batch_per = (batch_time / N_BATCH) * 1e6
        batch_limit = 8000 if is_win else 4000
        print(f"    Batch insert: {batch_per:.0f} us/vec ({N_BATCH} vecs)")
        check("Perf: batch insert < limit",
              batch_per < batch_limit,
              f"{batch_per:.0f} us/vec (limit {batch_limit})")

        # 33c. Search latency
        N_QUERIES = 200
        t0 = time.time()
        for i in range(N_QUERIES):
            db.search(vecs[i % N_INSERT], k=10, ef_search=50)
        search_time = time.time() - t0
        search_per = (search_time / N_QUERIES) * 1e6
        search_limit = 5000 if is_win else 3000
        print(f"    Search: {search_per:.0f} us/query ({N_QUERIES} queries, "
              f"{N_INSERT + N_BATCH} total nodes)")
        check("Perf: search < limit",
              search_per < search_limit,
              f"{search_per:.0f} us/query (limit {search_limit})")

        # 33d. Filtered search latency (should not be much slower)
        for off in offsets[:250]:
            db.set_metadata(off, "perf_tag", "yes")
        t0 = time.time()
        for i in range(N_QUERIES):
            db.search(vecs[i % N_INSERT], k=10,
                      where={"perf_tag": "yes"})
        filter_time = time.time() - t0
        filter_per = (filter_time / N_QUERIES) * 1e6
        filter_limit = 10000 if is_win else 6000
        print(f"    Filtered search: {filter_per:.0f} us/query")
        check("Perf: filtered search < limit",
              filter_per < filter_limit,
              f"{filter_per:.0f} us/query (limit {filter_limit})")

        # 33e. Delete throughput
        t0 = time.time()
        for off in offsets:
            db.delete(off)
        del_time = time.time() - t0
        del_per = (del_time / len(offsets)) * 1e6
        print(f"    Delete: {del_per:.0f} us/node ({len(offsets)} nodes)")
        check("Perf: delete < 100us", del_per < 100,
              f"{del_per:.0f} us/node")

        # 33f. Count / iteration performance
        t0 = time.time()
        for _ in range(100):
            _ = len(db)
        count_time = time.time() - t0
        print(f"    100x len(): {count_time*1000:.1f} ms")
        check("Perf: len() fast", count_time < 0.1,
              f"{count_time*1000:.1f} ms")

        t0 = time.time()
        _ = db.keys()
        keys_time = time.time() - t0
        print(f"    keys() ({len(db)} nodes): {keys_time*1000:.1f} ms")
        keys_limit = 100 if is_win else 50
        check("Perf: keys() < limit",
              keys_time * 1000 < keys_limit,
              f"{keys_time*1000:.1f} ms (limit {keys_limit}ms)")

        del db
    finally:
        if os.path.exists(path):
            safe_remove(path)


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AgentKV v0.9 — Full Test Suite")
    print("  (10 core + 10 extended + 6 v0.9 + 7 edge cases = 33 test groups)")
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

    # ── v0.9 Features (21-26) ──
    test_batch_insert()
    test_delete_update()
    test_metadata_filtering()
    test_distance_metrics()
    test_count_iteration()
    test_simd_correctness()

    # ── v0.9 Edge Cases (27-33) ──
    test_simd_edge_cases()
    test_metadata_stress()
    test_delete_update_edge_cases()
    test_batch_edge_cases()
    test_concurrent_new_features()
    test_cross_platform()
    test_performance_baseline()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    if errors:
        print(f"  Failed: {', '.join(errors)}")
    else:
        print("  ALL TESTS PASSED ✓")
    print(f"{'=' * 60}")

    sys.exit(0 if failed == 0 else 1)
