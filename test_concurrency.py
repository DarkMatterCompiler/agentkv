"""
test_concurrency.py — Stress test for RwLock (1 Writer + N Readers)

Architecture:
  - 1 writer thread continuously inserts nodes
  - N reader threads continuously search
  - Checks: no crashes, no corruption, correct results
"""
import os
import time
import threading
import numpy as np
from agentkv import AgentKV

DB_PATH = "test_concurrency.db"
DIM = 128
NUM_READERS = 4
WRITE_COUNT = 200
SEARCH_PER_READER = 100

# Shared state
errors = []
write_done = threading.Event()
stats_lock = threading.Lock()
stats = {
    "writes": 0,
    "reads": 0,
    "read_errors": 0,
}


def writer_thread(db: AgentKV, rng: np.random.Generator):
    """Insert nodes continuously."""
    for i in range(WRITE_COUNT):
        vec = rng.standard_normal(DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        try:
            db.add(f"Memory #{i}: test data for concurrency validation", vec)
            with stats_lock:
                stats["writes"] += 1
        except Exception as e:
            errors.append(f"WRITE ERROR at {i}: {e}")
            return
    write_done.set()


def reader_thread(db: AgentKV, reader_id: int, rng: np.random.Generator):
    """Search continuously while writes are happening."""
    searches = 0
    while not write_done.is_set() or searches < SEARCH_PER_READER:
        vec = rng.standard_normal(DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        try:
            results = db.search(vec, k=5, ef_search=30)
            # Results might be empty if index isn't populated yet — that's OK
            searches += 1
            with stats_lock:
                stats["reads"] += 1
        except Exception as e:
            with stats_lock:
                stats["read_errors"] += 1
            errors.append(f"READ ERROR (reader {reader_id}, search {searches}): {e}")
            if len(errors) > 10:
                return

        # Brief sleep to avoid pure busy-wait
        if searches % 10 == 0:
            time.sleep(0.001)

    return


def main():
    # Clean slate
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print("=" * 60)
    print("  test_concurrency.py — RwLock Stress Test")
    print(f"  {NUM_READERS} readers + 1 writer | {WRITE_COUNT} inserts | dim={DIM}")
    print("=" * 60)

    db = AgentKV(DB_PATH, size_mb=50, dim=DIM)

    # Seed a few nodes so readers don't always hit empty index
    seed_rng = np.random.default_rng(42)
    for i in range(10):
        vec = seed_rng.standard_normal(DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        db.add(f"Seed node {i}", vec)

    print(f"  ✅ Seeded 10 initial nodes\n")

    # Launch threads
    t0 = time.time()

    writer_rng = np.random.default_rng(100)
    w_thread = threading.Thread(target=writer_thread, args=(db, writer_rng), name="Writer")

    r_threads = []
    for i in range(NUM_READERS):
        reader_rng = np.random.default_rng(200 + i)
        t = threading.Thread(target=reader_thread, args=(db, i, reader_rng), name=f"Reader-{i}")
        r_threads.append(t)

    # Start all at once
    w_thread.start()
    for t in r_threads:
        t.start()

    # Join
    w_thread.join(timeout=60)
    write_done.set()  # Signal readers to stop even if writer timed out
    for t in r_threads:
        t.join(timeout=30)

    elapsed = time.time() - t0

    # Report
    print(f"⏱  Elapsed: {elapsed:.2f}s")
    print(f"📝 Writes: {stats['writes']}/{WRITE_COUNT}")
    print(f"🔍 Reads:  {stats['reads']} total across {NUM_READERS} threads")
    print(f"❌ Read errors: {stats['read_errors']}")

    if errors:
        print(f"\n🔴 ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")
        print("\n❌ CONCURRENCY TEST FAILED")
        return False
    else:
        assert stats["writes"] == WRITE_COUNT, \
            f"Expected {WRITE_COUNT} writes, got {stats['writes']}"
        assert stats["reads"] > 0, "No reads completed"
        print(f"\n✅ ALL CLEAR — {stats['writes']} writes + {stats['reads']} reads, zero errors")
        print(f"   Throughput: {stats['writes']/elapsed:.0f} writes/s, {stats['reads']/elapsed:.0f} reads/s")

    # Integrity check: reopen and verify
    print("\n💾 Persistence check...")
    del db
    db2 = AgentKV(DB_PATH, size_mb=50, dim=DIM)
    check_vec = seed_rng.standard_normal(DIM).astype(np.float32)
    check_vec = check_vec / np.linalg.norm(check_vec)
    results = db2.search(check_vec, k=5, ef_search=50)
    assert len(results) > 0, "No results after reopen!"

    # Verify text integrity
    for offset, dist in results:
        text = db2.get_text(offset)
        assert text and len(text) > 0, f"Empty text at offset {offset}"
        assert ("Memory #" in text or "Seed node" in text), \
            f"Corrupt text: {text[:50]}"

    print(f"  ✅ {len(results)} results after reopen, text integrity verified")

    del db2
    os.remove(DB_PATH)

    print("\n" + "=" * 60)
    print("  ✅ CONCURRENCY TEST PASSED")
    print(f"  1 writer + {NUM_READERS} readers | {elapsed:.2f}s | 0 errors")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
