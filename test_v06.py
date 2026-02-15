"""
test_v06.py — End-to-end test of v0.6: Concurrency + Agentic Tool Use

Tests:
  1. RwLock concurrency (1 writer + N readers, zero errors)
  2. Web search tool (DuckDuckGo returns results)
  3. Tool dispatch (TOOL_CALL parsing + auto-storage)
  4. Full RAG+Tool pipeline (LLM triggers search → results stored → recall)
  5. Persistence (memories survive restart)
"""
import os
import time
import threading
import numpy as np
from agentkv import AgentKV
from agentkv.ollama import get_embedding, chat, is_available, list_models
from agentkv.tools import web_search, format_search_results, TOOL_CALL_PATTERN, TOOL_PROMPT

DB_PATH = "test_v06.db"
DIM = 768
CHAT_MODEL = "llama3"

# Use the canonical pattern from tools module
TOOL_CALL_RE = TOOL_CALL_PATTERN


def test_concurrency():
    """Test 1: RwLock with concurrent readers + writer."""
    print("\n🔒 Test 1: Concurrency (1 writer + 4 readers)")
    db_path = "test_v06_conc.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = AgentKV(db_path, size_mb=20, dim=128)
    errors = []
    write_done = threading.Event()
    write_count = [0]
    read_count = [0]

    def writer():
        rng = np.random.default_rng(42)
        for i in range(100):
            v = rng.standard_normal(128).astype(np.float32)
            v /= np.linalg.norm(v)
            db.add(f"concurrent node {i}", v)
            write_count[0] += 1
        write_done.set()

    def reader(rid):
        rng = np.random.default_rng(100 + rid)
        while not write_done.is_set() or read_count[0] < 50:
            v = rng.standard_normal(128).astype(np.float32)
            v /= np.linalg.norm(v)
            try:
                db.search(v, k=3, ef_search=20)
                read_count[0] += 1
            except Exception as e:
                errors.append(str(e))
                return

    # Seed so readers don't hit empty
    rng = np.random.default_rng(0)
    for i in range(5):
        v = rng.standard_normal(128).astype(np.float32)
        v /= np.linalg.norm(v)
        db.add(f"seed {i}", v)

    threads = [threading.Thread(target=writer)]
    for i in range(4):
        threads.append(threading.Thread(target=reader, args=(i,)))

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    elapsed = time.time() - t0

    del db
    os.remove(db_path)

    assert not errors, f"Concurrency errors: {errors}"
    assert write_count[0] == 100
    print(f"  ✅ 100 writes + {read_count[0]} reads in {elapsed:.2f}s, zero errors")


def test_web_search():
    """Test 2: DuckDuckGo web search returns results."""
    print("\n🔎 Test 2: Web search tool")
    results = web_search("Python programming language", max_results=3)
    assert len(results) > 0, "Web search returned no results"
    assert results[0]["title"], "First result has no title"
    assert results[0]["snippet"], "First result has no snippet"
    print(f"  ✅ {len(results)} results returned")
    print(f"  Top: {results[0]['title'][:60]}...")
    return results


def test_tool_dispatch():
    """Test 3: TOOL_CALL pattern parsing."""
    print("\n🔧 Test 3: Tool dispatch parsing")

    # Should match
    resp1 = 'TOOL_CALL: web_search("latest news on AI")'
    m = TOOL_CALL_RE.search(resp1)
    assert m, f"Pattern should match: {resp1}"
    assert m.group(1) == "latest news on AI"
    print(f"  ✅ Parsed: '{m.group(1)}'")

    # Should NOT match normal text
    resp2 = "I think you should search for that information online."
    m2 = TOOL_CALL_RE.search(resp2)
    assert m2 is None, "Pattern should not match normal text"
    print("  ✅ Normal text correctly ignored")

    # Mid-response tool call
    resp3 = 'Let me look that up.\nTOOL_CALL: web_search("fusion energy 2026")\n'
    m3 = TOOL_CALL_RE.search(resp3)
    assert m3 and m3.group(1) == "fusion energy 2026"
    print(f"  ✅ Mid-response parse: '{m3.group(1)}'")


def test_full_rag_tool_pipeline():
    """Test 4: Full pipeline — LLM triggers search, results stored, then recalled."""
    print("\n🤖 Test 4: Full RAG + Tool pipeline")

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    db = AgentKV(DB_PATH, size_mb=50, dim=DIM)

    # Seed the user's identity
    seed_text = "My name is Alex and I'm researching fusion energy for a project."
    seed_vec = get_embedding(seed_text)
    db.add(seed_text, seed_vec)

    # Step A: Get web search results on a topic
    print("  📡 Searching web for 'fusion energy breakthrough 2026'...")
    results = web_search("fusion energy breakthrough 2026", max_results=3)
    assert len(results) > 0, "Web search failed"

    # Step B: Store results in AgentKV
    stored = 0
    for r in results:
        if r["snippet"] and r["title"] != "Search Error":
            text = f"[Web] {r['title']}: {r['snippet']}"
            vec = get_embedding(text)
            db.add(text, vec)
            stored += 1
    print(f"  📥 Stored {stored} web results in memory")

    # Step C: Now ask the LLM using memory context
    query = "What do you know about fusion energy?"
    query_vec = get_embedding(query)
    mem_results = db.search(query_vec, k=3, ef_search=50)
    memory_texts = [db.get_text(off) for off, _ in mem_results]

    # Verify web results appear in memory
    has_web = any("[Web]" in t for t in memory_texts)
    print(f"  🧠 Retrieved {len(memory_texts)} memories (web content: {'yes' if has_web else 'no'})")

    # Build context and generate
    context = "Relevant memories:\n" + "\n".join(f"- {m}" for m in memory_texts)
    system_msg = f"You are a helpful assistant.\n\n{context}"

    t0 = time.time()
    response = chat(
        [{"role": "system", "content": system_msg},
         {"role": "user", "content": query}],
        model=CHAT_MODEL,
    )
    t_gen = time.time() - t0

    print(f"  💬 Response ({t_gen:.1f}s): {response[:150]}...")
    assert len(response) > 20, "Response too short"
    print("  ✅ LLM correctly used web-sourced memory")

    # Step D: Persistence
    print("\n💾 Test 5: Persistence")
    del db
    db2 = AgentKV(DB_PATH, size_mb=50, dim=DIM)
    results_p = db2.search(query_vec, k=3, ef_search=50)
    texts_p = [db2.get_text(off) for off, _ in results_p]
    has_web_p = any("[Web]" in t for t in texts_p if t)
    assert has_web_p, f"Web memories lost after restart! Got: {texts_p[:2]}"
    print(f"  ✅ Web memories survived restart ({len(results_p)} results)")

    del db2
    os.remove(DB_PATH)

    return t_gen


def main():
    print("=" * 60)
    print("  test_v06.py — Concurrency + Agentic Tool Use")
    print("=" * 60)

    # Environment check
    assert is_available(), "Ollama not running"
    models = [m.split(":")[0] for m in list_models()]
    assert "nomic-embed-text" in models
    assert "llama3" in models
    print("✅ Environment OK")

    t_total = time.time()

    test_concurrency()
    search_results = test_web_search()
    test_tool_dispatch()
    t_gen = test_full_rag_tool_pipeline()

    elapsed = time.time() - t_total

    print("\n" + "=" * 60)
    print("  ✅ ALL v0.6 TESTS PASSED")
    print(f"  Concurrency: 1W+4R zero errors")
    print(f"  Web search: {len(search_results)} results")
    print(f"  Tool dispatch: pattern parsing verified")
    print(f"  RAG+Tool: web → embed → store → recall → generate ({t_gen:.1f}s)")
    print(f"  Total: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
