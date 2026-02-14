"""
test_v05.py — End-to-end test of the AgentKV Memory Chat pipeline.

Tests the full loop:
  embed → search → context → generate → store → persist → recall
"""
import os
import time
import numpy as np
from agentkv import AgentKV
from agentkv.ollama import get_embedding, chat, is_available, list_models

DB_PATH = "test_v05.db"
DIM = 768
CHAT_MODEL = "llama3"

SYSTEM_PROMPT = """You are a helpful assistant with access to long-term memory.
Use the context below to inform your answers naturally.
Do NOT mention "context" or "memory retrieval" — just answer naturally.

{context}"""


def format_context(memories):
    if not memories:
        return "No relevant memories found."
    return "Relevant memories:\n" + "\n".join(f"- {m}" for m in memories)


def main():
    # Clean slate
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print("=" * 55)
    print("  test_v05.py — Memory Chat Pipeline Test")
    print("=" * 55)

    # 1. Environment
    assert is_available(), "Ollama not running"
    models = [m.split(":")[0] for m in list_models()]
    assert "nomic-embed-text" in models, "nomic-embed-text missing"
    assert "llama3" in models, "llama3 missing"
    print("✅ Environment OK")

    # 2. Init AgentKV
    db = AgentKV(DB_PATH, size_mb=50, dim=DIM)
    print("✅ AgentKV initialized")

    # 3. Seed some memories
    seed_memories = [
        "My name is Alex and I'm building a database called AgentKV.",
        "AgentKV uses HNSW for vector search and mmap for persistence.",
        "I like coffee in the morning and tea in the evening.",
        "The project is written in C++ with Python bindings via nanobind.",
        "My favorite programming language is C++ but I also use Python daily.",
    ]

    print(f"\n📝 Seeding {len(seed_memories)} memories...")
    t0 = time.time()
    for mem in seed_memories:
        vec = get_embedding(mem)
        db.add(mem, vec)
    t_seed = time.time() - t0
    print(f"  ✅ Seeded in {t_seed:.1f}s ({t_seed/len(seed_memories)*1000:.0f}ms/memory)")

    # 4. Test retrieval — query related to coffee
    print("\n🔍 Test 1: Semantic recall (coffee query)")
    q1 = "What do I like to drink?"
    q1_vec = get_embedding(q1)
    results = db.search(q1_vec, k=3, ef_search=50)
    print(f"  Query: '{q1}'")
    for offset, dist in results:
        text = db.get_text(offset)
        print(f"  [{dist:.4f}] {text[:80]}")

    # The coffee memory should be in top results
    top_texts = [db.get_text(off) for off, _ in results]
    assert any("coffee" in t.lower() for t in top_texts), \
        f"Expected 'coffee' in top-3, got: {top_texts}"
    print("  ✅ Coffee memory recalled correctly")

    # 5. Test retrieval — query about the project
    print("\n🔍 Test 2: Semantic recall (project query)")
    q2 = "Tell me about the database project"
    q2_vec = get_embedding(q2)
    results2 = db.search(q2_vec, k=3, ef_search=50)
    print(f"  Query: '{q2}'")
    for offset, dist in results2:
        text = db.get_text(offset)
        print(f"  [{dist:.4f}] {text[:80]}")

    top_texts2 = [db.get_text(off) for off, _ in results2]
    assert any("agentkv" in t.lower() for t in top_texts2), \
        f"Expected 'AgentKV' in top-3, got: {top_texts2}"
    print("  ✅ Project memories recalled correctly")

    # 6. Full RAG loop: query → context → generate
    print("\n🤖 Test 3: Full RAG generation")
    user_msg = "What's my name and what am I working on?"
    user_vec = get_embedding(user_msg)

    # Retrieve
    results3 = db.search(user_vec, k=3, ef_search=50)
    memory_texts = [db.get_text(off) for off, _ in results3]
    context_block = format_context(memory_texts)
    system_msg = SYSTEM_PROMPT.format(context=context_block)

    # Generate
    t_gen = time.time()
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    response = chat(messages, model=CHAT_MODEL)
    t_gen = time.time() - t_gen

    print(f"  Query: '{user_msg}'")
    print(f"  Context: {len(memory_texts)} memories")
    print(f"  Response ({t_gen:.1f}s): {response[:200]}")

    # The response should mention Alex and AgentKV
    resp_lower = response.lower()
    assert "alex" in resp_lower, f"Expected 'Alex' in response: {response[:100]}"
    print("  ✅ LLM correctly used memory context")

    # 7. Store the conversation as new memory
    db.add(f"User asked: {user_msg}", user_vec)
    resp_vec = get_embedding(response)
    db.add(f"Assistant said: {response}", resp_vec)
    print("  ✅ Conversation stored as new memories")

    # 8. Persistence test — close and reopen
    print("\n💾 Test 4: Persistence across restart")
    del db

    db2 = AgentKV(DB_PATH, size_mb=50, dim=DIM)
    q_persist = get_embedding("What is AgentKV?")
    results_p = db2.search(q_persist, k=3, ef_search=50)
    persist_texts = [db2.get_text(off) for off, _ in results_p]
    assert any("agentkv" in t.lower() for t in persist_texts), \
        f"Persistence failed! Got: {persist_texts}"
    print(f"  ✅ {len(results_p)} memories survived restart")
    for off, dist in results_p:
        print(f"  [{dist:.4f}] {db2.get_text(off)[:80]}")

    del db2

    # Summary
    print("\n" + "=" * 55)
    print("  ✅ ALL v0.5 TESTS PASSED")
    print(f"  Seeding: {t_seed:.1f}s | RAG gen: {t_gen:.1f}s")
    print(f"  Pipeline: embed → search → context → generate → store → persist")
    print("=" * 55)

    # Cleanup
    os.remove(DB_PATH)


if __name__ == "__main__":
    main()
