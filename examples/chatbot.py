#!/usr/bin/env python3
"""
Example: Memory Chatbot with AgentKV + Ollama + Web Search

Interactive CLI that remembers conversations, searches the web, and stores
new knowledge automatically. Runs 100% local (except DuckDuckGo for web search).

Requires:
    pip install agentkv[all]
    ollama pull nomic-embed-text
    ollama pull llama3
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentkv import AgentKV
from agentkv.ollama import get_embedding, chat, is_available, list_models
from agentkv.tools import (
    web_search, format_search_results, TOOL_PROMPT, TOOL_CALL_PATTERN
)

DB_PATH = "chatbot_memory.db"
DB_SIZE_MB = 100
DIM = 768
TOP_K = 3
CHAT_MODEL = "llama3"

SYSTEM_PROMPT = """You are a helpful assistant with long-term memory and web search.
You remember previous conversations. Use memory context if relevant.
Do NOT mention "context", "memory", or "tool calls" — just answer naturally.

{tool_instructions}

{context}"""


def format_context(memories):
    if not memories:
        return "No relevant memories found."
    return "Relevant memories:\n" + "\n".join(f"- {m}" for m in memories)


def handle_tool_call(response, db):
    match = TOOL_CALL_PATTERN.search(response)
    if not match:
        return False, "", ""

    query = match.group(1)
    print(f"  Searching web: \"{query}\"...")

    t0 = time.time()
    results = web_search(query, max_results=5)
    elapsed = time.time() - t0

    formatted = format_search_results(results)

    stored = 0
    for r in results:
        if r["snippet"] and r["title"] != "Search Error":
            text = f"[Web] {r['title']}: {r['snippet']} (source: {r['url']})"
            try:
                vec = get_embedding(text)
                db.add(text, vec)
                stored += 1
            except Exception:
                pass

    status = f"  {stored} results stored in memory ({elapsed:.1f}s)"
    return True, formatted, status


def main():
    print("=" * 50)
    print("  AgentKV Memory Chatbot")
    print("=" * 50)

    if not is_available():
        print("ERROR: Ollama not running. Start with: ollama serve")
        sys.exit(1)

    models = list_models()
    names = [m.split(":")[0] for m in models]
    for req in ["nomic-embed-text", CHAT_MODEL]:
        if req.split(":")[0] not in names:
            print(f"ERROR: Model '{req}' not found. Run: ollama pull {req}")
            sys.exit(1)

    if os.path.exists(DB_PATH):
        print(f"Resuming from existing memory ({DB_PATH})")
    db = AgentKV(DB_PATH, size_mb=DB_SIZE_MB, dim=DIM)
    print("Memory engine ready.\n")

    conversation = []
    print("Type a message (or 'quit' to exit).\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! Memories saved.")
            break

        t0 = time.time()
        q_vec = get_embedding(user_input)

        results = db.search(q_vec, k=TOP_K, ef_search=50)
        mem_texts = [db.get_text(off) for off, _ in results if db.get_text(off)]

        system_msg = SYSTEM_PROMPT.format(
            tool_instructions=TOOL_PROMPT,
            context=format_context(mem_texts),
        )
        messages = [{"role": "system", "content": system_msg}]
        messages.extend(conversation[-6:])
        messages.append({"role": "user", "content": user_input})

        response = chat(messages, model=CHAT_MODEL)

        tool_used, tool_out, tool_status = handle_tool_call(response, db)
        if tool_used:
            print(tool_status)
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Web results:\n\n{tool_out}\n\nAnswer naturally. No TOOL_CALL."
            })
            response = chat(messages, model=CHAT_MODEL)

        elapsed = time.time() - t0
        label = " + web" if tool_used else ""
        print(f"\nAssistant: {response}")
        print(f"  [{elapsed:.1f}s | {len(mem_texts)} memories{label}]\n")

        db.add(f"User said: {user_input}", q_vec)
        db.add(f"Assistant said: {response}", get_embedding(response))
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
