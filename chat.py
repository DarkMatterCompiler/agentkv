"""
AgentKV Memory Chat v0.6 — CLI chatbot with persistent memory + web search.

Architecture:
    User Input → Embed (Ollama nomic-embed-text)
              → Search Memory (AgentKV HNSW)
              → Assemble Context
              → Generate (Ollama llama3)
              → [Tool Call?] → web_search → embed results → store in AgentKV
              → Final Response
              → Store Conversation as Memory

The agent autonomously decides when to search the web and stores
new knowledge into AgentKV for future recall.

Runs 100% locally. No cloud APIs (except DuckDuckGo for web search).
"""
import sys
import time
from agentkv import AgentKV
from agentkv.ollama import get_embedding, chat, is_available, list_models
from agentkv.tools import web_search, format_search_results, TOOL_PROMPT, TOOL_CALL_PATTERN

# --- Configuration ---
DB_PATH = "memory.db"
DB_SIZE_MB = 100
DIM = 768            # nomic-embed-text dimension
TOP_K = 3            # memories to retrieve per query
CHAT_MODEL = "llama3"

SYSTEM_PROMPT = """You are a helpful research assistant with long-term memory and web search.
You remember previous conversations with the user.
Use the memory context below if relevant. If not relevant, ignore it.
Do NOT mention "context", "memory retrieval", or "tool calls" to the user — just answer naturally.

{tool_instructions}

{context}"""


def format_context(memories: list[str]) -> str:
    """Format retrieved memories into a context block for the system prompt."""
    if not memories:
        return "No relevant memories found."
    parts = [f"- {mem}" for mem in memories]
    return "Relevant memories:\n" + "\n".join(parts)


def handle_tool_call(response: str, db: AgentKV) -> tuple[bool, str, str]:
    """
    Check if the LLM wants to use a tool. If so, execute it and store results.
    
    Returns:
        (tool_used, tool_output_for_llm, status_msg_for_user)
    """
    match = TOOL_CALL_PATTERN.search(response)
    if not match:
        return False, "", ""
    
    query = match.group(1)
    print(f"  🔎 Searching web: \"{query}\"...")
    
    t0 = time.time()
    results = web_search(query, max_results=5)
    t_search = time.time() - t0
    
    # Format results for the LLM
    formatted = format_search_results(results)
    
    # Store each search result as a memory in AgentKV
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
    
    status = f"  📥 {stored} results stored in memory ({t_search:.1f}s)"
    return True, formatted, status


def check_environment():
    """Verify Ollama is running and required models are available."""
    print("🔍 Checking environment...")

    if not is_available():
        print("❌ Ollama is not running. Start it with: ollama serve")
        sys.exit(1)
    print("  ✅ Ollama server is running")

    models = list_models()
    model_names = [m.split(":")[0] for m in models]

    for required in ["nomic-embed-text", CHAT_MODEL]:
        short = required.split(":")[0]
        if short in model_names:
            print(f"  ✅ Model '{required}' is available")
        else:
            print(f"  ❌ Model '{required}' not found. Run: ollama pull {required}")
            sys.exit(1)


def main():
    print("=" * 60)
    print("  AgentKV Memory Chat v0.6")
    print("  Persistent memory • Web search • Local LLM • Thread-safe")
    print("=" * 60)

    # 1. Environment check
    check_environment()

    # 2. Initialize AgentKV
    print(f"\n🧠 Loading memory from '{DB_PATH}'...")
    db = AgentKV(DB_PATH, size_mb=DB_SIZE_MB, dim=DIM)
    print("  ✅ Memory engine ready\n")

    # 3. Conversation state
    conversation: list[dict[str, str]] = []
    turn_count = 0

    print("Type your message (or 'quit' to exit, 'recall' to see memory).")
    print("The assistant can search the web and remember results.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye! Your memories are saved.")
            break

        # --- Debug command: show what the DB remembers ---
        if user_input.lower() == "recall":
            print("\n📂 Searching memory for recent context...")
            if conversation:
                last_msg = conversation[-1]["content"]
                q_vec = get_embedding(last_msg)
                results = db.search(q_vec, k=5, ef_search=50)
                if results:
                    for offset, dist in results:
                        text = db.get_text(offset)
                        print(f"  [{dist:.4f}] {text[:100]}...")
                else:
                    print("  (empty)")
            else:
                print("  No conversation yet.")
            print()
            continue

        turn_count += 1
        t_start = time.time()

        # --- Step 1: Embed the user's query ---
        query_vec = get_embedding(user_input)

        # --- Step 2: Search AgentKV for relevant memories ---
        results = db.search(query_vec, k=TOP_K, ef_search=50)
        memory_texts = []
        for offset, dist in results:
            text = db.get_text(offset)
            if text:
                memory_texts.append(text)

        # --- Step 3: Assemble context ---
        context_block = format_context(memory_texts)
        system_msg = SYSTEM_PROMPT.format(
            tool_instructions=TOOL_PROMPT,
            context=context_block,
        )

        # Build messages: system + rolling conversation + current input
        messages = [{"role": "system", "content": system_msg}]
        recent = conversation[-6:] if len(conversation) > 6 else conversation
        messages.extend(recent)
        messages.append({"role": "user", "content": user_input})

        # --- Step 4: Generate response (may include tool call) ---
        response = chat(messages, model=CHAT_MODEL)

        # --- Step 5: Tool dispatch ---
        tool_used, tool_output, tool_status = handle_tool_call(response, db)
        
        if tool_used:
            print(tool_status)
            
            # Feed tool results back to the LLM for a final answer
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Here are the web search results:\n\n{tool_output}\n\n"
                           f"Now answer the original question naturally using these results. "
                           f"Do NOT use TOOL_CALL again."
            })
            response = chat(messages, model=CHAT_MODEL)

        t_elapsed = time.time() - t_start

        # --- Step 6: Display ---
        tool_label = " + 🔎 web" if tool_used else ""
        print(f"\nAssistant: {response}")
        print(f"  ⏱ {t_elapsed:.1f}s | 🧠 {len(memory_texts)} memories{tool_label}\n")

        # --- Step 7: Store memories ---
        db.add(f"User said: {user_input}", query_vec)
        resp_vec = get_embedding(response)
        db.add(f"Assistant said: {response}", resp_vec)

        # --- Step 8: Update conversation window ---
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
