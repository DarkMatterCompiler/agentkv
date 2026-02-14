"""
AgentKV Memory Chat — A CLI chatbot with persistent long-term memory.

Architecture:
    User Input → Embed (Ollama nomic-embed-text)
              → Search (AgentKV HNSW)
              → Assemble Context (ContextBuilder)
              → Generate (Ollama llama3)
              → Store Memory (AgentKV insert)

Runs 100% locally. No cloud APIs.
"""
import sys
import time
import numpy as np
from agentkv import AgentKV
from agentkv.ollama import get_embedding, chat, is_available, list_models

# --- Configuration ---
DB_PATH = "memory.db"
DB_SIZE_MB = 100
DIM = 768            # nomic-embed-text dimension
TOP_K = 3            # memories to retrieve per query
CONTEXT_TOKENS = 1024
CHAT_MODEL = "llama3"

SYSTEM_PROMPT = """You are a helpful assistant with access to long-term memory.
You remember previous conversations with the user.
Use the context below to inform your answers. If the context is relevant, use it naturally.
If the context is not relevant to the current question, ignore it and answer normally.
Do NOT mention "context" or "memory retrieval" to the user — just answer naturally.

{context}"""


def format_context(memories: list[str]) -> str:
    """Format retrieved memories into a context block for the system prompt."""
    if not memories:
        return "No relevant memories found."
    parts = [f"- {mem}" for mem in memories]
    return "Relevant memories:\n" + "\n".join(parts)


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
    print("=" * 55)
    print("  AgentKV Memory Chat v0.5")
    print("  Persistent memory • Local LLM • Zero cloud")
    print("=" * 55)

    # 1. Environment check
    check_environment()

    # 2. Initialize AgentKV
    print(f"\n🧠 Loading memory from '{DB_PATH}'...")
    db = AgentKV(DB_PATH, size_mb=DB_SIZE_MB, dim=DIM)
    print("  ✅ Memory engine ready\n")

    # 3. Conversation state
    # We keep a short rolling window for multi-turn coherence,
    # but long-term memory lives in AgentKV.
    conversation: list[dict[str, str]] = []
    turn_count = 0

    print("Type your message (or 'quit' to exit, 'recall' to see memory).\n")

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
                        print(f"  [{dist:.4f}] {text[:80]}...")
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
        system_msg = SYSTEM_PROMPT.format(context=context_block)

        # Build messages: system + rolling conversation + current input
        messages = [{"role": "system", "content": system_msg}]

        # Keep last 6 turns (3 user + 3 assistant) for coherence
        recent = conversation[-6:] if len(conversation) > 6 else conversation
        messages.extend(recent)
        messages.append({"role": "user", "content": user_input})

        # --- Step 4: Generate response ---
        response = chat(messages, model=CHAT_MODEL)

        t_elapsed = time.time() - t_start

        # --- Step 5: Display ---
        print(f"\nAssistant: {response}")
        print(f"  ⏱ {t_elapsed:.1f}s | 🧠 {len(memory_texts)} memories recalled\n")

        # --- Step 6: Store memories ---
        # Store the user's message
        db.add(f"User said: {user_input}", query_vec)

        # Store the assistant's response (with its own embedding)
        resp_vec = get_embedding(response)
        db.add(f"Assistant said: {response}", resp_vec)

        # --- Step 7: Update conversation window ---
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
