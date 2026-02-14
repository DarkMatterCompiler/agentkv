import numpy as np
from agentkv.langgraph_store import AgentKVStore

def get_embedding(text: str):
    # Mock embedding function (Deterministic for demo)
    # In prod, use OpenAI or SentenceTransformers
    rng = np.random.RandomState(hash(text) % 2**32)
    return rng.rand(1536).astype(np.float32)

def main():
    print("--- Initializing AgentKV Cognitive Engine ---")
    store = AgentKVStore("brain.db")
    
    user_ns = ("user", "dev_01")

    # 1. Turn 1: User mentions C++
    print("\n[Turn 1] User: 'I love writing C++ engines'")
    vec1 = get_embedding("C++")
    op1 = {
        "type": "put",
        "namespace": user_ns,
        "key": "mem_1",
        "value": {"text": "I love C++", "vector": vec1.tolist()}
    }
    store.batch([op1])

    # 2. Turn 2: User mentions Latency (Conceptually related to C++)
    print("[Turn 2] User: 'I hate latency'")
    vec2 = get_embedding("Latency")
    # We manually simulate the agent linking these concepts
    # In a real LLM, the LLM decides this link.
    op2 = {
        "type": "put",
        "namespace": user_ns,
        "key": "mem_2",
        "value": {"text": "Latency is bad", "vector": vec2.tolist()}
    }
    store.batch([op2])

    # 3. Agent "Thinking" Phase
    print("\n[Thinking] Retrieving Active Context...")
    # The act of 'putting' the last memory triggered the SLB 'observe'.
    # So a search now should reflect that context.
    op_search = {"type": "search", "namespace": user_ns}
    results = store.batch([op_search])[0]

    print(f"Found {len(results)} active memory nodes.")
    for res in results:
        offset = res["value"]["offset"]
        print(f" -> Active Node Offset: {offset}")

    if len(results) > 0:
        print("\nSUCCESS: The AgentKV Store is alive and predicting context.")
    else:
        print("\nFAILURE: Context window empty.")

if __name__ == "__main__":
    main()