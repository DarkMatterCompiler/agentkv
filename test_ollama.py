"""Quick smoke test for Ollama API from WSL."""
from agentkv.ollama import get_embedding, chat, is_available, list_models
import time

print("=== Ollama Smoke Test ===")
print(f"Available: {is_available()}")
print(f"Models: {list_models()}")

# Test embedding
t0 = time.time()
v = get_embedding("hello world")
t1 = time.time()
print(f"Embedding: dim={v.shape[0]}, norm={float((v**2).sum()**0.5):.4f}, time={t1-t0:.2f}s")

# Test chat
t0 = time.time()
reply = chat([{"role": "user", "content": "Say hello in exactly 5 words."}], temperature=0.0)
t1 = time.time()
print(f"Chat: '{reply.strip()}'  time={t1-t0:.1f}s")

print("=== All OK ===")
