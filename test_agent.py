import agentkv
import numpy as np
import time

# 1. Initialize Engine (10MB)
db_path = "agent_memory_py.db"
db = agentkv.KVEngine(db_path, 10 * 1024 * 1024)

# 2. Create Data
print("[Python] Creating Vector Embeddings...")
vec_a = np.random.rand(1536).astype(np.float32)
vec_b = np.random.rand(1536).astype(np.float32)

# Pass list or numpy array to C++
node_a = db.create_node(100, vec_a) 
node_b = db.create_node(101, vec_b)

print(f"[Python] Created Nodes at offsets: {node_a}, {node_b}")

# 3. Verify Zero-Copy Read
# We read the vector back from the DB. 
# Changing 'view' SHOULD verify if it's a copy or view, 
# but for safety, mmap is usually read-only unless we expose write access.
view_a = db.get_vector(node_a)

print(f"[Python] Vector Shape: {view_a.shape}")
print(f"[Python] First 5 elements: {view_a[:5]}")

if np.allclose(vec_a, view_a):
    print("SUCCESS: Data integrity verified.")
else:
    print("FAILURE: Data mismatch.")

# 4. Test SLB (The Brain)
slb = agentkv.ContextManager(db)
db.add_edge(node_a, node_b, 0.95) # A -> B

print("[Python] Agent looking at Node A...")
start = time.time()
slb.observe(node_a) # Releases GIL!
end = time.time()

ctx = slb.get_context()
print(f"[Python] Predicted Context: {ctx}")
print(f"[Python] Inference Time: {(end-start)*1000:.3f} ms")

assert node_b in ctx, "SLB failed to predict Node B"