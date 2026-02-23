#!/usr/bin/env python3
"""Test pip install agentkv works correctly."""
import agentkv
import numpy as np

print(f"AgentKV version: {agentkv.__version__}")

# Create test database
db = agentkv.AgentKV("/tmp/test_pip_install.db", size_mb=5, dim=128)

# Test basic operations
vec = np.random.randn(128).astype(np.float32)
vec /= np.linalg.norm(vec)

off = db.add("Test memory", vec)
print(f"✓ Added node at offset: {off}")

results = db.search(vec, k=1)
print(f"✓ Search returned {len(results)} results")

text = db.get_text(off)
print(f"✓ Retrieved text: '{text}'")

print("\n✅ Installation successful! pip install agentkv works.")
