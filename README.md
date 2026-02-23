# AgentKV — The SQLite of Agent Memory

**A single-file, embeddable vector + graph store for AI agents. No server required.**

[![PyPI](https://img.shields.io/pypi/v/agentkv)](https://pypi.org/project/agentkv/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```bash
pip install agentkv
```

---

## Quickstart (5 lines)

```python
from agentkv import AgentKV
import numpy as np

db = AgentKV("brain.db", size_mb=50, dim=768)
db.add("Paris is the capital of France", np.random.rand(768).astype(np.float32))
results = db.search(np.random.rand(768).astype(np.float32), k=3)
print(db.get_text(results[0][0]))  # "Paris is the capital of France"
```

---

## What's New in v0.9.0

- **SIMD distance (AVX2):** SIMD-accelerated distance computation on supported platforms to improve search throughput.
- **Multiple metrics:** `cosine` and `l2` both supported and validated.
- **Batch insert:** `add_batch` supports bulk inserts (example usage: insert 100 nodes in one call).
- **Metadata filtering:** Search with `where={'color':'red'}` filters results by metadata tags.
- **Delete / Update:** Tombstone + re-insert workflow; deletes decrement the count.
- **Count / Iteration:** `len(db)`, `db.keys()`, `db.items()`, `get_all_metadata(offset)` supported.
- **Windows support:** Native Windows wheels produced; CI includes Windows builds.

## API Highlights

- `AgentKV(path, size_mb=100, dim=1536, metric="cosine")` — create/open DB
- `add(text, vector, relations=None, metadata=None) -> offset`
- `add_batch(contents, vectors, metadatas=None) -> List[offsets]`
- `search(query_vector, k=5, ef_search=50, where=None)` — `where` is metadata filter
- `delete(offset)`, `update(old_offset, content, vector, metadata=None)`
- `len(db)`, `db.keys()`, `db.items()`, `get_vector(offset)`, `get_text(offset)`, `get_all_metadata(offset)`
- Context helpers: `observe(node_id)`, `build_context(node_offsets, max_tokens=2048)`

## Example (using new features)

```python
import numpy as np
from agentkv import AgentKV

db = AgentKV("demo.db", size_mb=10, dim=128, metric="l2")

# Batch insert 100 random vectors with metadata
texts = [f"item-{i}" for i in range(100)]
vecs = np.random.rand(100, 128).astype(np.float32)
metas = [{"color": "red" if i % 2 == 0 else "blue"} for i in range(100)]
offsets = db.add_batch(texts, vecs, metadatas=metas)

# Search only red items
q = np.random.rand(128).astype(np.float32)
results = db.search(q, k=5, where={"color": "red"})

# Delete + update
db.delete(offsets[0])
new_offset = db.update(offsets[1], "updated text", vecs[1], metadata={"color":"green"})

print(len(db), db.keys(), db.get_all_metadata(new_offset))
```

---

## Why AgentKV?

| Problem | AgentKV Solution |
|---------|-----------------|
| Vector DBs need a server (Qdrant, Milvus) | **Single file**, no Docker, no network |
| FAISS has no persistence or text storage | **mmap persistence** + string arena + graph edges |
| RAG retrieves disjointed facts | **Graph + vector** for episodic continuity |
| Python GIL blocks concurrent search | **C++ core**, GIL released during search |
| Agent memory is stateless between runs | **Persistent** — memories survive restarts |

### Comparison

| Feature | AgentKV | FAISS | Chroma | Qdrant |
|---------|---------|-------|--------|--------|
| `pip install` | Yes | Yes | Yes | No (server) |
| Persistence | mmap | Manual | SQLite | Server |
| Text storage | Built-in | No | Yes | Yes |
| Graph edges | Yes | No | No | No |
| Crash recovery | CRC + rollback | No | Partial | Yes |
| GIL-free search | Yes | Yes | No | N/A |
| Zero-copy vectors | mmap to NumPy | Yes | No | No |

---

## Installation

```bash
pip install agentkv                    # core only
pip install agentkv[ollama]            # + Ollama embeddings/chat
pip install agentkv[all]               # + Ollama + web search
```

From source:
```bash
git clone https://github.com/DarkMatterCompiler/agentkv.git
cd agentkv && pip install -e ".[dev]"
```

Requires: Python 3.9+, C++20 compiler, CMake 3.15+

---

## Performance

| Metric | Result | Config |
|--------|--------|--------|
| Insert | 292 us/node | 768-dim, HNSW |
| Search | 130 us | k=5, ef=50 |
| Recall@5 | 98.4% | 500 nodes |
| Recall@5 | 91.2% | 2000 nodes |
| Persistence | 0.00% delta | Close + reopen |
| Concurrency | 100W + 387R / 0.10s | 1 writer, 4 readers |

---

## API

```python
from agentkv import AgentKV

db = AgentKV(path, size_mb=100, dim=768)   # Create or open
offset = db.add(text, vector)               # Store + auto-index
results = db.search(query_vec, k=5)         # K-NN search -> [(offset, dist)]
text = db.get_text(offset)                  # Retrieve text
vec = db.get_vector(offset)                 # Zero-copy NumPy view
context = db.observe(offset)                # Predict related context
```

---

## Examples

See the [examples/](examples/) directory:
- `local_rag.py` — Offline RAG with Ollama embeddings
- `agent_memory.py` — Persistent memory across restarts
- `chatbot.py` — Interactive CLI with web search

---

## Architecture

```
Python Agent / LangGraph
        |
  agentkv.AgentKV  (High-level Python API)
        |
  agentkv_core  (nanobind C++ extension — zero-copy, GIL-free)
        |
  C++ Engine
  +-- mmap Storage (single-file persistence, string arena for text + metadata)
  +-- Metadata Store (key/value tags, efficient multi-key filters)
  +-- HNSW Index (vector search) — SIMD-accelerated distance kernels (SSE/AVX2)
  +-- Distance Metrics: Cosine, L2, Inner-Product (selectable)
  +-- Batch Insert / WriteGuard (atomic bulk insert; serialized writers)
  +-- Property Graph (directed edges) + SLB (predictive context)
  +-- Tombstone + Update model (delete=tombstone + re-insert)
  +-- Crash Recovery (CRC checksums, header rollback)
  +-- Thread-safety (GIL released during search; reader-friendly; concurrent writers serialized)
  +-- Platform helpers (`mmap_platform.h`, `simd.h`) — cross-platform (Linux/macOS/Windows)
```

---

## License

MIT — see [LICENSE](LICENSE).

Built with [nanobind](https://github.com/wjakob/nanobind) + [scikit-build-core](https://github.com/scikit-build/scikit-build-core).
HNSW algorithm: [Malkov & Yashunin (2018)](https://arxiv.org/abs/1603.09320).
