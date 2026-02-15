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
  agentkv.AgentKV  (High-Level API)
        |
  agentkv_core.so  (nanobind, zero-copy, GIL-free)
        |
  C++ Engine
  +-- mmap Storage (single file)
  +-- HNSW Index (vector search)
  +-- Property Graph (relationships)
  +-- String Arena (text persistence)
  +-- Crash Recovery (CRC + rollback)
  +-- SLB (predictive context)
```

---

## License

MIT — see [LICENSE](LICENSE).

Built with [nanobind](https://github.com/wjakob/nanobind) + [scikit-build-core](https://github.com/scikit-build/scikit-build-core).
HNSW algorithm: [Malkov & Yashunin (2018)](https://arxiv.org/abs/1603.09320).
