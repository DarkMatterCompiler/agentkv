#pragma once
#include <cstdint>
#include <atomic>

constexpr uint64_t PAGE_SIZE = 4096;
constexpr uint64_t CACHE_LINE = 64;
constexpr uint64_t NULL_OFFSET = 0;
constexpr uint32_t AGENTKV_MAGIC = 0x41474B56;   // "AGKV"
constexpr uint32_t AGENTKV_VERSION = 3;           // v0.7 on-disk format

// --- Recovery flags (bitfield in StorageHeader::flags) ---
constexpr uint32_t FLAG_CLEAN_SHUTDOWN = 0x01;    // Set on close, cleared on open
constexpr uint32_t FLAG_WRITE_IN_PROGRESS = 0x02; // Set before mutation, cleared after

// -----------------------------------------------------------------------------
// Global Header — crash-recoverable
// -----------------------------------------------------------------------------
struct alignas(PAGE_SIZE) StorageHeader {
    uint32_t magic;              // offset 0  (4)
    uint32_t version;            // offset 4  (4)
    std::atomic<uint64_t> write_head;   // offset 8  (8)
    std::atomic<uint64_t> capacity;     // offset 16 (8)
    std::atomic<uint32_t> active_readers; // offset 24 (4)
    uint32_t _pad1;              // offset 28 (4) — explicit alignment for next uint64

    // --- HNSW Global State ---
    std::atomic<uint64_t> hnsw_entry_point; // offset 32 (8)
    std::atomic<int32_t>  hnsw_max_level;   // offset 40 (4)
    uint32_t M_max;                         // offset 44 (4)
    uint32_t ef_construction;               // offset 48 (4)

    // --- Crash Recovery Fields ---
    uint32_t flags;                         // offset 52 (4)
    uint32_t checksum;                      // offset 56 (4)
    uint32_t _pad2;                         // offset 60 (4) — alignment for uint64

    // Snapshot of last known-good state (written before each mutation)
    uint64_t safe_write_head;               // offset 64 (8)
    uint64_t safe_hnsw_entry_point;         // offset 72 (8)
    int32_t  safe_hnsw_max_level;           // offset 80 (4)
    uint32_t _recovery_pad;                 // offset 84 (4)

    // Total explicit fields: 88 bytes
    uint8_t _pad[PAGE_SIZE - 88];
};

// -----------------------------------------------------------------------------
// Vector & Edge Structures
// -----------------------------------------------------------------------------
struct alignas(64) VectorBlock {
    uint32_t dim;
    uint32_t _pad;
    float data[0]; 
};

struct Edge {
    uint64_t target_node_id;
    float weight;
    uint32_t type; 
};

struct EdgeList {
    uint32_t count;
    uint32_t capacity;
    Edge edges[0]; 
};

// -----------------------------------------------------------------------------
// String Arena
// -----------------------------------------------------------------------------
struct alignas(8) StringBlock {
    uint32_t length;        // Byte count (excluding null terminator)
    uint32_t _pad;          // Explicit padding for 8-byte alignment
    char data[0];           // Flexible array, null-terminated UTF-8
};

// -----------------------------------------------------------------------------
// HNSW Node Topology
// -----------------------------------------------------------------------------
// This struct sits separately from the Node to keep Node small.
// It contains offsets to EdgeLists for each layer.
struct HNSWNodeData {
    int32_t level;          // The max layer this node exists in
    uint32_t _pad;          // Alignment
    uint64_t layer_offsets[0]; // Flexible Array of offsets to EdgeLists
};

// -----------------------------------------------------------------------------
// The Node (v0.2)
// -----------------------------------------------------------------------------
struct alignas(CACHE_LINE) Node {
    uint64_t id;             
    uint64_t timestamp;      

    std::atomic<uint64_t> edge_list_head; // Semantic Graph (Knowledge)
    uint64_t vector_head;                 // Vector Data
    
    // HNSW Index Pointer — points to HNSWNodeData struct
    std::atomic<uint64_t> hnsw_head;      

    std::atomic<uint64_t> version_lock;

    // String Arena pointer — offset to StringBlock
    uint64_t text_offset;

    // Padding Calculation:
    // 8 (id) + 8 (ts) + 8 (edge) + 8 (vec) + 8 (hnsw) + 8 (lock) + 8 (text) = 56 bytes.
    // Need 64 bytes total.
    // Padding = 64 - 56 = 8 bytes.
    uint8_t _pad[8]; 
};

static_assert(sizeof(StorageHeader) == PAGE_SIZE, "Header must be exactly one page");
static_assert(sizeof(Node) == CACHE_LINE, "Node Size Mismatch");