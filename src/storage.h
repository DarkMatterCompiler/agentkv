#pragma once
#include <cstdint>
#include <atomic>

constexpr uint64_t PAGE_SIZE = 4096;
constexpr uint64_t CACHE_LINE = 64;
constexpr uint64_t NULL_OFFSET = 0;

// -----------------------------------------------------------------------------
// Global Header
// -----------------------------------------------------------------------------
struct alignas(PAGE_SIZE) StorageHeader {
    uint32_t magic = 0x41474B56; 
    uint32_t version = 2;        // v0.2
    std::atomic<uint64_t> write_head;
    std::atomic<uint64_t> capacity;
    std::atomic<uint32_t> active_readers;

    // --- HNSW Global State ---
    std::atomic<uint64_t> hnsw_entry_point; // Offset to the "Top" Node
    std::atomic<int32_t>  hnsw_max_level;   // Current highest layer
    uint32_t M_max;                         // Config: Max neighbors per layer
    uint32_t ef_construction;               // Config: Beam width during build

    // Padding Calculation:
    // 4+4+8+8+4 + 8+4+4+4 = 48 bytes used.
    // Padding to next PAGE_SIZE boundary.
    uint8_t _pad[PAGE_SIZE - 48]; 
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

static_assert(sizeof(StorageHeader) % PAGE_SIZE == 0, "Header must be page-aligned");
static_assert(sizeof(Node) == CACHE_LINE, "Node Size Mismatch");