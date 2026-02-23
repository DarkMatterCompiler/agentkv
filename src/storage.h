#pragma once
#include <cstdint>
#include <atomic>

constexpr uint64_t PAGE_SIZE = 4096;
constexpr uint64_t CACHE_LINE = 64;
constexpr uint64_t NULL_OFFSET = 0;
constexpr uint32_t AGENTKV_MAGIC = 0x41474B56;   // "AGKV"
constexpr uint32_t AGENTKV_VERSION = 4;           // v0.9 on-disk format

// --- Recovery flags (bitfield in StorageHeader::flags) ---
constexpr uint32_t FLAG_CLEAN_SHUTDOWN = 0x01;    // Set on close, cleared on open
constexpr uint32_t FLAG_WRITE_IN_PROGRESS = 0x02; // Set before mutation, cleared after

// --- Node flags (bitfield in Node::node_flags) ---
constexpr uint32_t NODE_DELETED = 0x01;           // Tombstone — skipped in search/iteration

// --- Distance Metrics ---
enum DistanceMetric : uint32_t {
    METRIC_COSINE         = 0, // 1 - dot(a,b)   (requires normalized vectors)
    METRIC_L2             = 1, // sum((a-b)^2)    (squared Euclidean)
    METRIC_INNER_PRODUCT  = 2, // -dot(a,b)       (higher dot = smaller distance)
};

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

    // --- v0.9 fields ---
    uint32_t metric;                        // offset 88 (4) — DistanceMetric enum
    uint32_t _pad3;                         // offset 92 (4)
    std::atomic<uint64_t> node_count;       // offset 96 (8) — total nodes created
    std::atomic<uint64_t> deleted_count;    // offset 104 (8) — tombstoned nodes
    std::atomic<uint64_t> first_node_offset;// offset 112 (8) — head of node linked list

    // Total explicit fields: 120 bytes
    uint8_t _pad[PAGE_SIZE - 120];
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
// Metadata Entry (linked list per node)
// Each entry stores one key=value string pair.
// Layout: [MetadataEntry header] [key bytes] [value bytes]
// Both key and value are NOT null-terminated; lengths are explicit.
// -----------------------------------------------------------------------------
struct alignas(8) MetadataEntry {
    uint64_t next;          // Offset to next MetadataEntry (or NULL_OFFSET)
    uint32_t key_len;       // Length of key in bytes
    uint32_t val_len;       // Length of value in bytes
    char data[0];           // key_len bytes of key, then val_len bytes of value
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
// The Node (v0.9)
// 64 bytes = 1 cache line.  Changes from v0.2:
//   timestamp   -> next_node_offset  (iteration linked list)
//   version_lock -> node_flags       (DELETED bit)
//   _pad[8]     -> metadata_offset   (key-value tags)
// -----------------------------------------------------------------------------
struct alignas(CACHE_LINE) Node {
    uint64_t id;                           // 8  (offset 0)
    uint64_t next_node_offset;             // 8  (offset 8)  — linked list for iteration

    std::atomic<uint64_t> edge_list_head;  // 8  (offset 16)
    uint64_t vector_head;                  // 8  (offset 24)
    
    // HNSW Index Pointer — points to HNSWNodeData struct
    std::atomic<uint64_t> hnsw_head;       // 8  (offset 32)

    std::atomic<uint32_t> node_flags;      // 4  (offset 40)  — NODE_DELETED bit
    uint32_t _node_pad;                    // 4  (offset 44)

    // String Arena pointer — offset to StringBlock
    uint64_t text_offset;                  // 8  (offset 48)

    // Metadata pointer — offset to first MetadataEntry (linked list)
    uint64_t metadata_offset;              // 8  (offset 56)
};

static_assert(sizeof(StorageHeader) == PAGE_SIZE, "Header must be exactly one page");
static_assert(sizeof(Node) == CACHE_LINE, "Node Size Mismatch");