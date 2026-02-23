#pragma once
#include "storage.h"
#include "mmap_platform.h"
#include "simd.h"
#include <string>
#include <vector>
#include <random>
#include <utility>
#include <shared_mutex>
#include <cstdint>
#include <cstddef>

// Simple CRC-32 (ISO 3309 polynomial)
inline uint32_t crc32_compute(const void* data, size_t length) {
    const uint8_t* buf = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; ++i) {
        crc ^= buf[i];
        for (int j = 0; j < 8; ++j)
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
    }
    return ~crc;
}

// Metadata filter for search: exact match on key=value
struct MetadataFilter {
    std::string key;
    std::string value;
};

// RAII Read Lock Guard
class ReadGuard {
    std::shared_mutex& mtx_;
public:
    explicit ReadGuard(std::shared_mutex& m) : mtx_(m) { mtx_.lock_shared(); }
    ~ReadGuard() { mtx_.unlock_shared(); }
    ReadGuard(const ReadGuard&) = delete;
    ReadGuard& operator=(const ReadGuard&) = delete;
};

// RAII Write Lock Guard
class WriteGuard {
    std::shared_mutex& mtx_;
public:
    explicit WriteGuard(std::shared_mutex& m) : mtx_(m) { mtx_.lock(); }
    ~WriteGuard() { mtx_.unlock(); }
    WriteGuard(const WriteGuard&) = delete;
    WriteGuard& operator=(const WriteGuard&) = delete;
};

class KVEngine {
private:
    platform::MMapHandle mmap_;
    StorageHeader* header;
    std::string filepath;
    std::mt19937 rng;
    mutable std::shared_mutex rwlock_;  // 1 writer + N readers

    // Internal Helpers
    template <typename T> T* get_ptr(uint64_t offset);
    uint64_t allocate(uint64_t size, uint64_t alignment);
    int get_random_level();

    // Crash Recovery Helpers
    void checkpoint_header();      // Save known-good state before mutation
    void update_header_checksum(); // Recompute CRC-32 over header
    bool validate_header();        // Check magic + checksum
    void recover_if_needed();      // Rollback to safe state on dirty open
    bool validate_header_const() const {
        if (header->magic != AGENTKV_MAGIC) return false;
        size_t checksum_offset = offsetof(StorageHeader, checksum);
        uint32_t expected = crc32_compute(header, checksum_offset);
        return header->checksum == expected;
    }

    // HNSW Internal
    // Compute distance using the DB's configured metric + SIMD
    float compute_dist(const float* a, const float* b, uint32_t dim);
    // Search within a single HNSW layer. Returns up to ef nearest neighbors.
    // result: vector of (distance, node_offset) — sorted ascending by distance.
    std::vector<std::pair<float, uint64_t>> search_layer(
        const float* query, uint32_t dim,
        uint64_t entry_offset, int layer, int ef);
    // Prune a neighbor list to keep only the best M_max by distance (ascending)
    void prune_neighbors(uint64_t node_offset, int layer);
    // Select up to M best neighbors from candidates (simple heuristic: closest M)
    std::vector<std::pair<float, uint64_t>> select_neighbors(
        const std::vector<std::pair<float, uint64_t>>& candidates, int M);
    // Check if a node passes all metadata filters
    bool passes_filters(uint64_t node_offset,
                        const std::vector<MetadataFilter>& filters);

    // Node linked-list helper (prepend to list, update header)
    void link_node(uint64_t node_offset);

public:
    KVEngine(const std::string& path, size_t size_bytes,
             DistanceMetric metric = METRIC_COSINE);
    ~KVEngine();

    // Core API
    uint64_t create_node(uint64_t id, const std::vector<float>& embedding,
                         const std::string& text = "");
    void add_edge(uint64_t node_offset, uint64_t target_id, float weight);
    void print_node(uint64_t offset);

    // Crash Recovery API
    bool is_valid() const { return validate_header_const(); }
    void sync() { platform::mmap_sync(mmap_, header->capacity); }

    // String Arena API
    uint64_t store_text(const std::string& text);
    // Returns {pointer_to_char_data, length}. Pointer into mmap.
    std::pair<const char*, uint32_t> get_text_raw(uint64_t node_offset);

    // HNSW API
    void init_hnsw(uint64_t node_offset);
    void add_hnsw_link(uint64_t node_offset, int layer, uint64_t target_offset, float dist);
    // K-NN search: returns vector of (node_offset, distance) sorted ascending by distance
    // Thread-safe: acquires read lock.
    std::vector<std::pair<uint64_t, float>> search_knn(
        const float* query, uint32_t dim, int k, int ef_search = 50);
    // Filtered K-NN search: same as search_knn but skips nodes failing metadata filters.
    std::vector<std::pair<uint64_t, float>> search_knn_filtered(
        const float* query, uint32_t dim, int k, int ef_search,
        const std::vector<MetadataFilter>& filters);
    // Dynamic HNSW insertion: creates node, assigns level, wires bidirectional links.
    // Thread-safe: acquires write lock. Returns the node offset.
    uint64_t insert(uint64_t id, const std::vector<float>& embedding,
                    const std::string& text = "");
    // Batch insert: insert N vectors in a single lock acquisition.
    // Returns vector of node offsets. Thread-safe.
    std::vector<uint64_t> insert_batch(
        const uint64_t* ids, const float* data, uint32_t n, uint32_t dim,
        const std::vector<std::string>& texts);

    // Delete / Update API
    // Tombstone a node — it will be skipped in search and iteration.
    void delete_node(uint64_t node_offset);
    bool is_deleted(uint64_t node_offset);
    // Update = tombstone old + insert new. Returns new node offset.
    uint64_t update_node(uint64_t old_offset, uint64_t new_id,
                         const std::vector<float>& embedding,
                         const std::string& text = "");

    // Metadata API
    void set_metadata(uint64_t node_offset, const std::string& key,
                      const std::string& value);
    std::string get_metadata(uint64_t node_offset, const std::string& key);
    std::vector<std::pair<std::string, std::string>>
        get_all_metadata(uint64_t node_offset);

    // Count / Iteration API
    uint64_t count() const;       // live nodes (total - deleted)
    uint64_t total_count() const; // total nodes ever created
    // Walk the node linked list, skipping deleted. Returns offsets.
    std::vector<uint64_t> get_all_node_offsets();
    // Get the configured distance metric
    uint32_t get_metric() const { return header->metric; }

    // --- Friend Class for SLB ---
    friend class ContextManager;

    // --- Binding Helpers: Zero-Copy Access ---
    std::pair<float*, size_t> get_vector_raw(uint64_t node_offset) {
        Node* node = get_ptr<Node>(node_offset);
        if (!node || node->vector_head == NULL_OFFSET) return {nullptr, 0};
        VectorBlock* vb = get_ptr<VectorBlock>(node->vector_head);
        return {vb->data, vb->dim};
    }
};

// Template definition must be in header
template <typename T>
T* KVEngine::get_ptr(uint64_t offset) {
    if (offset == NULL_OFFSET) return nullptr;
    return reinterpret_cast<T*>((char*)mmap_.ptr + offset);
}