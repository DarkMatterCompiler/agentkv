#pragma once
#include "storage.h"
#include <string>
#include <vector>
#include <random>
#include <utility>

class KVEngine {
private:
    int fd;
    void* map_base;
    StorageHeader* header;
    std::string filepath;
    std::mt19937 rng;

    // Internal Helpers
    template <typename T> T* get_ptr(uint64_t offset);
    uint64_t allocate(uint64_t size, uint64_t alignment);
    int get_random_level();

    // HNSW Internal
    // Returns raw dot product (higher = more similar)
    float dist_dot(const float* a, const float* b, uint32_t dim);
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

public:
    KVEngine(const std::string& path, size_t size_bytes);
    ~KVEngine();

    // Core API
    uint64_t create_node(uint64_t id, const std::vector<float>& embedding,
                         const std::string& text = "");
    void add_edge(uint64_t node_offset, uint64_t target_id, float weight);
    void print_node(uint64_t offset);

    // String Arena API
    uint64_t store_text(const std::string& text);
    // Returns {pointer_to_char_data, length}. Pointer into mmap.
    std::pair<const char*, uint32_t> get_text_raw(uint64_t node_offset);

    // HNSW API
    void init_hnsw(uint64_t node_offset);
    void add_hnsw_link(uint64_t node_offset, int layer, uint64_t target_offset, float dist);
    // K-NN search: returns vector of (node_offset, distance) sorted ascending by distance
    std::vector<std::pair<uint64_t, float>> search_knn(
        const float* query, uint32_t dim, int k, int ef_search = 50);
    // Dynamic HNSW insertion: creates node, assigns level, wires bidirectional links.
    // Returns the node offset.
    uint64_t insert(uint64_t id, const std::vector<float>& embedding,
                    const std::string& text = "");

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
    return reinterpret_cast<T*>((char*)map_base + offset);
}