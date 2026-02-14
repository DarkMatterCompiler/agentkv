#pragma once
#include "storage.h"
#include <string>
#include <vector>
#include <random>

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

public:
    KVEngine(const std::string& path, size_t size_bytes);
    ~KVEngine();

    // Core API
    uint64_t create_node(uint64_t id, const std::vector<float>& embedding);
    void add_edge(uint64_t node_offset, uint64_t target_id, float weight);
    void print_node(uint64_t offset);

    // HNSW API
    void init_hnsw(uint64_t node_offset);
    void add_hnsw_link(uint64_t node_offset, int layer, uint64_t target_offset, float dist);

    // --- NEW: Friend Class for SLB ---
    // The ContextManager needs raw access to crawl the graph fast.
    friend class ContextManager;
    // In src/engine.h, inside public:

    // --- Binding Helper: Zero-Copy Access ---
    // Returns {pointer_to_float_array, dimension}
    // The pointer points DIRECTLY into the mmap file.
    std::pair<float*, size_t> get_vector_raw(uint64_t node_offset) {
        Node* node = get_ptr<Node>(node_offset);
        if (!node || node->vector_head == NULL_OFFSET) return {nullptr, 0};
        
        VectorBlock* vb = get_ptr<VectorBlock>(node->vector_head);
        return {vb->data, vb->dim};
    }
};

// Add this at the bottom of engine.h
template <typename T>
T* KVEngine::get_ptr(uint64_t offset) {
    if (offset == NULL_OFFSET) return nullptr;
    // Note: We skip bounds check for speed in release, or keep it for safety
    return reinterpret_cast<T*>((char*)map_base + offset);
}