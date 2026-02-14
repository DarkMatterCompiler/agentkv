#include "engine.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <cmath>

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------
KVEngine::KVEngine(const std::string& path, size_t size_bytes) : filepath(path) {
    fd = open(path.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd == -1) throw std::runtime_error("Failed to open file");
    if (ftruncate(fd, size_bytes) == -1) throw std::runtime_error("Resize failed");
    map_base = mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map_base == MAP_FAILED) throw std::runtime_error("mmap failed");

    header = static_cast<StorageHeader*>(map_base);
    
    // Initialize RNG
    std::random_device rd;
    rng = std::mt19937(rd());
    
    if (header->magic != 0x41474B56) {
        header->magic = 0x41474B56;
        header->write_head = sizeof(StorageHeader); 
        header->capacity = size_bytes;
        
        // Initialize HNSW Defaults
        header->hnsw_entry_point = NULL_OFFSET;
        header->hnsw_max_level = -1;
        header->M_max = 16;           // Default 16 neighbors
        header->ef_construction = 100;
    }
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
KVEngine::~KVEngine() {
    munmap(map_base, header->capacity);
    close(fd);
}

// -----------------------------------------------------------------------------
// Allocator (With Alignment)
// -----------------------------------------------------------------------------
uint64_t KVEngine::allocate(uint64_t size, uint64_t alignment) {
    uint64_t old_head, new_head, aligned_offset;
    do {
        old_head = header->write_head.load();
        aligned_offset = (old_head + alignment - 1) & ~(alignment - 1);
        new_head = aligned_offset + size;

        if (new_head > header->capacity) throw std::runtime_error("Disk Full: Increase file size");
    } while (!header->write_head.compare_exchange_weak(old_head, new_head));
    return aligned_offset;
}

// -----------------------------------------------------------------------------
// API: Create Node with Vector
// -----------------------------------------------------------------------------
uint64_t KVEngine::create_node(uint64_t id, const std::vector<float>& embedding) {
    // 1. Write the Vector first (Data-Oriented)
    uint32_t dim = embedding.size();
    uint64_t vec_size = sizeof(VectorBlock) + (dim * sizeof(float));
    uint64_t vec_offset = allocate(vec_size, 64); // 64-byte align for AVX

    VectorBlock* vb = get_ptr<VectorBlock>(vec_offset);
    vb->dim = dim;
    std::memcpy(vb->data, embedding.data(), dim * sizeof(float));

    // 2. Allocate the Node
    uint64_t node_offset = allocate(sizeof(Node), 64);
    Node* node = get_ptr<Node>(node_offset);
    
    node->id = id;
    node->timestamp = 0; // TODO: Realtime clock
    node->vector_head = vec_offset;
    node->edge_list_head = NULL_OFFSET;
    node->hnsw_head = NULL_OFFSET; // Must init to avoid garbage pointer
    node->version_lock = 0;

    return node_offset;
}

// -----------------------------------------------------------------------------
// API: Add Edge (Copy-On-Write)
// -----------------------------------------------------------------------------
void KVEngine::add_edge(uint64_t node_offset, uint64_t target_id, float weight) {
    Node* node = get_ptr<Node>(node_offset);
    
    // 1. Read old list
    uint64_t old_list_offset = node->edge_list_head.load();
    EdgeList* old_list = get_ptr<EdgeList>(old_list_offset);

    uint32_t old_count = (old_list) ? old_list->count : 0;
    uint32_t new_capacity = (old_count == 0) ? 4 : old_count * 2; // Geometric growth

    // 2. Allocate NEW list
    uint64_t new_size = sizeof(EdgeList) + (new_capacity * sizeof(Edge));
    uint64_t new_list_offset = allocate(new_size, 8);
    EdgeList* new_list = get_ptr<EdgeList>(new_list_offset);

    // 3. Copy old data
    if (old_list) {
        std::memcpy(new_list->edges, old_list->edges, old_count * sizeof(Edge));
    }

    // 4. Append new edge
    new_list->count = old_count + 1;
    new_list->capacity = new_capacity;
    new_list->edges[old_count] = {target_id, weight, 1}; // Type 1 = Generic

    // 5. ATOMIC SWAP (The magic moment)
    // This is safe because new_list is not visible to anyone yet.
    node->edge_list_head.store(new_list_offset);
}

// -----------------------------------------------------------------------------
// Debug Helper
// -----------------------------------------------------------------------------
void KVEngine::print_node(uint64_t offset) {
    Node* n = get_ptr<Node>(offset);
    std::cout << "Node ID: " << n->id << "\n";
    
    if (n->vector_head) {
        VectorBlock* vb = get_ptr<VectorBlock>(n->vector_head);
        std::cout << "  Vector [" << vb->dim << "]: [" << vb->data[0] << ", " << vb->data[1] << "...]\n";
    }

    if (n->edge_list_head) {
        EdgeList* el = get_ptr<EdgeList>(n->edge_list_head);
        std::cout << "  Edges (" << el->count << "/" << el->capacity << "):\n";
        for(uint32_t i=0; i<el->count; i++) {
            std::cout << "    -> Target: " << el->edges[i].target_node_id << " (w=" << el->edges[i].weight << ")\n";
        }
    }
}

// -----------------------------------------------------------------------------
// HNSW: Random Level Generator
// -----------------------------------------------------------------------------
int KVEngine::get_random_level() {
    double m_l = 1.0 / log(1.0 * header->M_max);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double r = -log(dis(rng)) * m_l;
    return (int)r;
}

// -----------------------------------------------------------------------------
// API: Initialize HNSW for a Node
// -----------------------------------------------------------------------------
void KVEngine::init_hnsw(uint64_t node_offset) {
    Node* node = get_ptr<Node>(node_offset);
    int level = get_random_level();

    // allocate HNSWNodeData
    // Size = struct + (level + 1) * sizeof(uint64_t)
    uint64_t size = sizeof(HNSWNodeData) + (level + 1) * sizeof(uint64_t);
    uint64_t hnsw_offset = allocate(size, 8); // 8-byte align is fine

    HNSWNodeData* h_data = get_ptr<HNSWNodeData>(hnsw_offset);
    h_data->level = level;
    
    // Initialize all layer offsets to NULL (Empty EdgeLists)
    for(int i=0; i<=level; i++) {
        h_data->layer_offsets[i] = NULL_OFFSET;
    }

    node->hnsw_head.store(hnsw_offset);

    // Update Global Max Level if we beat it
    int current_max = header->hnsw_max_level.load();
    if (level > current_max) {
        header->hnsw_max_level.store(level);
        header->hnsw_entry_point.store(node_offset);
    }
}

// -----------------------------------------------------------------------------
// API: Add HNSW Connection
// -----------------------------------------------------------------------------
void KVEngine::add_hnsw_link(uint64_t node_offset, int layer, uint64_t target_offset, float dist) {
    Node* node = get_ptr<Node>(node_offset);
    uint64_t h_offset = node->hnsw_head.load();
    if(h_offset == NULL_OFFSET) return; // Should not happen

    HNSWNodeData* h_data = get_ptr<HNSWNodeData>(h_offset);
    if(layer > h_data->level) return; // Invalid layer

    // 1. Get the EdgeList for this layer
    uint64_t list_offset = h_data->layer_offsets[layer];
    
    // 2. Reuse our existing Copy-On-Write Logic!
    // We replicate the logic here to update the specific layer pointer
    EdgeList* old_list = get_ptr<EdgeList>(list_offset);
    uint32_t old_count = (old_list) ? old_list->count : 0;
    uint32_t new_capacity = (old_count == 0) ? header->M_max : old_count + 1; // Simple growth for now

    // Check M_max constraint (HNSW specific)
    if (old_count >= header->M_max) {
        // In a real implementation, we would "SelectNeighbors" (keep best K).
        // For this Step 1.4 MVP, we just stop adding. 
        // Phase 2 will implement the heuristic to replace the worst neighbor.
        return; 
    }

    uint64_t new_size = sizeof(EdgeList) + (new_capacity * sizeof(Edge));
    uint64_t new_list_offset = allocate(new_size, 8);
    EdgeList* new_list = get_ptr<EdgeList>(new_list_offset);

    if (old_list) {
        std::memcpy(new_list->edges, old_list->edges, old_count * sizeof(Edge));
    }

    // Add new link (HNSW usually stores distance, not weight, but we use the same field)
    new_list->count = old_count + 1;
    new_list->capacity = new_capacity;
    new_list->edges[old_count] = {target_offset, dist, 0}; // Type 0 = HNSW Link

    // 3. Update the pointer in the HNSW block
    // (Note: This is not atomic relative to other layers, but atomic for this layer)
    h_data->layer_offsets[layer] = new_list_offset;
}