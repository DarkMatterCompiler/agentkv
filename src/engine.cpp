#include "engine.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <queue>
#include <unordered_set>

// =============================================================================
// Constructor — with crash recovery
// =============================================================================
KVEngine::KVEngine(const std::string& path, size_t size_bytes) : filepath(path) {
    fd = open(path.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd == -1) throw std::runtime_error("Failed to open file: " + path);
    if (ftruncate(fd, size_bytes) == -1) throw std::runtime_error("Resize failed");
    map_base = mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map_base == MAP_FAILED) throw std::runtime_error("mmap failed");

    header = static_cast<StorageHeader*>(map_base);
    
    // Initialize RNG
    std::random_device rd;
    rng = std::mt19937(rd());
    
    if (header->magic != AGENTKV_MAGIC) {
        // --- Fresh database: zero header then initialize ---
        std::memset(header, 0, sizeof(StorageHeader));
        header->magic = AGENTKV_MAGIC;
        header->version = AGENTKV_VERSION;
        header->write_head = sizeof(StorageHeader); 
        header->capacity = size_bytes;
        
        // HNSW defaults
        header->hnsw_entry_point = NULL_OFFSET;
        header->hnsw_max_level = -1;
        header->M_max = 16;
        header->ef_construction = 100;

        // Recovery fields
        header->flags = FLAG_CLEAN_SHUTDOWN;
        header->safe_write_head = sizeof(StorageHeader);
        header->safe_hnsw_entry_point = NULL_OFFSET;
        header->safe_hnsw_max_level = -1;
        header->_recovery_pad = 0;
        
        update_header_checksum();
        msync(map_base, PAGE_SIZE, MS_SYNC);
    } else {
        // --- Existing database: check for crash recovery ---
        recover_if_needed();
    }

    // Mark as "open / not cleanly shut down"
    header->flags &= ~FLAG_CLEAN_SHUTDOWN;
    update_header_checksum();
    msync(map_base, PAGE_SIZE, MS_SYNC);
}

// =============================================================================
// Destructor — clean shutdown with sync
// =============================================================================
KVEngine::~KVEngine() {
    // Mark clean shutdown
    header->flags |= FLAG_CLEAN_SHUTDOWN;
    header->flags &= ~FLAG_WRITE_IN_PROGRESS;
    checkpoint_header();
    update_header_checksum();
    msync(map_base, header->capacity, MS_SYNC);
    munmap(map_base, header->capacity);
    close(fd);
}

// =============================================================================
// Crash Recovery: Checkpoint — save known-good state before mutation
// =============================================================================
void KVEngine::checkpoint_header() {
    header->safe_write_head = header->write_head.load();
    header->safe_hnsw_entry_point = header->hnsw_entry_point.load();
    header->safe_hnsw_max_level = header->hnsw_max_level.load();
}

// =============================================================================
// Crash Recovery: CRC-32 checksum over header (excluding checksum + pad)
// =============================================================================
void KVEngine::update_header_checksum() {
    // Checksum covers bytes [0, offset_of_checksum)
    // checksum field is at offset 52 (after flags at 48 + 4 bytes)
    size_t checksum_offset = offsetof(StorageHeader, checksum);
    header->checksum = crc32_compute(header, checksum_offset);
}

// =============================================================================
// Crash Recovery: Validate header integrity
// =============================================================================
bool KVEngine::validate_header() {
    if (header->magic != AGENTKV_MAGIC) return false;
    size_t checksum_offset = offsetof(StorageHeader, checksum);
    uint32_t expected = crc32_compute(header, checksum_offset);
    return header->checksum == expected;
}

// =============================================================================
// Crash Recovery: Rollback to last-known-good state if dirty
// =============================================================================
void KVEngine::recover_if_needed() {
    bool clean = (header->flags & FLAG_CLEAN_SHUTDOWN) != 0;
    bool checksum_ok = validate_header();

    if (clean && checksum_ok) {
        return; // Normal reopen — nothing to do
    }

    // Dirty open or corrupted checksum: rollback to safe state
    if (header->safe_write_head >= sizeof(StorageHeader) &&
        header->safe_write_head <= header->capacity.load()) {
        header->write_head.store(header->safe_write_head);
    }
    if (header->safe_hnsw_entry_point != NULL_OFFSET) {
        header->hnsw_entry_point.store(header->safe_hnsw_entry_point);
        header->hnsw_max_level.store(header->safe_hnsw_max_level);
    }

    header->flags &= ~FLAG_WRITE_IN_PROGRESS;
    update_header_checksum();
    msync(map_base, PAGE_SIZE, MS_SYNC);
}

// =============================================================================
// Allocator (Lock-Free CAS Bump)
// =============================================================================
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

// =============================================================================
// String Arena: Store Text
// =============================================================================
uint64_t KVEngine::store_text(const std::string& text) {
    if (text.empty()) return NULL_OFFSET;
    
    uint32_t len = static_cast<uint32_t>(text.size());
    // StringBlock header (8 bytes) + text bytes + null terminator
    uint64_t block_size = sizeof(StringBlock) + len + 1;
    uint64_t offset = allocate(block_size, 8);
    
    StringBlock* sb = get_ptr<StringBlock>(offset);
    sb->length = len;
    sb->_pad = 0;
    std::memcpy(sb->data, text.c_str(), len + 1); // include null terminator
    
    return offset;
}

// =============================================================================
// String Arena: Read Text (Zero-Copy)
// =============================================================================
std::pair<const char*, uint32_t> KVEngine::get_text_raw(uint64_t node_offset) {
    Node* node = get_ptr<Node>(node_offset);
    if (!node || node->text_offset == NULL_OFFSET) return {nullptr, 0};
    
    StringBlock* sb = get_ptr<StringBlock>(node->text_offset);
    return {sb->data, sb->length};
}

// =============================================================================
// API: Create Node with Vector + Text
// =============================================================================
uint64_t KVEngine::create_node(uint64_t id, const std::vector<float>& embedding,
                               const std::string& text) {
    // 1. Write the Vector (64-byte aligned for AVX)
    uint32_t dim = embedding.size();
    uint64_t vec_size = sizeof(VectorBlock) + (dim * sizeof(float));
    uint64_t vec_offset = allocate(vec_size, 64);

    VectorBlock* vb = get_ptr<VectorBlock>(vec_offset);
    vb->dim = dim;
    std::memcpy(vb->data, embedding.data(), dim * sizeof(float));

    // 2. Write the Text (if provided)
    uint64_t txt_offset = store_text(text);

    // 3. Allocate the Node
    uint64_t node_offset = allocate(sizeof(Node), 64);
    Node* node = get_ptr<Node>(node_offset);
    
    node->id = id;
    node->timestamp = 0; // TODO: Realtime clock
    node->vector_head = vec_offset;
    node->text_offset = txt_offset;
    node->edge_list_head = NULL_OFFSET;
    node->hnsw_head = NULL_OFFSET;
    node->version_lock = 0;

    return node_offset;
}

// =============================================================================
// API: Add Edge (Copy-On-Write)
// =============================================================================
void KVEngine::add_edge(uint64_t node_offset, uint64_t target_id, float weight) {
    Node* node = get_ptr<Node>(node_offset);
    
    uint64_t old_list_offset = node->edge_list_head.load();
    EdgeList* old_list = get_ptr<EdgeList>(old_list_offset);

    uint32_t old_count = (old_list) ? old_list->count : 0;
    uint32_t new_capacity = (old_count == 0) ? 4 : old_count * 2;

    uint64_t new_size = sizeof(EdgeList) + (new_capacity * sizeof(Edge));
    uint64_t new_list_offset = allocate(new_size, 8);
    EdgeList* new_list = get_ptr<EdgeList>(new_list_offset);

    if (old_list) {
        std::memcpy(new_list->edges, old_list->edges, old_count * sizeof(Edge));
    }

    new_list->count = old_count + 1;
    new_list->capacity = new_capacity;
    new_list->edges[old_count] = {target_id, weight, 1};

    node->edge_list_head.store(new_list_offset);
}

// =============================================================================
// Debug Helper
// =============================================================================
void KVEngine::print_node(uint64_t offset) {
    Node* n = get_ptr<Node>(offset);
    std::cout << "Node ID: " << n->id << "\n";
    
    if (n->vector_head) {
        VectorBlock* vb = get_ptr<VectorBlock>(n->vector_head);
        std::cout << "  Vector [" << vb->dim << "]: [" << vb->data[0] << ", " << vb->data[1] << "...]\n";
    }

    if (n->text_offset != NULL_OFFSET) {
        StringBlock* sb = get_ptr<StringBlock>(n->text_offset);
        std::cout << "  Text [" << sb->length << "]: \"" << sb->data << "\"\n";
    }

    if (n->edge_list_head) {
        EdgeList* el = get_ptr<EdgeList>(n->edge_list_head);
        std::cout << "  Edges (" << el->count << "/" << el->capacity << "):\n";
        for(uint32_t i=0; i<el->count; i++) {
            std::cout << "    -> Target: " << el->edges[i].target_node_id << " (w=" << el->edges[i].weight << ")\n";
        }
    }
}

// =============================================================================
// HNSW: Dot Product (auto-vectorizable with -march=native -O3)
// =============================================================================
float KVEngine::dist_dot(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    // Compiler will auto-vectorize this loop with SSE/AVX
    for (uint32_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// =============================================================================
// HNSW: Random Level Generator
// =============================================================================
int KVEngine::get_random_level() {
    double m_l = 1.0 / log(1.0 * header->M_max);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double r = -log(dis(rng)) * m_l;
    return (int)r;
}

// =============================================================================
// HNSW: Initialize Node in the Index
// =============================================================================
void KVEngine::init_hnsw(uint64_t node_offset) {
    Node* node = get_ptr<Node>(node_offset);
    int level = get_random_level();

    uint64_t size = sizeof(HNSWNodeData) + (level + 1) * sizeof(uint64_t);
    uint64_t hnsw_offset = allocate(size, 8);

    HNSWNodeData* h_data = get_ptr<HNSWNodeData>(hnsw_offset);
    h_data->level = level;
    
    for(int i = 0; i <= level; i++) {
        h_data->layer_offsets[i] = NULL_OFFSET;
    }

    node->hnsw_head.store(hnsw_offset);

    int current_max = header->hnsw_max_level.load();
    if (level > current_max) {
        header->hnsw_max_level.store(level);
        header->hnsw_entry_point.store(node_offset);
    }
}

// =============================================================================
// HNSW: Add Link + Prune (with "Keep Best M" enforcement)
// =============================================================================
void KVEngine::add_hnsw_link(uint64_t node_offset, int layer, uint64_t target_offset, float dist) {
    Node* node = get_ptr<Node>(node_offset);
    uint64_t h_offset = node->hnsw_head.load();
    if(h_offset == NULL_OFFSET) return;

    HNSWNodeData* h_data = get_ptr<HNSWNodeData>(h_offset);
    if(layer > h_data->level) return;

    // Get the EdgeList for this layer
    uint64_t list_offset = h_data->layer_offsets[layer];
    EdgeList* old_list = get_ptr<EdgeList>(list_offset);
    uint32_t old_count = (old_list) ? old_list->count : 0;
    uint32_t M = header->M_max;

    // --- Case 1: Node is FULL (count == M_max) ---
    // Zero-Alloc Pruning: overwrite the worst neighbor in-place if new is closer.
    if (old_count >= M) {
        // Scan for the worst (highest distance) neighbor
        uint32_t worst_idx = 0;
        float worst_dist = old_list->edges[0].weight;
        for (uint32_t i = 1; i < old_count; ++i) {
            if (old_list->edges[i].weight > worst_dist) {
                worst_dist = old_list->edges[i].weight;
                worst_idx = i;
            }
        }
        // Only replace if the new edge is strictly closer
        if (dist < worst_dist) {
            old_list->edges[worst_idx] = {target_offset, dist, 0};
        }
        // Either way, no allocation. Done.
        return;
    }

    // --- Case 2: Node has room (count < M_max) ---
    // COW append: allocate new list, copy old, append new edge.
    uint32_t new_count = old_count + 1;
    uint32_t new_capacity = (old_count == 0) ? M : old_list->capacity;
    if (new_capacity < new_count) new_capacity = M;

    uint64_t new_size = sizeof(EdgeList) + (new_capacity * sizeof(Edge));
    uint64_t new_list_offset = allocate(new_size, 8);
    EdgeList* new_list = get_ptr<EdgeList>(new_list_offset);

    if (old_list) {
        std::memcpy(new_list->edges, old_list->edges, old_count * sizeof(Edge));
    }

    new_list->edges[old_count] = {target_offset, dist, 0}; // Type 0 = HNSW Link
    new_list->count = new_count;
    new_list->capacity = new_capacity;

    h_data->layer_offsets[layer] = new_list_offset;
}

// =============================================================================
// HNSW: Prune Neighbors — Keep Best M_max by distance (ascending)
// (Legacy: kept for backward compat, but add_hnsw_link now self-prunes)
// =============================================================================
void KVEngine::prune_neighbors(uint64_t node_offset, int layer) {
    Node* node = get_ptr<Node>(node_offset);
    uint64_t h_offset = node->hnsw_head.load();
    HNSWNodeData* h_data = get_ptr<HNSWNodeData>(h_offset);
    
    uint64_t list_offset = h_data->layer_offsets[layer];
    EdgeList* list = get_ptr<EdgeList>(list_offset);
    if (!list || list->count <= header->M_max) return;

    // Sort edges by distance (weight field stores distance) — ascending
    std::vector<Edge> edges(list->edges, list->edges + list->count);
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.weight < b.weight; // lower distance = better
    });

    // Keep only M_max best
    uint32_t keep = header->M_max;
    uint64_t new_size = sizeof(EdgeList) + (keep * sizeof(Edge));
    uint64_t new_list_offset = allocate(new_size, 8);
    EdgeList* new_list = get_ptr<EdgeList>(new_list_offset);
    
    new_list->count = keep;
    new_list->capacity = keep;
    std::memcpy(new_list->edges, edges.data(), keep * sizeof(Edge));

    h_data->layer_offsets[layer] = new_list_offset;
}

// =============================================================================
// HNSW: Search Layer (Greedy BFS with beam width ef)
// Returns vector of (distance, node_offset) sorted ascending by distance
// =============================================================================
std::vector<std::pair<float, uint64_t>> KVEngine::search_layer(
    const float* query, uint32_t dim,
    uint64_t entry_offset, int layer, int ef) 
{
    // Min-heap: candidates to explore (smallest distance on top)
    // Max-heap: result set W (largest distance on top — for easy eviction)
    using Pair = std::pair<float, uint64_t>; // (distance, node_offset)

    // Compute distance for the entry point
    Node* entry_node = get_ptr<Node>(entry_offset);
    if (!entry_node || entry_node->vector_head == NULL_OFFSET) return {};
    VectorBlock* entry_vb = get_ptr<VectorBlock>(entry_node->vector_head);
    float entry_dist = 1.0f - dist_dot(query, entry_vb->data, dim);

    // candidates: min-heap (smallest dist on top)
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> candidates;
    // result: max-heap (largest dist on top)
    std::priority_queue<Pair> result;
    
    std::unordered_set<uint64_t> visited;

    candidates.push({entry_dist, entry_offset});
    result.push({entry_dist, entry_offset});
    visited.insert(entry_offset);

    while (!candidates.empty()) {
        auto [c_dist, c_offset] = candidates.top();
        candidates.pop();

        // If the closest candidate is farther than the farthest result, stop
        float f_dist = result.top().first;
        if (c_dist > f_dist) break;

        // Explore neighbors at this layer
        Node* c_node = get_ptr<Node>(c_offset);
        uint64_t h_offset = c_node->hnsw_head.load();
        if (h_offset == NULL_OFFSET) continue;

        HNSWNodeData* h_data = get_ptr<HNSWNodeData>(h_offset);
        if (layer > h_data->level) continue;

        uint64_t layer_list_offset = h_data->layer_offsets[layer];
        EdgeList* neighbors = get_ptr<EdgeList>(layer_list_offset);
        if (!neighbors) continue;

        for (uint32_t i = 0; i < neighbors->count; ++i) {
            uint64_t n_offset = neighbors->edges[i].target_node_id;
            
            if (visited.count(n_offset)) continue;
            visited.insert(n_offset);

            Node* n_node = get_ptr<Node>(n_offset);
            if (!n_node || n_node->vector_head == NULL_OFFSET) continue;

            VectorBlock* n_vb = get_ptr<VectorBlock>(n_node->vector_head);
            if (n_vb->dim != dim) continue; // Dimension mismatch guard
            float n_dist = 1.0f - dist_dot(query, n_vb->data, dim);

            f_dist = result.top().first;

            if (n_dist < f_dist || (int)result.size() < ef) {
                candidates.push({n_dist, n_offset});
                result.push({n_dist, n_offset});

                if ((int)result.size() > ef) {
                    result.pop(); // Evict the farthest
                }
            }
        }
    }

    // Drain max-heap into sorted vector (ascending distance)
    std::vector<Pair> results;
    results.reserve(result.size());
    while (!result.empty()) {
        results.push_back(result.top());
        result.pop();
    }
    // Reverse: max-heap pops largest first, we want ascending
    std::reverse(results.begin(), results.end());
    return results;
}

// =============================================================================
// HNSW: K-NN Search (Multi-Layer Descent)
// Returns vector of (node_offset, distance) sorted ascending by distance
// =============================================================================
std::vector<std::pair<uint64_t, float>> KVEngine::search_knn(
    const float* query, uint32_t dim, int k, int ef_search) 
{
    ReadGuard rg(rwlock_);  // N readers can search concurrently

    uint64_t ep = header->hnsw_entry_point.load();
    int max_level = header->hnsw_max_level.load();
    
    if (ep == NULL_OFFSET || max_level < 0) {
        return {}; // Empty index
    }

    // Phase 1: Greedy descent from top layer to layer 1
    // At each layer, find the single closest node (ef=1)
    uint64_t current_ep = ep;
    
    for (int lc = max_level; lc >= 1; --lc) {
        auto layer_result = search_layer(query, dim, current_ep, lc, 1);
        if (!layer_result.empty()) {
            current_ep = layer_result[0].second; // closest node becomes new EP
        }
    }

    // Phase 2: Full beam search at layer 0 with ef_search
    auto layer0_result = search_layer(query, dim, current_ep, 0, ef_search);

    // Take top-k results, convert to (offset, dist) format
    std::vector<std::pair<uint64_t, float>> knn;
    int count = std::min(k, (int)layer0_result.size());
    knn.reserve(count);
    for (int i = 0; i < count; ++i) {
        knn.push_back({layer0_result[i].second, layer0_result[i].first});
    }
    return knn;
}

// =============================================================================
// HNSW: Select Neighbors — Simple heuristic (pick M closest)
// Input: candidates sorted ascending by distance. Returns at most M closest.
// =============================================================================
std::vector<std::pair<float, uint64_t>> KVEngine::select_neighbors(
    const std::vector<std::pair<float, uint64_t>>& candidates, int M)
{
    // candidates are already sorted ascending by distance from search_layer
    int count = std::min(M, (int)candidates.size());
    return std::vector<std::pair<float, uint64_t>>(
        candidates.begin(), candidates.begin() + count);
}

// =============================================================================
// HNSW: Dynamic Insert — Full HNSW insertion algorithm
// Creates node, assigns random level, wires bidirectional connections.
// =============================================================================
uint64_t KVEngine::insert(uint64_t id, const std::vector<float>& embedding,
                          const std::string& text) 
{
    WriteGuard wg(rwlock_);  // Exclusive access during mutation

    // --- Crash Recovery: checkpoint before mutation ---
    checkpoint_header();
    header->flags |= FLAG_WRITE_IN_PROGRESS;
    msync(map_base, PAGE_SIZE, MS_SYNC);

    // 1. Create the node (vector + text + node struct)
    uint64_t new_offset = create_node(id, embedding, text);
    
    // 2. Initialize HNSW data (assigns random level, allocates layer_offsets)
    Node* new_node = get_ptr<Node>(new_offset);
    int new_level = get_random_level();

    uint64_t hnsw_size = sizeof(HNSWNodeData) + (new_level + 1) * sizeof(uint64_t);
    uint64_t hnsw_offset = allocate(hnsw_size, 8);
    HNSWNodeData* new_hnsw = get_ptr<HNSWNodeData>(hnsw_offset);
    new_hnsw->level = new_level;
    for (int i = 0; i <= new_level; i++) {
        new_hnsw->layer_offsets[i] = NULL_OFFSET;
    }
    new_node->hnsw_head.store(hnsw_offset);

    // Get the new node's vector for distance calculations
    VectorBlock* new_vb = get_ptr<VectorBlock>(new_node->vector_head);
    const float* q = new_vb->data;
    uint32_t dim = new_vb->dim;

    // 3. Check if index is empty — first node becomes entry point
    uint64_t ep = header->hnsw_entry_point.load();
    int current_max_level = header->hnsw_max_level.load();

    if (ep == NULL_OFFSET || current_max_level < 0) {
        // First node in the index
        header->hnsw_entry_point.store(new_offset);
        header->hnsw_max_level.store(new_level);

        // --- Crash Recovery: commit (early return path) ---
        header->flags &= ~FLAG_WRITE_IN_PROGRESS;
        checkpoint_header();
        update_header_checksum();
        msync(map_base, PAGE_SIZE, MS_SYNC);

        return new_offset;
    }

    // 4. Phase 1: Greedy descent from top layer to (new_level + 1)
    //    At each of these layers, we just find the single nearest node (ef=1)
    //    to narrow down the entry point for the connection layers.
    uint64_t current_ep = ep;

    for (int lc = current_max_level; lc > new_level; --lc) {
        auto layer_result = search_layer(q, dim, current_ep, lc, 1);
        if (!layer_result.empty()) {
            current_ep = layer_result[0].second; // closest node at this layer
        }
    }

    // 5. Phase 2: For layers min(new_level, current_max_level) down to 0,
    //    do a full beam search with ef_construction, select neighbors,
    //    and add bidirectional connections.
    int connect_top = std::min(new_level, current_max_level);
    int M = header->M_max;
    int ef = header->ef_construction;

    for (int lc = connect_top; lc >= 0; --lc) {
        // Search for ef_construction nearest neighbors at this layer
        auto candidates = search_layer(q, dim, current_ep, lc, ef);
        
        // Select M best neighbors
        auto neighbors = select_neighbors(candidates, M);
        
        // Wire bidirectional connections
        for (auto& [dist, neighbor_offset] : neighbors) {
            // New -> Neighbor
            add_hnsw_link(new_offset, lc, neighbor_offset, dist);
            // Neighbor -> New (symmetric distance)
            add_hnsw_link(neighbor_offset, lc, new_offset, dist);
        }

        // Update entry point for next layer: closest from this search
        if (!candidates.empty()) {
            current_ep = candidates[0].second;
        }
    }

    // 6. Update global entry point if new node has a higher level
    if (new_level > current_max_level) {
        header->hnsw_max_level.store(new_level);
        header->hnsw_entry_point.store(new_offset);
    }

    // --- Crash Recovery: commit — mutation complete ---
    header->flags &= ~FLAG_WRITE_IN_PROGRESS;
    checkpoint_header();
    update_header_checksum();
    msync(map_base, PAGE_SIZE, MS_SYNC);

    return new_offset;
}