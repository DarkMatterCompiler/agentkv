#include "slb.h"
#include <iostream>

void ContextManager::observe_and_predict(uint64_t start_node_offset) {
    // 1. Reset Context for this turn
    // (In a real system, we would decay old context, not clear it)
    hot_context.clear();
    
    // BFS Queue: {Node Offset, Current Activation Energy, Depth}
    // We use offsets directly to avoid pointer chasing overhead
    struct QueuedNode {
        uint64_t offset;
        float energy;
        int depth;
    };

    std::queue<QueuedNode> q;
    std::unordered_map<uint64_t, float> visited; // Map Offset -> Max Energy Seen

    // Start with the observed node (Energy = 1.0)
    q.push({start_node_offset, 1.0f, 0});
    visited[start_node_offset] = 1.0f;

    std::vector<ScoredNode> candidates;

    while(!q.empty()) {
        QueuedNode current = q.front();
        q.pop();

        if (current.depth >= max_depth) continue;
        if (current.energy < activation_threshold) continue;

        // Access the Node in Zero-Copy mode
        Node* node_ptr = engine->get_ptr<Node>(current.offset);
        
        // Read Semantic Edges
        uint64_t edge_head = node_ptr->edge_list_head.load();
        EdgeList* edges = engine->get_ptr<EdgeList>(edge_head);

        if (edges) {
            for(uint32_t i=0; i < edges->count; i++) {
                Edge& e = edges->edges[i];
                
                // Calculate new energy
                float new_energy = current.energy * e.weight * decay_factor;
                
                // Note: We need to resolve Target ID -> Offset. 
                // For MVP, we assumed Target ID was Offset or we had a hash map.
                // CRITICAL FIX: In Phase 1, Edge stored `target_node_id`. 
                // We cannot jump to an ID without an Index. 
                // FOR NOW: We will assume `target_node_id` IS `target_node_offset` 
                // (Optimization often used in graph DBs: Store physical ID in edge).
                
                uint64_t target_offset = e.target_node_id; 

                // Check if visited with higher energy
                if (visited.find(target_offset) == visited.end() || visited[target_offset] < new_energy) {
                    visited[target_offset] = new_energy;
                    q.push({target_offset, new_energy, current.depth + 1});
                    candidates.push_back({target_offset, new_energy});
                }
            }
        }
    }

    // Sort candidates by relevance
    std::sort(candidates.begin(), candidates.end()); 
    // (Default sort is ascending, we want descending. Reverse it.)
    std::reverse(candidates.begin(), candidates.end());

    // Keep Top-5
    for(int i=0; i < std::min((size_t)5, candidates.size()); i++) {
        hot_context.push_back(candidates[i].node_offset);
    }
}