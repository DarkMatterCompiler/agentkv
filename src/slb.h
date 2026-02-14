#pragma once
#include "engine.h"
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>

struct ScoredNode {
    uint64_t node_offset;
    float score;

    // Max heap comparison (higher score = top)
    bool operator<(const ScoredNode& other) const {
        return score < other.score;
    }
};

class ContextManager {
private:
    KVEngine* engine;
    
    // The "Hot Context" - Nodes we think the agent needs next
    std::vector<uint64_t> hot_context;

    // Configuration
    float decay_factor = 0.8f;  // Energy loss per hop
    float activation_threshold = 0.1f;
    int max_depth = 2;

public:
    ContextManager(KVEngine* eng) : engine(eng) {}

    // The Core Algorithm: Spreading Activation
    // Traverses the graph from the start_node and finds relevant context.
    void observe_and_predict(uint64_t start_node_offset);

    // Get the current predicted context
    std::vector<uint64_t> get_context_window() {
        return hot_context;
    }
};