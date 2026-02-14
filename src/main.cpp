#include "engine.h"
#include "slb.h"
#include <iostream>
#include <vector>

int main() {
    try {
        KVEngine db("agent_memory.db", 10 * 1024 * 1024);
        ContextManager slb(&db);

        // 1. Setup Graph: A -> B -> C
        std::cout << "[Setup] Creating Graph Trajectory..." << std::endl;
        std::vector<float> vec(1536, 0.0f);
        
        uint64_t node_A = db.create_node(1, vec);
        uint64_t node_B = db.create_node(2, vec);
        uint64_t node_C = db.create_node(3, vec);

        // Link A -> B (Strong connection)
        // CRITICAL: We pass OFFSETS as target IDs for now to allow traversal
        db.add_edge(node_A, node_B, 0.9f); 
        
        // Link B -> C (Medium connection)
        db.add_edge(node_B, node_C, 0.8f);

        // 2. Simulate Agent accessing Node A
        std::cout << "[Action] Agent accessing Node A..." << std::endl;
        slb.observe_and_predict(node_A);

        // 3. Check Prediction
        // We expect Node B (Direct child) and Node C (2nd hop) to be in context
        std::vector<uint64_t> context = slb.get_context_window();
        
        std::cout << "[Result] Predicted Context (Offsets): ";
        for(uint64_t offset : context) {
            std::cout << offset << " ";
        }
        std::cout << std::endl;

        if (context.size() >= 2) {
            std::cout << "SUCCESS: Spreading activation found multi-hop context." << std::endl;
        } else {
            std::cout << "FAILURE: Context missing." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}