#include <iostream>
#include <atomic>
#include <cstdint>

constexpr uint64_t CACHE_LINE = 64;

struct alignas(CACHE_LINE) Node {
    uint64_t id;             
    uint64_t timestamp;      
    std::atomic<uint64_t> edge_list_head; 
    uint64_t vector_head;    
    std::atomic<uint64_t> version_lock;
    uint8_t _pad[26];
};

int main() {
    std::cout << "Node size: " << sizeof(Node) << " bytes\n";
    std::cout << "Data used: 8+8+8+8+8+26 = " << (8+8+8+8+8+26) << " bytes\n";
    std::cout << "alignof(Node): " << alignof(Node) << " bytes\n";
    std::cout << "\nNeed to adjust padding to: " << (64 - (8+8+8+8+8)) << " bytes\n";
}
