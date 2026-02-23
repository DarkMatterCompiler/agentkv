#include <iostream>
#include <atomic>
#include <cstdint>

constexpr uint64_t PAGE_SIZE = 4096;

struct alignas(PAGE_SIZE) StorageHeader {
    uint32_t magic = 0x41474B56; 
    uint32_t version = 1;
    std::atomic<uint64_t> write_head;
    std::atomic<uint64_t> capacity;
    std::atomic<uint32_t> active_readers;

    // --- HNSW Global State ---
    std::atomic<uint64_t> hnsw_entry_point;
    std::atomic<int32_t>  hnsw_max_level;
    uint32_t M_max;
    uint32_t ef_construction;

    uint8_t _pad[PAGE_SIZE - 48]; 
};

int main() {
    std::cout << "StorageHeader size: " << sizeof(StorageHeader) << " bytes\n";
    std::cout << "Expected: " << PAGE_SIZE << " bytes\n";
    std::cout << "\nField sizes:\n";
    std::cout << "magic: " << sizeof(uint32_t) << "\n";
    std::cout << "version: " << sizeof(uint32_t) << "\n";
    std::cout << "write_head: " << sizeof(std::atomic<uint64_t>) << "\n";
    std::cout << "capacity: " << sizeof(std::atomic<uint64_t>) << "\n";
    std::cout << "active_readers: " << sizeof(std::atomic<uint32_t>) << "\n";
    std::cout << "hnsw_entry_point: " << sizeof(std::atomic<uint64_t>) << "\n";
    std::cout << "hnsw_max_level: " << sizeof(std::atomic<int32_t>) << "\n";
    std::cout << "M_max: " << sizeof(uint32_t) << "\n";
    std::cout << "ef_construction: " << sizeof(uint32_t) << "\n";
    
    int total = 4 + 4 + 8 + 8 + 4 + 8 + 4 + 4 + 4;
    std::cout << "\nTotal data: " << total << " bytes\n";
    std::cout << "Padding needed: " << (PAGE_SIZE - total) << " bytes\n";
}
