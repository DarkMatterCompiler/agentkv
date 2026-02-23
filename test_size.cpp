#include <iostream>
#include <atomic>
#include <cstdint>

constexpr uint64_t PAGE_SIZE = 4096;
constexpr uint64_t CACHE_LINE = 64;

struct alignas(PAGE_SIZE) StorageHeader {
    uint32_t magic = 0x41474B56;
    uint32_t version = 1;
    std::atomic<uint64_t> write_head;
    std::atomic<uint64_t> capacity;
    std::atomic<uint32_t> active_readers;
    uint8_t _pad[PAGE_SIZE - 24];
};

struct alignas(CACHE_LINE) Node {
    uint64_t id;
    uint64_t timestamp;
    uint64_t edge_list_head;
    uint32_t edge_count;
    uint64_t vector_head;
    uint16_t vector_dim;
    std::atomic<uint64_t> version_lock;
    uint8_t _pad[18];
};

int main() {
    std::cout << "StorageHeader: " << sizeof(StorageHeader) << " bytes\n";
    std::cout << "Node: " << sizeof(Node) << " bytes\n";
    std::cout << "std::atomic<uint64_t>: " << sizeof(std::atomic<uint64_t>) << " bytes\n";
    std::cout << "std::atomic<uint32_t>: " << sizeof(std::atomic<uint32_t>) << " bytes\n";
    std::cout << "alignof(std::atomic<uint64_t>): " << alignof(std::atomic<uint64_t>) << " bytes\n";
}
