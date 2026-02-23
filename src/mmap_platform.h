#pragma once
// =============================================================================
// mmap_platform.h — Cross-platform memory-mapped file I/O
//
// Provides zero-overhead wrappers around OS-specific mmap APIs.
// All functions are inline; the preprocessor eliminates the unused branch
// at compile time, so the generated code is identical to hand-written
// platform-specific code.
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace platform {

// ─── Handle ──────────────────────────────────────────────────────────────────
struct MMapHandle {
    void*  ptr  = nullptr;
    size_t size = 0;
#ifdef _WIN32
    HANDLE hFile    = INVALID_HANDLE_VALUE;
    HANDLE hMapping = nullptr;
#else
    int fd = -1;
#endif
};

// ─── Open + Map ──────────────────────────────────────────────────────────────
// Opens (or creates) a file and maps it into memory.
// If the file is new or smaller than `size`, it is extended to `size` bytes.
inline MMapHandle mmap_open(const char* path, size_t size) {
    MMapHandle h;
    h.size = size;

#ifdef _WIN32
    // --- Windows: CreateFile → SetFileSize → CreateFileMapping → MapViewOfFile ---
    h.hFile = CreateFileA(
        path,
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ,          // allow concurrent readers
        nullptr,
        OPEN_ALWAYS,              // open existing or create new
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (h.hFile == INVALID_HANDLE_VALUE)
        throw std::runtime_error("CreateFile failed: " + std::string(path));

    // Extend file to requested size (idempotent if already big enough)
    LARGE_INTEGER li;
    li.QuadPart = static_cast<LONGLONG>(size);
    if (!SetFilePointerEx(h.hFile, li, nullptr, FILE_BEGIN) ||
        !SetEndOfFile(h.hFile)) {
        CloseHandle(h.hFile);
        throw std::runtime_error("SetEndOfFile failed");
    }

    h.hMapping = CreateFileMappingA(
        h.hFile, nullptr, PAGE_READWRITE,
        static_cast<DWORD>(size >> 32),
        static_cast<DWORD>(size & 0xFFFFFFFF),
        nullptr
    );
    if (!h.hMapping) {
        CloseHandle(h.hFile);
        throw std::runtime_error("CreateFileMapping failed");
    }

    h.ptr = MapViewOfFile(h.hMapping, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!h.ptr) {
        CloseHandle(h.hMapping);
        CloseHandle(h.hFile);
        throw std::runtime_error("MapViewOfFile failed");
    }

#else
    // --- POSIX: open → ftruncate → mmap ---
    h.fd = open(path, O_RDWR | O_CREAT, 0666);
    if (h.fd < 0)
        throw std::runtime_error("open() failed: " + std::string(path));

    if (ftruncate(h.fd, static_cast<off_t>(size)) != 0) {
        close(h.fd);
        throw std::runtime_error("ftruncate failed");
    }

    h.ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, h.fd, 0);
    if (h.ptr == MAP_FAILED) {
        close(h.fd);
        throw std::runtime_error("mmap failed");
    }
#endif

    return h;
}

// ─── Sync (flush dirty pages to disk) ────────────────────────────────────────
inline void mmap_sync(const MMapHandle& h, size_t len) {
#ifdef _WIN32
    FlushViewOfFile(h.ptr, len);
    FlushFileBuffers(h.hFile);
#else
    msync(h.ptr, len, MS_SYNC);
#endif
}

// ─── Sync entire mapping ────────────────────────────────────────────────────
inline void mmap_sync_all(const MMapHandle& h) {
    mmap_sync(h, h.size);
}

// ─── Close (unmap + close file) ──────────────────────────────────────────────
inline void mmap_close(MMapHandle& h) {
#ifdef _WIN32
    if (h.ptr)      { UnmapViewOfFile(h.ptr);   h.ptr = nullptr; }
    if (h.hMapping) { CloseHandle(h.hMapping);   h.hMapping = nullptr; }
    if (h.hFile != INVALID_HANDLE_VALUE)
                    { CloseHandle(h.hFile);      h.hFile = INVALID_HANDLE_VALUE; }
#else
    if (h.ptr && h.ptr != MAP_FAILED)
                    { munmap(h.ptr, h.size);     h.ptr = nullptr; }
    if (h.fd >= 0)  { close(h.fd);               h.fd = -1; }
#endif
}

} // namespace platform
