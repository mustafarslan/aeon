#pragma once

/**
 * @file platform.hpp
 * @brief Cross-platform OS abstraction for memory-mapped I/O.
 *
 * Provides a unified API over:
 *   - POSIX:   mmap(), munmap(), msync(), madvise() (Linux/macOS/iOS/Android)
 *   - Win32:   CreateFileMapping(), MapViewOfFile(), etc.   (Windows/Xbox)
 *
 * All platform-specific headers and syscalls are confined to this single
 * translation boundary. The rest of Aeon only uses the aeon::platform:: API.
 *
 * Mobile/Edge memory hints:
 *   - platform::advise_random():   MADV_RANDOM (reduce readahead waste)
 *   - platform::advise_dontneed(): MADV_DONTNEED (release pages under LMK
 * pressure)
 *   - platform::advise_hugepage(): MADV_HUGEPAGE (Linux server only)
 */

#include <cstddef>
#include <cstdint>

// ============================================================================
// Platform-specific includes
// ============================================================================
#if defined(_WIN32) || defined(_WIN64)
#define AEON_PLATFORM_WINDOWS 1
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#define AEON_PLATFORM_POSIX 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace aeon::platform {

// ============================================================================
// File Handle Abstraction
// ============================================================================

#if defined(AEON_PLATFORM_WINDOWS)
using FileHandle = HANDLE;
inline constexpr FileHandle INVALID_FILE_HANDLE = INVALID_HANDLE_VALUE;
#else
using FileHandle = int;
inline constexpr FileHandle INVALID_FILE_HANDLE = -1;
#endif

// ============================================================================
// File Operations
// ============================================================================

/**
 * @brief Open or create a file for memory-mapped access.
 * @param path  Null-terminated file path (UTF-8 on POSIX, narrow char on Win32)
 * @param mode  0644-style permissions (POSIX only, ignored on Windows)
 * @return File handle or INVALID_FILE_HANDLE on failure.
 */
inline FileHandle file_open(const char *path,
                            [[maybe_unused]] int mode = 0644) {
#if defined(AEON_PLATFORM_WINDOWS)
  HANDLE h = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
                         nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  return h;
#else
  int flags = O_RDWR | O_CREAT;
  return ::open(path, flags, mode);
#endif
}

/**
 * @brief Close a file handle.
 */
inline void file_close(FileHandle h) {
  if (h == INVALID_FILE_HANDLE)
    return;
#if defined(AEON_PLATFORM_WINDOWS)
  CloseHandle(h);
#else
  ::close(h);
#endif
}

/**
 * @brief Get the size of an open file.
 * @return File size in bytes, or 0 on failure.
 */
inline size_t file_size(FileHandle h) {
#if defined(AEON_PLATFORM_WINDOWS)
  LARGE_INTEGER sz;
  if (!GetFileSizeEx(h, &sz))
    return 0;
  return static_cast<size_t>(sz.QuadPart);
#else
  struct stat sb;
  if (fstat(h, &sb) == -1)
    return 0;
  return static_cast<size_t>(sb.st_size);
#endif
}

/**
 * @brief Resize (truncate/extend) a file.
 * @return true on success.
 */
inline bool file_resize(FileHandle h, size_t new_size) {
#if defined(AEON_PLATFORM_WINDOWS)
  LARGE_INTEGER li;
  li.QuadPart = static_cast<LONGLONG>(new_size);
  if (!SetFilePointerEx(h, li, nullptr, FILE_BEGIN))
    return false;
  return SetEndOfFile(h) != 0;
#else
  return ftruncate(h, static_cast<off_t>(new_size)) == 0;
#endif
}

/**
 * @brief Check if a file existed before the last open call.
 *        On Windows, checks via GetLastError() == ERROR_ALREADY_EXISTS after
 *        OPEN_ALWAYS. On POSIX, caller should check filesystem::exists before
 *        open.
 */
inline bool file_existed_before_open([[maybe_unused]] FileHandle h) {
#if defined(AEON_PLATFORM_WINDOWS)
  return GetLastError() == ERROR_ALREADY_EXISTS;
#else
  return false; // Caller checks existence separately on POSIX
#endif
}

// ============================================================================
// Memory Mapping
// ============================================================================

/// Sentinel for failed mmap operations.
#if defined(AEON_PLATFORM_WINDOWS)
inline void *const MAP_FAILED_PTR = nullptr;
#else
// MAP_FAILED is (void*)-1 which is not constexpr — use inline const instead.
inline void *const MAP_FAILED_PTR = MAP_FAILED;
#endif

/**
 * @brief Create a read-write shared memory mapping of a file.
 * @param h     File handle
 * @param size  Size in bytes to map
 * @return Pointer to mapped region, or MAP_FAILED_PTR on failure.
 *
 * On Windows, this creates a file mapping object and maps a view.
 * The mapping handle is stored internally (WIN32 requires it for cleanup).
 * On POSIX, this calls mmap(MAP_SHARED).
 */
inline void *mem_map(FileHandle h, size_t size) {
#if defined(AEON_PLATFORM_WINDOWS)
  HANDLE mapping = CreateFileMappingA(
      h, nullptr, PAGE_READWRITE, static_cast<DWORD>(size >> 32),
      static_cast<DWORD>(size & 0xFFFFFFFF), nullptr);
  if (!mapping)
    return MAP_FAILED_PTR;
  void *ptr = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, size);
  // We can close the mapping handle immediately; the view keeps it alive.
  CloseHandle(mapping);
  return ptr ? ptr : MAP_FAILED_PTR;
#else
  return mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, h, 0);
#endif
}

/**
 * @brief Unmap a previously mapped memory region.
 */
inline void mem_unmap(void *addr, [[maybe_unused]] size_t size) {
  if (!addr || addr == MAP_FAILED_PTR)
    return;
#if defined(AEON_PLATFORM_WINDOWS)
  UnmapViewOfFile(addr);
#else
  munmap(addr, size);
#endif
}

/**
 * @brief Flush dirty pages to disk.
 */
inline void mem_sync(void *addr, size_t size) {
  if (!addr || addr == MAP_FAILED_PTR)
    return;
#if defined(AEON_PLATFORM_WINDOWS)
  FlushViewOfFile(addr, size);
#else
  msync(addr, size, MS_SYNC);
#endif
}

// ============================================================================
// Memory Advice (POSIX madvise / Windows prefetch hints)
// ============================================================================

/**
 * @brief Advise the kernel that access will be random (reduces readahead).
 * Critical for tree-structured indices where access patterns are non-serial.
 */
inline void advise_random(void *addr, size_t size) {
#if defined(AEON_PLATFORM_POSIX)
  madvise(addr, size, MADV_RANDOM);
#elif defined(AEON_PLATFORM_WINDOWS)
  // Windows: WIN32_MEMORY_RANGE_ENTRY + PrefetchVirtualMemory could be used
  // for targeted prefetch, but there is no direct MADV_RANDOM equivalent.
  // No-op on Windows.
  (void)addr;
  (void)size;
#endif
}

/**
 * @brief Release pages back to the OS without unmapping the region.
 * Essential for mobile/edge devices under Low Memory Killer (LMK) pressure.
 * After this call, pages are zeroed on next access (no data persistence).
 */
inline void advise_dontneed(void *addr, size_t size) {
#if defined(AEON_PLATFORM_POSIX)
  madvise(addr, size, MADV_DONTNEED);
#elif defined(AEON_PLATFORM_WINDOWS)
  // Windows equivalent: DiscardVirtualMemory (Win8.1+) or VirtualUnlock
  // For broad compatibility, use OFFER_PRIORITY_NORMAL via OfferVirtualMemory
  // Fallback: no-op
  (void)addr;
  (void)size;
#endif
}

/**
 * @brief Request transparent huge pages (2MB) for TLB efficiency.
 * Linux-only; no-op on macOS, Windows, and mobile.
 */
inline void advise_hugepage([[maybe_unused]] void *addr,
                            [[maybe_unused]] size_t size) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  madvise(addr, size, MADV_HUGEPAGE);
#endif
}

} // namespace aeon::platform
