#pragma once

/**
 * @file platform_io.hpp
 * @brief Zero-copy O_DIRECT / F_NOCACHE platform I/O abstraction.
 *
 * Ensures all allocations and I/O operations are strictly aligned to the
 * physical sector size (4096 bytes) for bypass of the OS unified buffer cache.
 */

#include <cstddef>
#include <cstdint>

#include "aeon/platform.hpp"

#if defined(_WIN32) || defined(_WIN64)
#define AEON_PLATFORM_WINDOWS 1
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#define AEON_PLATFORM_POSIX 1
#endif

namespace aeon::platform {

/// Strict hardware alignment requirement for O_DIRECT / F_NOCACHE.
inline constexpr size_t PAGE_SIZE = 4096;

/**
 * @brief Allocate memory strictly aligned to PAGE_SIZE.
 * Uses std::aligned_alloc (POSIX) or _aligned_malloc (Windows).
 */
[[nodiscard]] void *aligned_alloc_pages(size_t size) noexcept;

/**
 * @brief Free memory allocated by aligned_alloc_pages.
 */
void aligned_free_pages(void *ptr) noexcept;

/**
 * @brief Open file bypassing the OS page cache (O_DIRECT or F_NOCACHE).
 */
[[nodiscard]] FileHandle file_open_direct(const char *path,
                                          int mode = 0644) noexcept;

/**
 * @brief Zero-copy aligned positional read.
 * @param buf Must be aligned to PAGE_SIZE.
 * @param count Must be a multiple of PAGE_SIZE.
 * @param offset Must be a multiple of PAGE_SIZE.
 */
size_t pread_aligned(FileHandle h, void *buf, size_t count,
                     size_t offset) noexcept;

/**
 * @brief Zero-copy aligned positional write.
 * @param buf Must be aligned to PAGE_SIZE.
 * @param count Must be a multiple of PAGE_SIZE.
 * @param offset Must be a multiple of PAGE_SIZE.
 */
size_t pwrite_aligned(FileHandle h, const void *buf, size_t count,
                      size_t offset) noexcept;

/**
 * @brief Flush hardware write buffers.
 * fdatasync() on Linux, fcntl(F_FULLFSYNC) on Darwin.
 */
bool fdatasync_file(FileHandle h) noexcept;

} // namespace aeon::platform
