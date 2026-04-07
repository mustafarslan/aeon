#include "aeon/platform_io.hpp"

#include <cstdlib>

#if defined(AEON_PLATFORM_WINDOWS)
#include <malloc.h>
#include <windows.h>
#else
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace aeon::platform {

[[nodiscard]] void *aligned_alloc_pages(size_t size) noexcept {
#if defined(AEON_PLATFORM_WINDOWS)
  return _aligned_malloc(size, PAGE_SIZE);
#else
  void *ptr = nullptr;
  if (posix_memalign(&ptr, PAGE_SIZE, size) != 0) {
    return nullptr;
  }
  return ptr;
#endif
}

void aligned_free_pages(void *ptr) noexcept {
#if defined(AEON_PLATFORM_WINDOWS)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

[[nodiscard]] FileHandle file_open_direct(const char *path, int mode) noexcept {
#if defined(AEON_PLATFORM_WINDOWS)
  // FILE_FLAG_NO_BUFFERING ensures O_DIRECT semantics on Windows
  return CreateFileA(path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
                     nullptr, OPEN_ALWAYS,
                     FILE_FLAG_NO_BUFFERING | FILE_ATTRIBUTE_NORMAL, nullptr);
#elif defined(__APPLE__) || defined(__MACH__)
  // Darwin lacks O_DIRECT, fallback to F_NOCACHE
  int fd = ::open(path, O_RDWR | O_CREAT, mode);
  if (fd != -1) {
    ::fcntl(fd, F_NOCACHE, 1);
  }
  return fd;
#else
  // Linux O_DIRECT
  return ::open(path, O_RDWR | O_CREAT | O_DIRECT, mode);
#endif
}

size_t pread_aligned(FileHandle h, void *buf, size_t count,
                     size_t offset) noexcept {
#if defined(AEON_PLATFORM_WINDOWS)
  OVERLAPPED overlap = {};
  overlap.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
  overlap.OffsetHigh = static_cast<DWORD>(offset >> 32);
  DWORD bytesRead = 0;
  if (ReadFile(h, buf, static_cast<DWORD>(count), &bytesRead, &overlap)) {
    return static_cast<size_t>(bytesRead);
  }
  return 0;
#else
  ssize_t res = ::pread(h, buf, count, static_cast<off_t>(offset));
  return res > 0 ? static_cast<size_t>(res) : 0;
#endif
}

size_t pwrite_aligned(FileHandle h, const void *buf, size_t count,
                      size_t offset) noexcept {
#if defined(AEON_PLATFORM_WINDOWS)
  OVERLAPPED overlap = {};
  overlap.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
  overlap.OffsetHigh = static_cast<DWORD>(offset >> 32);
  DWORD bytesWritten = 0;
  if (WriteFile(h, buf, static_cast<DWORD>(count), &bytesWritten, &overlap)) {
    return static_cast<size_t>(bytesWritten);
  }
  return 0;
#else
  ssize_t res = ::pwrite(h, buf, count, static_cast<off_t>(offset));
  return res > 0 ? static_cast<size_t>(res) : 0;
#endif
}

bool fdatasync_file(FileHandle h) noexcept {
  if (h == INVALID_FILE_HANDLE)
    return false;
#if defined(AEON_PLATFORM_WINDOWS)
  return FlushFileBuffers(h) != 0;
#elif defined(__APPLE__) || defined(__MACH__)
  // Darwin requires F_FULLFSYNC for hardware drive flush
  return ::fcntl(h, F_FULLFSYNC) != -1;
#else
  // Linux supports fdatasync
  return ::fdatasync(h) == 0;
#endif
}

} // namespace aeon::platform
