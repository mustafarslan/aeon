#pragma once

/**
 * @file hash.hpp
 * @brief Header-only FNV-1a 64-bit hash for WAL record checksums.
 *
 * FNV-1a is a non-cryptographic hash chosen for:
 *   - Zero external dependencies
 *   - Excellent distribution for binary payloads
 *   - Branchless inner loop (no if/else per byte)
 *   - constexpr-capable for compile-time evaluation
 */

#include <cstddef>
#include <cstdint>

namespace aeon::hash {

/// FNV-1a offset basis (64-bit).
inline constexpr uint64_t FNV1A_OFFSET_BASIS = 14695981039346656037ULL;

/// FNV-1a prime (64-bit).
inline constexpr uint64_t FNV1A_PRIME = 1099511628211ULL;

/**
 * @brief Computes FNV-1a 64-bit hash over a byte range.
 *
 * @param data  Pointer to first byte.
 * @param size  Number of bytes to hash.
 * @return      64-bit FNV-1a digest.
 */
constexpr uint64_t fnv1a_64(const void *data, size_t size) noexcept {
  const auto *bytes = static_cast<const uint8_t *>(data);
  uint64_t hash = FNV1A_OFFSET_BASIS;
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= FNV1A_PRIME;
  }
  return hash;
}

} // namespace aeon::hash
