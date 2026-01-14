#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aeon {

// Magic bytes: "ATLAS_01" in hex
constexpr uint64_t ATLAS_MAGIC = 0x41544C41535F3031;
constexpr uint64_t ATLAS_VERSION = 1;

/**
 * @brief Global file header for the memory-mapped region.
 * Ensures we are reading a valid Atlas file.
 */
struct alignas(64) AtlasHeader {
  uint64_t magic;       // 0x00: Magic number identifier
  uint64_t version;     // 0x08: Format version
  uint64_t node_count;  // 0x10: Current number of actively used nodes
  uint64_t capacity;    // 0x18: Total capacity (allocated slots) in the file
  uint8_t reserved[32]; // 0x20: Padding to ensure 64-byte size
};

static_assert(sizeof(AtlasHeader) == 64,
              "AtlasHeader must be exactly 64 bytes");
static_assert(std::is_standard_layout_v<AtlasHeader>);
static_assert(std::is_trivially_copyable_v<AtlasHeader>);

/**
 * @brief A single node in the Spatial Memory Palace.
 *
 * Layout is optimized for AVX-512 (64-byte alignment).
 * We explicitly pad the structure so that the `centroid` array
 * starts exactly at the beginning of a cache line (offset 64).
 */
struct alignas(64) Node {
  // --- Header Block (64 bytes) ---
  uint64_t id;                 // 0x00: Unique ID
  uint64_t parent_offset;      // 0x08: Byte offset to parent (0 if root)
  uint64_t first_child_offset; // 0x10: Byte offset to first child
  uint16_t child_count;        // 0x18: Number of children
  uint16_t flags;              // 0x1A: Flags (Leaf, etc.)
  uint8_t reserved[36];        // 0x1C: Padding to reach offset 64

  // --- Data Block (Aligned) ---
  // 768 floats * 4 bytes = 3072 bytes
  float centroid[768]; // 0x40: Semantic vector (Starts at 64)

  // --- Metadata Block ---
  char metadata[256]; // 0xC40: Fixed description
};

// Compile-time layout verification
static_assert(sizeof(Node) == 3392, "Node size must be exactly 3392 bytes");
static_assert(offsetof(Node, centroid) == 64,
              "Node centroid must start at offset 64 for alignment");
static_assert(std::is_standard_layout_v<Node>, "Node must be standard layout");
static_assert(std::is_trivially_copyable_v<Node>,
              "Node must be trivially copyable for mmap");

} // namespace aeon
