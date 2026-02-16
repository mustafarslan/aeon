/**
 * @file AeonClient.hpp
 * @brief Header-Only Unreal Engine Wrapper for the Aeon Memory OS C-API.
 *
 * UNREAL ENGINE CONSTRAINTS:
 *   1. Unreal disables C++ exceptions (-fno-exceptions / /EHs-).
 *      We MUST NOT throw std::exception or use try/catch.
 *   2. Unreal has its own allocator (FMalloc). We use caller-allocated
 *      buffers and never cross-allocate with C++ new/delete.
 *   3. Unreal uses FString/TArray instead of std::string/std::vector.
 *      This wrapper exposes both raw C arrays and Unreal-friendly helpers.
 *   4. All functions check aeon_error_t and return it directly.
 *
 * USAGE IN UNREAL:
 *   // In your UObject or AActor
 *   FAeonClient Memory;
 *   if (Memory.Open(TEXT("Content/Memory/npc_brain.bin")))
 *   {
 *       TArray<float> Query;  // 768-dim embedding from your model
 *       TArray<FAeonResult> Results;
 *       Memory.Navigate(Query, Results);
 *
 *       for (const auto& R : Results)
 *           UE_LOG(LogTemp, Log, TEXT("Node %llu sim=%.4f"), R.Id,
 * R.Similarity);
 *   }
 *
 * @copyright 2024-2026 Aeon Project. MIT License.
 */

#pragma once

#include "aeon/aeon_c_api.h"

#include <cstdint>
#include <cstring>

// Unreal-specific includes (conditional — allows standalone use too)
#if defined(UNREAL_ENGINE) || defined(WITH_EDITOR) || defined(UE_BUILD_SHIPPING)
#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "CoreMinimal.h"
#define AEON_UE_AVAILABLE 1
#else
#define AEON_UE_AVAILABLE 0
#endif

// ============================================================================
// Result struct — POD, no constructors that require exceptions
// ============================================================================

struct FAeonResult {
  uint64_t Id;
  float Similarity;
  float PreviewX;
  float PreviewY;
  float PreviewZ;
  int32_t RequiresCloudFetch;
};

static_assert(sizeof(FAeonResult) == sizeof(aeon_result_node_t),
              "FAeonResult must match aeon_result_node_t layout");

// ============================================================================
// FAeonClient — Flat, exception-free Unreal Engine wrapper
// ============================================================================

class FAeonClient {
public:
  FAeonClient() : Atlas(nullptr) {}

  ~FAeonClient() { Close(); }

  // Non-copyable, moveable
  FAeonClient(const FAeonClient &) = delete;
  FAeonClient &operator=(const FAeonClient &) = delete;

  FAeonClient(FAeonClient &&Other) noexcept : Atlas(Other.Atlas) {
    Other.Atlas = nullptr;
  }

  FAeonClient &operator=(FAeonClient &&Other) noexcept {
    if (this != &Other) {
      Close();
      Atlas = Other.Atlas;
      Other.Atlas = nullptr;
    }
    return *this;
  }

  // -----------------------------------------------------------------
  // Lifecycle
  // -----------------------------------------------------------------

  /**
   * Open or create an Atlas file at the given path.
   * @return true on success, false on failure.
   */
  bool Open(const char *Path) {
    if (Atlas)
      Close();
    LastError = aeon_atlas_create(Path, &Atlas);
    return LastError == AEON_OK;
  }

#if AEON_UE_AVAILABLE
  bool Open(const FString &Path) { return Open(TCHAR_TO_UTF8(*Path)); }
#endif

  void Close() {
    if (Atlas) {
      aeon_atlas_destroy(Atlas);
      Atlas = nullptr;
    }
  }

  bool IsOpen() const { return Atlas != nullptr; }

  // -----------------------------------------------------------------
  // Query
  // -----------------------------------------------------------------

  /**
   * Navigate the Atlas with a 768-dim query vector.
   *
   * @param QueryVector   Pointer to 768 floats.
   * @param OutResults    Caller-allocated buffer of at least MaxResults
   * elements.
   * @param MaxResults    Capacity of OutResults.
   * @param OutCount      Number of results actually written.
   * @param BeamWidth     Beam width (1 = greedy, max 16).
   * @param ApplyCSLS     Apply CSLS hubness correction.
   * @return AEON_OK on success.
   */
  aeon_error_t Navigate(const float *QueryVector, FAeonResult *OutResults,
                        size_t MaxResults, size_t &OutCount,
                        uint32_t BeamWidth = 1, bool ApplyCSLS = false) {
    if (!Atlas)
      return AEON_ERR_NULL_PTR;

    LastError = aeon_atlas_navigate(
        Atlas, QueryVector, AEON_EMBEDDING_DIM, BeamWidth, ApplyCSLS ? 1 : 0,
        reinterpret_cast<aeon_result_node_t *>(OutResults), MaxResults,
        &OutCount);
    return LastError;
  }

#if AEON_UE_AVAILABLE
  /**
   * Unreal-friendly Navigate that returns results in a TArray.
   */
  aeon_error_t Navigate(const TArray<float> &Query,
                        TArray<FAeonResult> &OutResults, uint32_t BeamWidth = 1,
                        bool ApplyCSLS = false) {
    if (Query.Num() != AEON_EMBEDDING_DIM)
      return AEON_ERR_INVALID_ARG;

    OutResults.SetNum(AEON_TOP_K_LIMIT);
    size_t Count = 0;
    auto Err = Navigate(Query.GetData(), OutResults.GetData(), AEON_TOP_K_LIMIT,
                        Count, BeamWidth, ApplyCSLS);
    OutResults.SetNum(static_cast<int32_t>(Count));
    return Err;
  }
#endif

  // -----------------------------------------------------------------
  // Mutation
  // -----------------------------------------------------------------

  aeon_error_t Insert(uint64_t ParentId, const float *Vector,
                      const char *Metadata, uint64_t &OutId) {
    if (!Atlas)
      return AEON_ERR_NULL_PTR;
    LastError = aeon_atlas_insert(Atlas, ParentId, Vector, AEON_EMBEDDING_DIM,
                                  Metadata, &OutId);
    return LastError;
  }

  // -----------------------------------------------------------------
  // Inspection
  // -----------------------------------------------------------------

  aeon_error_t GetSize(size_t &OutSize) {
    if (!Atlas)
      return AEON_ERR_NULL_PTR;
    LastError = aeon_atlas_size(Atlas, &OutSize);
    return LastError;
  }

  aeon_error_t GetTombstoneCount(size_t &OutCount) {
    if (!Atlas)
      return AEON_ERR_NULL_PTR;
    LastError = aeon_atlas_tombstone_count(Atlas, &OutCount);
    return LastError;
  }

  // -----------------------------------------------------------------
  // Dreaming — Memory Consolidation
  // -----------------------------------------------------------------

  /**
   * Consolidate old nodes into a single summary node.
   * Call during loading screens, idle frames, or sleep states.
   */
  aeon_error_t ConsolidateSubgraph(const uint64_t *OldNodeIds,
                                   size_t OldNodeCount,
                                   const float *SummaryVector,
                                   const char *SummaryMetadata,
                                   uint64_t &OutSummaryId) {
    if (!Atlas)
      return AEON_ERR_NULL_PTR;
    LastError = aeon_atlas_consolidate_subgraph(
        Atlas, OldNodeIds, OldNodeCount, SummaryVector, AEON_EMBEDDING_DIM,
        SummaryMetadata, &OutSummaryId);
    return LastError;
  }

  /**
   * Compact the Atlas file, physically reclaiming tombstoned storage.
   */
  aeon_error_t Compact(const char *NewPath = nullptr) {
    if (!Atlas)
      return AEON_ERR_NULL_PTR;
    LastError = aeon_atlas_compact(Atlas, NewPath);
    return LastError;
  }

  // -----------------------------------------------------------------
  // Diagnostics
  // -----------------------------------------------------------------

  aeon_error_t GetLastError() const { return LastError; }

  static const char *GetVersion() { return aeon_version(); }

private:
  aeon_atlas_t *Atlas;
  aeon_error_t LastError = AEON_OK;
};
