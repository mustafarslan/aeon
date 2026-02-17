/**
 * @file aeon_node.cpp
 * @brief Ultra-low-latency Node.js native bridge for the Aeon V4.1 Memory OS.
 *
 * DESIGN INVARIANTS:
 *   1. Zero-Copy V8 Blood-Brain Barrier: Float32Array → raw float* via .Data()
 *      with ZERO V8 heap allocations or ArrayBuffer copies.
 *   2. Synchronous Hot Paths: atlasInsert, atlasNavigate, traceAppend execute
 *      on the main V8 thread. The ~3µs C++ latency is cheaper than ~50µs
 *      libuv async context-switch overhead.
 *   3. Strict Lifecycle: close() nullifies pointers; destructor guards
 * double-free. Every method checks liveness before touching C handles.
 *   4. No C++ exceptions cross FFI: node-addon-api compiled with
 *      NAPI_DISABLE_CPP_EXCEPTIONS. All errors go through Napi::Error::New().
 *
 * TARGET: macOS Apple Silicon (M4 Max) — NEON SDOT via -mcpu=apple-m4.
 *
 * @copyright 2024–2026 Aeon Project. All rights reserved.
 */

#include "aeon/aeon_c_api.h"
#include <napi.h>

#include <cstring>
#include <string>

// ═══════════════════════════════════════════════════════════════════════════════
// Error Code → Human-Readable String
// ═══════════════════════════════════════════════════════════════════════════════

static const char *aeon_error_string(aeon_error_t err) {
  switch (err) {
  case AEON_OK:
    return "AEON_OK";
  case AEON_ERR_NULL_PTR:
    return "AEON_ERR_NULL_PTR";
  case AEON_ERR_INVALID_ARG:
    return "AEON_ERR_INVALID_ARG";
  case AEON_ERR_FILE_IO:
    return "AEON_ERR_FILE_IO";
  case AEON_ERR_INVALID_FORMAT:
    return "AEON_ERR_INVALID_FORMAT";
  case AEON_ERR_OUT_OF_MEMORY:
    return "AEON_ERR_OUT_OF_MEMORY";
  case AEON_ERR_NODE_NOT_FOUND:
    return "AEON_ERR_NODE_NOT_FOUND";
  case AEON_ERR_ALREADY_DEAD:
    return "AEON_ERR_ALREADY_DEAD";
  case AEON_ERR_BUFFER_TOO_SMALL:
    return "AEON_ERR_BUFFER_TOO_SMALL";
  case AEON_ERR_UNKNOWN:
    return "AEON_ERR_UNKNOWN";
  default:
    return "AEON_ERR_UNRECOGNIZED";
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Macro: Check aeon_error_t and throw Napi::Error on failure
// Uses do-while(0) idiom for safe use in if/else without braces.
// ═══════════════════════════════════════════════════════════════════════════════

#define AEON_CHECK(env, expr, context)                                         \
  do {                                                                         \
    aeon_error_t _rc = (expr);                                                 \
    if (_rc != AEON_OK) {                                                      \
      std::string _msg = std::string("Aeon ") + (context) +                    \
                         " failed: " + aeon_error_string(_rc) + " (" +         \
                         std::to_string(static_cast<int>(_rc)) + ")";          \
      Napi::Error::New((env), _msg).ThrowAsJavaScriptException();              \
      return (env).Undefined();                                                \
    }                                                                          \
  } while (0)

// Variant for constructor (returns void, not Napi::Value)
#define AEON_CHECK_CTOR(env, expr, context)                                    \
  do {                                                                         \
    aeon_error_t _rc = (expr);                                                 \
    if (_rc != AEON_OK) {                                                      \
      std::string _msg = std::string("Aeon ") + (context) +                    \
                         " failed: " + aeon_error_string(_rc) + " (" +         \
                         std::to_string(static_cast<int>(_rc)) + ")";          \
      Napi::Error::New((env), _msg).ThrowAsJavaScriptException();              \
      return;                                                                  \
    }                                                                          \
  } while (0)

// Variant for void-returning methods (close)
#define AEON_CHECK_VOID(env, expr, context)                                    \
  do {                                                                         \
    aeon_error_t _rc = (expr);                                                 \
    if (_rc != AEON_OK) {                                                      \
      std::string _msg = std::string("Aeon ") + (context) +                    \
                         " failed: " + aeon_error_string(_rc) + " (" +         \
                         std::to_string(static_cast<int>(_rc)) + ")";          \
      Napi::Error::New((env), _msg).ThrowAsJavaScriptException();              \
      return;                                                                  \
    }                                                                          \
  } while (0)

// ═══════════════════════════════════════════════════════════════════════════════
// AeonDB — Napi::ObjectWrap Class
// ═══════════════════════════════════════════════════════════════════════════════

class AeonDB : public Napi::ObjectWrap<AeonDB> {
public:
  /**
   * @brief Register the AeonDB class with the Node.js module exports.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(
        env, "AeonDB",
        {
            InstanceMethod<&AeonDB::AtlasInsert>("atlasInsert"),
            InstanceMethod<&AeonDB::AtlasNavigate>("atlasNavigate"),
            InstanceMethod<&AeonDB::TraceAppend>("traceAppend"),
            InstanceMethod<&AeonDB::TraceGetHistory>("traceGetHistory"),
            InstanceMethod<&AeonDB::AtlasSize>("atlasSize"),
            InstanceMethod<&AeonDB::TraceSize>("traceSize"),
            InstanceMethod<&AeonDB::Close>("close"),
            InstanceMethod<&AeonDB::IsClosed>("isClosed"),
        });

    Napi::FunctionReference *constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("AeonDB", func);
    return exports;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Constructor: AeonDB(atlasPath, tracePath, dim?, quantizationType?)
  // ─────────────────────────────────────────────────────────────────────────

  AeonDB(const Napi::CallbackInfo &info)
      : Napi::ObjectWrap<AeonDB>(info), atlas_(nullptr), trace_(nullptr),
        closed_(true) // Set to true until both handles are successfully created
  {
    Napi::Env env = info.Env();

    // ── Argument validation ──────────────────────────────────────────────
    if (info.Length() < 2) {
      Napi::TypeError::New(
          env, "AeonDB constructor requires at least 2 arguments: "
               "(atlasPath: string, tracePath: string, dim?: number, "
               "quantizationType?: number)")
          .ThrowAsJavaScriptException();
      return;
    }

    if (!info[0].IsString() || !info[1].IsString()) {
      Napi::TypeError::New(env, "atlasPath and tracePath must be strings")
          .ThrowAsJavaScriptException();
      return;
    }

    std::string atlas_path = info[0].As<Napi::String>().Utf8Value();
    std::string trace_path = info[1].As<Napi::String>().Utf8Value();

    // Optional: dim (default 768)
    uint32_t dim = AEON_EMBEDDING_DIM_DEFAULT;
    if (info.Length() > 2 && info[2].IsNumber()) {
      dim = info[2].As<Napi::Number>().Uint32Value();
    }

    // Optional: quantization_type (default 0 = FP32)
    uint32_t quant_type = 0;
    if (info.Length() > 3 && info[3].IsNumber()) {
      quant_type = info[3].As<Napi::Number>().Uint32Value();
    }

    // ── Create Atlas via V4.1 extended API ───────────────────────────────
    aeon_atlas_options_t opts;
    std::memset(&opts, 0, sizeof(opts));
    opts.dim = dim;
    opts.quantization_type = quant_type;
    opts.enable_wal = 1; // WAL enabled by default for durability

    AEON_CHECK_CTOR(env,
                    aeon_atlas_create_ex(atlas_path.c_str(), &opts, &atlas_),
                    "atlas_create_ex");

    // ── Create Trace ─────────────────────────────────────────────────────
    AEON_CHECK_CTOR(env, aeon_trace_create(trace_path.c_str(), &trace_),
                    "trace_create");

    // ── Store dimension for runtime validation ───────────────────────────
    // Query the actual dim from the Atlas (handles existing-file case)
    AEON_CHECK_CTOR(env, aeon_atlas_get_dim(atlas_, &dim_), "atlas_get_dim");

    closed_ = false;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Destructor: Safe cleanup (double-free guarded)
  // ─────────────────────────────────────────────────────────────────────────

  ~AeonDB() { CloseInternal(); }

private:
  // ═══════════════════════════════════════════════════════════════════════
  // Liveness Guard — every method MUST call this first
  // ═══════════════════════════════════════════════════════════════════════

  bool CheckLive(Napi::Env env) {
    if (closed_) {
      Napi::Error::New(
          env, "Aeon: Instance is closed. Cannot call methods after close().")
          .ThrowAsJavaScriptException();
      return false;
    }
    return true;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // atlasInsert(parentId: bigint, vector: Float32Array,
  //             metadata?: string, sessionId?: string) → bigint
  //
  // ZERO-COPY: Float32Array.Data() → raw float* passed directly to C API.
  // SYNCHRONOUS: ~2.23µs C++ execution ≪ ~50µs libuv async overhead.
  // ═══════════════════════════════════════════════════════════════════════

  Napi::Value AtlasInsert(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    if (!CheckLive(env))
      return env.Undefined();

    // ── Argument validation ──────────────────────────────────────────────
    if (info.Length() < 2) {
      Napi::TypeError::New(
          env, "atlasInsert requires at least 2 arguments: "
               "(parentId: bigint, vector: Float32Array, metadata?: string, "
               "sessionId?: string)")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    if (!info[0].IsBigInt()) {
      Napi::TypeError::New(env, "parentId must be a BigInt (e.g. 0n)")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    if (!info[1].IsTypedArray()) {
      Napi::TypeError::New(env, "vector must be a Float32Array")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    // ── Extract parent_id (BigInt → uint64_t) ────────────────────────────
    bool lossless = false;
    uint64_t parent_id = info[0].As<Napi::BigInt>().Uint64Value(&lossless);

    // ── ZERO-COPY: Extract raw float* from V8 Float32Array ───────────────
    Napi::Float32Array vec_arr = info[1].As<Napi::Float32Array>();
    const float *vec_data = vec_arr.Data();
    size_t vec_dim = vec_arr.ElementLength();

    // Dimension check against Atlas
    if (vec_dim != dim_) {
      Napi::RangeError::New(env, "Vector dimension mismatch: expected " +
                                     std::to_string(dim_) + ", got " +
                                     std::to_string(vec_dim))
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    // ── Optional: metadata ───────────────────────────────────────────────
    const char *metadata = "";
    std::string metadata_str;
    if (info.Length() > 2 && info[2].IsString()) {
      metadata_str = info[2].As<Napi::String>().Utf8Value();
      metadata = metadata_str.c_str();
    }

    // ── Optional: session_id ─────────────────────────────────────────────
    const char *session_id = nullptr;
    std::string session_str;
    if (info.Length() > 3 && info[3].IsString()) {
      session_str = info[3].As<Napi::String>().Utf8Value();
      session_id = session_str.c_str();
    }

    // ── C API call ───────────────────────────────────────────────────────
    uint64_t out_id = 0;
    AEON_CHECK(env,
               aeon_atlas_insert(atlas_, parent_id, vec_data, vec_dim, metadata,
                                 session_id, &out_id),
               "atlas_insert");

    return Napi::BigInt::New(env, out_id);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // atlasNavigate(query: Float32Array, topK?: number,
  //               beamWidth?: number, applyCSLS?: boolean,
  //               sessionId?: string) → NavigateResult[]
  //
  // ZERO-COPY: Float32Array.Data() → raw float*.
  // SYNCHRONOUS: ~3.56µs SLB lookup ≪ ~50µs async overhead.
  //
  // Result buffer is stack-allocated (50 entries × ~32 bytes = ~1.6KB).
  // Returned JS Array is the ONLY V8 allocation (unavoidable).
  // ═══════════════════════════════════════════════════════════════════════

  Napi::Value AtlasNavigate(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    if (!CheckLive(env))
      return env.Undefined();

    // ── Argument validation ──────────────────────────────────────────────
    if (info.Length() < 1) {
      Napi::TypeError::New(
          env, "atlasNavigate requires at least 1 argument: "
               "(query: Float32Array, topK?: number, beamWidth?: number, "
               "applyCSLS?: boolean, sessionId?: string)")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    if (!info[0].IsTypedArray()) {
      Napi::TypeError::New(env, "query must be a Float32Array")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    // ── ZERO-COPY: Extract raw float* ────────────────────────────────────
    Napi::Float32Array query_arr = info[0].As<Napi::Float32Array>();
    const float *query_data = query_arr.Data();
    size_t query_dim = query_arr.ElementLength();

    // Dimension check
    if (query_dim != dim_) {
      Napi::RangeError::New(env, "Query dimension mismatch: expected " +
                                     std::to_string(dim_) + ", got " +
                                     std::to_string(query_dim))
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    // ── Optional parameters ──────────────────────────────────────────────
    uint32_t top_k = 10;
    if (info.Length() > 1 && info[1].IsNumber()) {
      top_k = info[1].As<Napi::Number>().Uint32Value();
      if (top_k > AEON_TOP_K_LIMIT)
        top_k = AEON_TOP_K_LIMIT;
      if (top_k == 0)
        top_k = 1;
    }

    uint32_t beam_width = 1;
    if (info.Length() > 2 && info[2].IsNumber()) {
      beam_width = info[2].As<Napi::Number>().Uint32Value();
      if (beam_width > 16)
        beam_width = 16;
      if (beam_width == 0)
        beam_width = 1;
    }

    int apply_csls = 0;
    if (info.Length() > 3 && info[3].IsBoolean()) {
      apply_csls = info[3].As<Napi::Boolean>().Value() ? 1 : 0;
    }

    const char *session_id = nullptr;
    std::string session_str;
    if (info.Length() > 4 && info[4].IsString()) {
      session_str = info[4].As<Napi::String>().Utf8Value();
      session_id = session_str.c_str();
    }

    // ── Stack-allocated result buffer (ZERO heap allocation) ─────────────
    aeon_result_node_t results[AEON_TOP_K_LIMIT];
    size_t actual_count = 0;

    AEON_CHECK(env,
               aeon_atlas_navigate(atlas_, query_data, query_dim, beam_width,
                                   apply_csls, session_id, results,
                                   static_cast<size_t>(top_k), &actual_count),
               "atlas_navigate");

    // ── Marshal results to JS Array of { id: BigInt, score: Number } ─────
    // This V8 allocation is unavoidable — it's the return value.
    // Pre-sized array avoids internal realloc.
    Napi::Array js_results = Napi::Array::New(env, actual_count);

    for (size_t i = 0; i < actual_count; ++i) {
      Napi::Object obj = Napi::Object::New(env);
      obj.Set("id", Napi::BigInt::New(env, results[i].id));
      obj.Set("score", Napi::Number::New(
                           env, static_cast<double>(results[i].similarity)));
      js_results.Set(static_cast<uint32_t>(i), obj);
    }

    return js_results;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // traceAppend(role: number, text: string,
  //             sessionId?: string, atlasId?: bigint) → bigint
  //
  // SYNCHRONOUS: ~2.23µs WAL insert ≪ ~50µs async overhead.
  // ═══════════════════════════════════════════════════════════════════════

  Napi::Value TraceAppend(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    if (!CheckLive(env))
      return env.Undefined();

    // ── Argument validation ──────────────────────────────────────────────
    if (info.Length() < 2) {
      Napi::TypeError::New(env,
                           "traceAppend requires at least 2 arguments: "
                           "(role: number, text: string, sessionId?: string, "
                           "atlasId?: bigint)")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    if (!info[0].IsNumber()) {
      Napi::TypeError::New(
          env, "role must be a number (0=User, 1=System, 2=Concept, 3=Summary)")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    if (!info[1].IsString()) {
      Napi::TypeError::New(env, "text must be a string")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    uint16_t role =
        static_cast<uint16_t>(info[0].As<Napi::Number>().Uint32Value());
    std::string text = info[1].As<Napi::String>().Utf8Value();

    // ── Optional: session_id ─────────────────────────────────────────────
    const char *session_id = "";
    std::string session_str;
    if (info.Length() > 2 && info[2].IsString()) {
      session_str = info[2].As<Napi::String>().Utf8Value();
      session_id = session_str.c_str();
    }

    // ── Optional: atlas_id ───────────────────────────────────────────────
    uint64_t atlas_id = 0;
    if (info.Length() > 3 && info[3].IsBigInt()) {
      bool lossless = false;
      atlas_id = info[3].As<Napi::BigInt>().Uint64Value(&lossless);
    }

    // ── C API call ───────────────────────────────────────────────────────
    uint64_t out_id = 0;
    AEON_CHECK(env,
               aeon_trace_append_event(trace_, session_id, role, text.c_str(),
                                       atlas_id, &out_id),
               "trace_append_event");

    return Napi::BigInt::New(env, out_id);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // atlasSize() → number
  // ═══════════════════════════════════════════════════════════════════════

  Napi::Value AtlasSize(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    if (!CheckLive(env))
      return env.Undefined();

    size_t size = 0;
    AEON_CHECK(env, aeon_atlas_size(atlas_, &size), "atlas_size");

    return Napi::Number::New(env, static_cast<double>(size));
  }

  // ═══════════════════════════════════════════════════════════════════════
  // traceSize() → number
  // ═══════════════════════════════════════════════════════════════════════

  Napi::Value TraceSize(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    if (!CheckLive(env))
      return env.Undefined();

    size_t size = 0;
    AEON_CHECK(env, aeon_trace_size(trace_, &size), "trace_size");

    return Napi::Number::New(env, static_cast<double>(size));
  }

  // ═══════════════════════════════════════════════════════════════════════
  // traceGetHistory(sessionId: string, limit: number)
  //   → Array<{ id: bigint, role: number, text: string, timestamp: bigint }>
  //
  // Wraps aeon_trace_get_history + aeon_trace_get_event_text for each
  // returned event. Heap-allocates via std::vector — no stack bombs.
  // Node.js dictates the limit — no silent truncation.
  // SYNCHRONOUS: executes on V8 main thread.
  // ═══════════════════════════════════════════════════════════════════════

  Napi::Value TraceGetHistory(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    if (!CheckLive(env))
      return env.Undefined();

    // ── Argument validation ──────────────────────────────────────────────
    if (info.Length() < 2) {
      Napi::TypeError::New(env, "traceGetHistory requires 2 arguments: "
                                "(sessionId: string, limit: number)")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    if (!info[0].IsString()) {
      Napi::TypeError::New(env, "sessionId must be a string")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    if (!info[1].IsNumber()) {
      Napi::TypeError::New(env, "limit must be a number")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    std::string session_id = info[0].As<Napi::String>().Utf8Value();
    uint32_t limit = info[1].As<Napi::Number>().Uint32Value();
    if (limit == 0)
      limit = 1;

    // ── Heap-allocated event buffer ──────────────────────────────────────
    // No stack allocation. No silent cap. Node.js dictates the limit.
    // Each aeon_trace_event_t is 512 bytes; std::vector handles it safely.
    std::vector<aeon_trace_event_t> events(limit);
    size_t actual_count = 0;

    AEON_CHECK(env,
               aeon_trace_get_history(trace_, session_id.c_str(), events.data(),
                                      static_cast<size_t>(limit),
                                      &actual_count),
               "trace_get_history");

    // ── Marshal results to JS Array ──────────────────────────────────────
    Napi::Array js_results = Napi::Array::New(env, actual_count);

    // Reusable text buffer for blob reads. We track capacity manually
    // and use C++23 resize_and_overwrite to bypass zero-initialization
    // on growth — critical for hot-path FFI where blobs may be large.
    std::string text_buf;
    size_t text_buf_cap = 0;

    for (size_t i = 0; i < actual_count; ++i) {
      const aeon_trace_event_t &evt = events[i];

      Napi::Object obj = Napi::Object::New(env);
      obj.Set("id", Napi::BigInt::New(env, evt.id));
      obj.Set("role", Napi::Number::New(env, static_cast<double>(evt.role)));
      obj.Set("timestamp", Napi::BigInt::New(env, evt.timestamp));

      // Retrieve full text from blob arena
      if (evt.blob_size > 0) {
        const size_t needed = static_cast<size_t>(evt.blob_size) + 1;

        // Grow capacity only when needed — monotonically increasing.
        if (needed > text_buf_cap) {
          text_buf.reserve(needed);
          text_buf_cap = needed;
        }

        // C++23 resize_and_overwrite: sets .size() without zero-fill.
        // The lambda receives (buf, count) and returns actual chars written.
        size_t actual_len = 0;
        aeon_error_t rc = AEON_OK;
        text_buf.resize_and_overwrite(needed, [&](char *buf, size_t count) {
          rc = aeon_trace_get_event_text(trace_, evt.blob_offset, evt.blob_size,
                                         buf, count, &actual_len);
          return (rc == AEON_OK) ? actual_len : size_t{0};
        });

        if (rc == AEON_OK) {
          obj.Set("text",
                  Napi::String::New(env, text_buf.data(), text_buf.size()));
        } else {
          // Fallback to inline preview
          obj.Set("text", Napi::String::New(env, evt.text_preview));
        }
      } else {
        // No blob — use inline preview
        obj.Set("text", Napi::String::New(env, evt.text_preview));
      }

      js_results.Set(static_cast<uint32_t>(i), obj);
    }

    return js_results;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // close() — Explicit resource cleanup
  //
  // Calls aeon_atlas_destroy + aeon_trace_destroy, then nullifies pointers.
  // Safe to call multiple times (idempotent).
  // ═══════════════════════════════════════════════════════════════════════

  Napi::Value Close(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();

    if (!closed_) {
      // Destroy Atlas
      if (atlas_) {
        aeon_error_t rc = aeon_atlas_destroy(atlas_);
        if (rc != AEON_OK) {
          // Log but don't throw — we still want to destroy trace
          // Best-effort cleanup during close().
        }
        atlas_ = nullptr;
      }

      // Destroy Trace
      if (trace_) {
        aeon_error_t rc = aeon_trace_destroy(trace_);
        if (rc != AEON_OK) {
          // Same best-effort policy
        }
        trace_ = nullptr;
      }

      closed_ = true;
    }

    return env.Undefined();
  }

  // ═══════════════════════════════════════════════════════════════════════
  // isClosed() → boolean
  // ═══════════════════════════════════════════════════════════════════════

  Napi::Value IsClosed(const Napi::CallbackInfo &info) {
    return Napi::Boolean::New(info.Env(), closed_);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Internal cleanup (called by destructor — no Napi::Env available)
  // ═══════════════════════════════════════════════════════════════════════

  void CloseInternal() {
    if (!closed_) {
      if (atlas_) {
        aeon_atlas_destroy(atlas_);
        atlas_ = nullptr;
      }
      if (trace_) {
        aeon_trace_destroy(trace_);
        trace_ = nullptr;
      }
      closed_ = true;
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Instance State
  // ═══════════════════════════════════════════════════════════════════════

  aeon_atlas_t *atlas_; ///< Opaque handle to Atlas (mmap-backed vector index)
  aeon_trace_t *trace_; ///< Opaque handle to Trace (episodic WAL store)
  uint32_t dim_;        ///< Embedding dimensionality (queried from Atlas)
  bool closed_;         ///< Lifecycle flag: true after close() or destruction
};

// ═══════════════════════════════════════════════════════════════════════════════
// Module Initialization — N-API Entry Point
// ═══════════════════════════════════════════════════════════════════════════════

static Napi::Object Init(Napi::Env env, Napi::Object exports) {
  AeonDB::Init(env, exports);
  exports.Set("version", Napi::String::New(env, aeon_version()));
  return exports;
}

NODE_API_MODULE(aeon_node, Init)
