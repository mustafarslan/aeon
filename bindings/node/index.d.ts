/**
 * @aeon/node-mac — TypeScript definitions for the Aeon V4.1 Node.js bridge.
 *
 * DESIGN CONTRACTS:
 *   - All node/event IDs are `bigint` (uint64_t — exceeds Number.MAX_SAFE_INTEGER).
 *   - Vectors are `Float32Array` (enforced zero-copy V8→C++ transfer).
 *   - All hot-path methods are synchronous (no Promises).
 *
 * @module @aeon/node-mac
 * @copyright 2024–2026 Aeon Project. All rights reserved.
 */

/**
 * A single navigation result from the Atlas beam search.
 */
export interface NavigateResult {
    /** Unique node identifier (uint64_t — always BigInt). */
    id: bigint;
    /** Cosine similarity score (or CSLS-adjusted). Range: [-1.0, 1.0]. */
    score: number;
}

/**
 * A single trace event returned by traceGetHistory.
 */
export interface TraceHistoryEvent {
    /** Unique event identifier (uint64_t — BigInt). */
    id: bigint;
    /** Role as integer (matches AEON_ROLE_* C-API enum). */
    role: number;
    /** Full text content retrieved from blob arena (or inline preview). */
    text: string;
    /** Nanosecond UNIX timestamp (uint64_t — BigInt). */
    timestamp: bigint;
}

/**
 * AeonDB — Production-grade interface to the Aeon Memory OS.
 *
 * Manages an Atlas (semantic vector index) and a Trace (episodic WAL store)
 * through the Aeon V4.1 C-API. All hot-path methods execute synchronously
 * on the V8 main thread for sub-10µs latency.
 *
 * @example
 * ```typescript
 * import { AeonDB } from '@aeon/node-mac';
 *
 * const db = new AeonDB('/data/atlas.bin', '/data/trace.wal', 768, 1);
 *
 * // Insert with zero-copy Float32Array
 * const vec = new Float32Array(768).fill(0.1);
 * const nodeId = db.atlasInsert(0n, vec, 'concept:memory');
 *
 * // Navigate — returns nearest neighbors
 * const results = db.atlasNavigate(vec, 10);
 * for (const r of results) {
 *     console.log(`Node ${r.id}: similarity=${r.score.toFixed(4)}`);
 * }
 *
 * // Append episodic event
 * const eventId = db.traceAppend(0, 'User asked about quantum computing');
 *
 * db.close();
 * ```
 */
export class AeonDB {
    /**
     * Create an AeonDB instance backed by an Atlas file and a Trace WAL file.
     *
     * @param atlasPath  - Path to the Atlas memory-mapped file (created if missing).
     * @param tracePath  - Path to the Trace WAL file (created if missing).
     * @param dim        - Embedding dimensionality (default: 768). Ignored for existing files.
     * @param quantizationType - 0 = FP32 (default), 1 = INT8_SYMMETRIC.
     * @throws {Error} If file creation or handle allocation fails.
     */
    constructor(
        atlasPath: string,
        tracePath: string,
        dim?: number,
        quantizationType?: number
    );

    /**
     * Insert a new node into the Atlas tree.
     *
     * ZERO-COPY: The Float32Array's backing ArrayBuffer is passed directly
     * to the C++ kernel without copying. Do NOT modify the array during this call.
     *
     * @param parentId  - Parent node ID (0n for root if Atlas is empty).
     * @param vector    - Embedding vector (must match Atlas dimension).
     * @param metadata  - Optional metadata string (max 255 chars). Default: "".
     * @param sessionId - Optional session UUID for SLB cache routing. Default: null.
     * @returns The ID of the newly inserted node.
     * @throws {RangeError} If vector dimension doesn't match Atlas dimension.
     * @throws {Error}      If the C-API insert call fails.
     */
    atlasInsert(
        parentId: bigint,
        vector: Float32Array,
        metadata?: string,
        sessionId?: string
    ): bigint;

    /**
     * Navigate the Atlas tree — find nearest nodes to the query vector.
     *
     * ZERO-COPY: The Float32Array's backing ArrayBuffer is passed directly
     * to the C++ kernel without copying.
     *
     * SYNCHRONOUS: Executes on the V8 main thread (~3.56µs typical).
     *
     * @param query     - Query embedding vector (must match Atlas dimension).
     * @param topK      - Maximum results to return (default: 10, max: 50).
     * @param beamWidth - Beam search width (default: 1 = greedy, max: 16).
     * @param applyCSLS - Apply CSLS hubness correction (default: false).
     * @param sessionId - Session UUID for L1 SLB routing (default: null = global L2).
     * @returns Array of { id: bigint, score: number } sorted by descending similarity.
     * @throws {RangeError} If query dimension doesn't match Atlas dimension.
     * @throws {Error}      If the C-API navigate call fails.
     */
    atlasNavigate(
        query: Float32Array,
        topK?: number,
        beamWidth?: number,
        applyCSLS?: boolean,
        sessionId?: string
    ): NavigateResult[];

    /**
     * Append an episodic event to the Trace WAL.
     *
     * SYNCHRONOUS: Executes on the V8 main thread (~2.23µs typical).
     *
     * @param role      - Event role: 0=User, 1=System, 2=Concept, 3=Summary.
     * @param text      - Full text content (unlimited length; V4.1 blob arena).
     * @param sessionId - Multi-tenant session UUID (default: "").
     * @param atlasId   - Linked Atlas concept node ID (default: 0n = none).
     * @returns The unique ID of the newly appended event.
     * @throws {Error} If the C-API append call fails.
     */
    traceAppend(
        role: number,
        text: string,
        sessionId?: string,
        atlasId?: bigint
    ): bigint;

    /**
     * Returns the total number of nodes in the Atlas (including tombstoned).
     */
    atlasSize(): number;

    /**
     * Returns the total number of events in the Trace (mmap + delta buffer).
     */
    traceSize(): number;

    /**
     * Retrieve session history (newest first) with full text from blob arena.
     *
     * SYNCHRONOUS: Executes on the V8 main thread.
     *
     * @param sessionId - Session UUID filter ("" for default/all sessions).
     * @param limit     - Maximum events to return (capped at 1000).
     * @returns Array of { id, role, text, timestamp } sorted newest-first.
     * @throws {Error} If the C-API call fails.
     */
    traceGetHistory(sessionId: string, limit: number): TraceHistoryEvent[];

    /**
     * Explicitly release all native resources.
     *
     * Calls `aeon_atlas_destroy` and `aeon_trace_destroy`, then nullifies
     * internal handles. Safe to call multiple times (idempotent).
     * All subsequent method calls will throw.
     *
     * MUST be called when done to prevent memory/file-handle leaks.
     * The destructor will also call this automatically during GC,
     * but GC timing is non-deterministic — always call close() explicitly.
     */
    close(): void;

    /**
     * Check if this instance has been closed.
     * @returns true if close() has been called, false otherwise.
     */
    isClosed(): boolean;
}

/**
 * Aeon SDK version string (e.g. "4.1.0").
 */
export const version: string;
