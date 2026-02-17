/**
 * @file AeonOpenClawAdapter.ts
 * @brief Drop-in replacement for OpenClaw's JSONL-based session transcript I/O.
 *
 * DESIGN INVARIANTS:
 *   1. Zero-Overhead Abstraction — ultra-thin wrapper, no deep cloning.
 *   2. Synchronous Hot Paths — all calls are sync (no async/await/Promises).
 *   3. OpenClaw-Compatible Shapes — exact role strings OpenClaw expects.
 *
 * ROLE MAPPING (OpenClaw string ↔ Aeon uint16):
 *   "user"      ↔ 0    (AEON_ROLE_USER — native)
 *   "system"    ↔ 1    (AEON_ROLE_SYSTEM — native)
 *   "assistant" ↔ 10   (Extended: isolated from ROLE_CONCEPT=2)
 *   "tool"      ↔ 11   (Extended: isolated from ROLE_SUMMARY=3)
 *   "other"     ↔ 12   (Extended: unmapped catch-all)
 *
 * Roles 2-9 are RESERVED for Aeon's internal semantic graph. OpenClaw
 * roles are mapped to the 10+ range to prevent semantic corruption.
 *
 * @module @aeon/node-mac
 * @copyright 2024–2026 Aeon Project. All rights reserved.
 */

import type { AeonDB, TraceHistoryEvent } from "../index";

// ─── OpenClaw-Compatible Interfaces ─────────────────────────────────────────

/**
 * Matches OpenClaw's `SessionPreviewItem` shape from
 * `src/gateway/session-utils.types.ts`.
 *
 * OpenClaw's TranscriptMessage uses `content: string | Array<{type, text}>`,
 * but the actual preview/read path normalizes to flat text. We match the
 * minimal contract OpenClaw consumers expect.
 */
export interface OpenClawTurn {
    role: "user" | "assistant" | "tool" | "system" | "other";
    content: string;
}

// ─── Role Mapping (compile-time constant tables) ────────────────────────────

/** OpenClaw string → Aeon uint16 (write path). */
const ROLE_STR_TO_INT: Readonly<Record<string, number>> = {
    user: 0,
    system: 1,
    assistant: 10,
    tool: 11,
    other: 12,
} as const;

/**
 * Aeon uint16 → OpenClaw string (read path).
 * Sparse map — only populated for values we actually emit.
 * Roles 2-9 are Aeon-internal (Concept, Summary, etc.) and map to "other".
 */
const ROLE_INT_TO_STR: Readonly<Record<number, OpenClawTurn["role"]>> = {
    0: "user",
    1: "system",
    10: "assistant",
    11: "tool",
    12: "other",
} as const;

// ─── Adapter Class ──────────────────────────────────────────────────────────

/**
 * AeonOpenClawAdapter — Synchronous, zero-overhead TypeScript adapter.
 *
 * Intercepts OpenClaw's JSONL file I/O and routes it through the
 * Aeon C++23 native bridge for sub-10µs latency.
 *
 * @example
 * ```ts
 * import { AeonDB } from "@aeon/node-mac";
 * import { AeonOpenClawAdapter } from "./src/AeonOpenClawAdapter.js";
 *
 * const db = new AeonDB("/tmp/atlas.bin", "/tmp/trace.wal", 768);
 * const adapter = new AeonOpenClawAdapter(db);
 *
 * const id = adapter.saveTurn({ role: "user", content: "Hello, world!" });
 * const transcript = adapter.getTranscript(100);
 * ```
 */
export class AeonOpenClawAdapter {
    private readonly aeon: AeonDB;

    /**
     * @param aeon — An already-initialized AeonDB instance.
     *               Lifecycle ownership remains with the caller.
     */
    constructor(aeon: AeonDB) {
        this.aeon = aeon;
    }

    /**
     * Save a single OpenClaw turn into the Aeon Trace WAL.
     *
     * REPLACES: `fs.appendFileSync(transcriptPath, JSON.stringify({message}) + '\n')`
     *
     * ZERO-COPY: No JSON.stringify, no string concatenation, no newline append.
     * Directly maps the role string to a uint16 and calls the native C++ bridge.
     *
     * @param turn — The OpenClaw turn to persist.
     * @returns The monotonic Trace event ID (uint64 as bigint).
     */
    saveTurn(turn: OpenClawTurn): bigint {
        const roleInt: number = ROLE_STR_TO_INT[turn.role] ?? 12;
        return this.aeon.traceAppend(roleInt, turn.content);
    }

    /**
     * Retrieve the most recent N turns as OpenClaw-compatible objects.
     *
     * REPLACES: `fs.readFileSync(path).split('\n').map(JSON.parse)`
     *
     * Instead of reading and parsing the ENTIRE JSONL file (O(n) where n =
     * total session length), this performs a bounded O(limit) read from
     * the mmap'd Trace + Blob Arena — fetching ONLY the chronological tail.
     *
     * This eliminates OpenClaw's "Context Bloat" bottleneck where reading
     * a 10,000-turn JSONL file requires parsing ALL 10,000 lines just to
     * retrieve the last 100 for LLM prefill.
     *
     * @param limit — Maximum number of turns to retrieve. No silent cap.
     * @returns Array of OpenClawTurn objects, newest-first.
     */
    getTranscript(limit: number): OpenClawTurn[] {
        const events: TraceHistoryEvent[] = this.aeon.traceGetHistory("", limit);

        // Pre-allocate output array (avoid push/resize overhead)
        const result: OpenClawTurn[] = new Array<OpenClawTurn>(events.length);

        for (let i = 0; i < events.length; i++) {
            const evt: TraceHistoryEvent = events[i]!;
            result[i] = {
                role: ROLE_INT_TO_STR[evt.role] ?? "other",
                content: evt.text,
            };
        }

        return result;
    }

    /**
     * Get the total number of events in the Trace store.
     * Useful for diagnostics without loading full history.
     */
    size(): number {
        return this.aeon.traceSize();
    }
}
