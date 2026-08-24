#!/usr/bin/env python3
"""
Generates figures from a LongMemEval benchmark results JSON
(scripts/longmemeval/run_benchmark.py's --out file), for inclusion in v4
docs (v4-plan.md Stage 6). Mirrors scripts/plot_results.py's style
conventions (serif, 300dpi, colorblind-friendly palette) for visual
consistency with the existing paper figures.

Usage:
    python scripts/longmemeval/plot_results.py \\
        reproducibility_benchmarks/longmemeval/pilot_50_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})

BAR_COLOR = '#2E86AB'
RECALL_COLOR = '#10B981'
LATENCY_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#6B7280']


def plot_accuracy_by_type(summary: dict, out_dir: Path) -> None:
    per_type = summary["per_question_type"]
    types = sorted(per_type.keys(), key=lambda t: -per_type[t]["n"])
    accs = [per_type[t]["accuracy"] * 100 for t in types]
    ns = [per_type[t]["n"] for t in types]

    labels = [f"{t}\n(n={n})" for t, n in zip(types, ns)]
    labels.append(f"OVERALL\n(n={summary['num_questions']})")
    values = accs + [summary["overall_accuracy"] * 100]
    colors = [BAR_COLOR] * len(types) + ['#C73E1D']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%",
                 ha='center', va='bottom', fontsize=8)
    ax.set_ylabel("QA accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"LongMemEval-S accuracy by question type — {summary['model']}, top_k={summary['top_k']}")
    plt.xticks(rotation=20, ha='right')
    fig.tight_layout()
    fig.savefig(out_dir / "longmemeval_accuracy_by_type.png")
    plt.close(fig)


def plot_retrieval_recall(summary: dict, out_dir: Path) -> None:
    per_type = summary["per_question_type"]
    types = sorted(per_type.keys(), key=lambda t: -per_type[t]["n"])
    recalls = [per_type[t]["gold_session_recall_at_k"] * 100 for t in types]
    labels = types + ["OVERALL"]
    values = recalls + [summary["overall_gold_session_recall_at_k"] * 100]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [RECALL_COLOR] * len(types) + ['#065F46']
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%",
                 ha='center', va='bottom', fontsize=8)
    ax.set_ylabel(f"Gold-session recall@{summary['top_k']} (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Did Trace retrieval surface the right session at all?")
    plt.xticks(rotation=20, ha='right')
    fig.tight_layout()
    fig.savefig(out_dir / "longmemeval_retrieval_recall.png")
    plt.close(fig)


def plot_latency(summary: dict, out_dir: Path) -> None:
    # "search" is the ONLY phase here that measures Aeon's own kernel work
    # (TraceManager.semantic_search); ingest/query_encode are Python-side
    # sentence-transformers encoding, generation/judge are the local LLM --
    # kept as separate bars specifically so this chart can't be misread as
    # attributing encoder/LLM latency to Aeon itself (v4-plan.md Stage 6).
    phases = [p for p in ("ingest", "query_encode", "search", "generation", "judge")
              if p in summary["latency_seconds"]]
    labels = {
        "ingest": "Ingest\n(haystack, encoder)",
        "query_encode": "Query encode\n(encoder)",
        "search": "Search\n(Aeon semantic_search)",
        "generation": "Generation\n(LLM answer)",
        "judge": "Judge\n(LLM score)",
    }
    means = [summary["latency_seconds"][p]["mean"] for p in phases]
    p95s = [summary["latency_seconds"][p]["p95"] for p in phases]

    x = np.arange(len(phases))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width / 2, means, width, label="mean", color=LATENCY_COLORS[0])
    ax.bar(x + width / 2, p95s, width, label="p95", color=LATENCY_COLORS[1])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[p] for p in phases])
    ax.set_yscale("log")
    ax.set_ylabel("Seconds per question (log scale)")
    ax.set_title(f"Per-phase latency — {summary['model']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "longmemeval_latency.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)
    summary = data["summary"]

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_json).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_by_type(summary, out_dir)
    plot_retrieval_recall(summary, out_dir)
    plot_latency(summary, out_dir)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
