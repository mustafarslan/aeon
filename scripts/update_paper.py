#!/usr/bin/env python3
"""
Aeon LaTeX Benchmark Injector

Reads benchmark results from JSON and injects them into LaTeX paper files,
replacing placeholder variables with actual measured values.

Placeholders use the format: \\VAR{metric_name}

Example:
    The warm search latency is \\VAR{warm_search_p99}ms.
    
Becomes:
    The warm search latency is 1.42ms.

Safety: Never overwrites source files. Creates paper_final.tex as output.

Usage:
    python scripts/update_paper.py [--input results/benchmark_final.json]
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PAPER_DIR = PROJECT_ROOT / "paper"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_INPUT = RESULTS_DIR / "benchmark_final.json"

# LaTeX source files to process
SOURCE_FILES = [
    PAPER_DIR / "paper.tex",
    PAPER_DIR / "paper_preprint.tex",
]

# Output file (never overwrites source)
OUTPUT_FILE = PAPER_DIR / "paper_final.tex"

# Placeholder pattern: \VAR{variable_name}
VAR_PATTERN = re.compile(r"\\VAR\{([a-zA-Z0-9_]+)\}")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class InjectionResult:
    """Result of LaTeX injection process."""
    source_file: Path
    output_file: Path
    replacements_made: int
    unresolved_vars: list[str]
    timestamp: str
    success: bool
    error: Optional[str] = None


# =============================================================================
# Metrics Extraction
# =============================================================================

def extract_metrics(benchmark_data: dict) -> dict[str, float]:
    """
    Extract flat metric dictionary from benchmark JSON.
    
    Creates keys like:
        - math_kernel_mean
        - warm_search_p99
        - cold_search_median
        - etc.
    """
    metrics = {}
    
    benchmarks = benchmark_data.get("benchmarks", {})
    
    for bench_name, bench_data in benchmarks.items():
        # Normalize benchmark name to valid variable name
        # e.g., "AtlasFixture/WarmSearch" -> "warm_search"
        clean_name = bench_name.lower()
        clean_name = re.sub(r"[^a-z0-9]+", "_", clean_name)
        clean_name = clean_name.strip("_")
        
        # Remove common prefixes
        for prefix in ["atlasfixture_", "bm_", "benchmark_"]:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
        
        # Extract stat values
        for stat_key in ["mean", "median", "p99", "stddev", "min_val", "max_val"]:
            if stat_key in bench_data:
                metric_key = f"{clean_name}_{stat_key}"
                metrics[metric_key] = bench_data[stat_key]
        
        # Also store unit for reference
        if "unit" in bench_data:
            metrics[f"{clean_name}_unit"] = bench_data["unit"]
    
    # Add some computed convenience metrics
    if "warmsearch_mean" in metrics:
        metrics["latency_warm_mean"] = metrics["warmsearch_mean"]
    if "warmsearch_p99" in metrics:
        metrics["latency_warm_p99"] = metrics["warmsearch_p99"]
    if "coldsearch_mean" in metrics:
        metrics["latency_cold_mean"] = metrics["coldsearch_mean"]
    if "coldsearch_p99" in metrics:
        metrics["latency_cold_p99"] = metrics["coldsearch_p99"]
    if "mathkernel_mean" in metrics:
        metrics["kernel_latency_ns"] = metrics["mathkernel_mean"]
    
    return metrics


def format_value(value: float, precision: int = 2) -> str:
    """Format a numeric value for LaTeX output."""
    if isinstance(value, str):
        return value
    
    # Use appropriate precision based on magnitude
    if abs(value) >= 1000:
        return f"{value:,.{precision}f}"
    elif abs(value) >= 1:
        return f"{value:.{precision}f}"
    elif abs(value) >= 0.01:
        return f"{value:.{min(precision + 2, 4)}f}"
    else:
        return f"{value:.{min(precision + 4, 6)}f}"


# =============================================================================
# LaTeX Processing
# =============================================================================

def find_source_file() -> Optional[Path]:
    """Find the first available source file."""
    for source in SOURCE_FILES:
        if source.exists():
            return source
    return None


def process_latex_file(
    source_path: Path,
    metrics: dict[str, float],
    output_path: Path,
    precision: int = 2
) -> InjectionResult:
    """
    Process a LaTeX file, replacing VAR placeholders with actual values.
    """
    result = InjectionResult(
        source_file=source_path,
        output_file=output_path,
        replacements_made=0,
        unresolved_vars=[],
        timestamp=datetime.now().isoformat(),
        success=False
    )
    
    try:
        content = source_path.read_text(encoding="utf-8")
    except Exception as e:
        result.error = f"Failed to read source file: {e}"
        return result
    
    # Find all VAR placeholders
    all_vars = VAR_PATTERN.findall(content)
    unique_vars = set(all_vars)
    
    print(f"\n  Found {len(unique_vars)} unique placeholders in {source_path.name}")
    
    # Replace each placeholder
    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        
        if var_name in metrics:
            value = metrics[var_name]
            formatted = format_value(value, precision)
            result.replacements_made += 1
            print(f"    ✓ \\VAR{{{var_name}}} → {formatted}")
            return formatted
        else:
            result.unresolved_vars.append(var_name)
            print(f"    ⚠️  \\VAR{{{var_name}}} → [UNRESOLVED]")
            return match.group(0)  # Keep original
    
    processed_content = VAR_PATTERN.sub(replace_var, content)
    
    # Add generation header comment
    header = f"""% ============================================================================
% AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
% Generated by: scripts/update_paper.py
% Timestamp: {result.timestamp}
% Source: {source_path.name}
% Replacements: {result.replacements_made}
% ============================================================================

"""
    
    final_content = header + processed_content
    
    # Write output file
    try:
        output_path.write_text(final_content, encoding="utf-8")
        result.success = True
        print(f"\n  ✓ Output written to: {output_path}")
    except Exception as e:
        result.error = f"Failed to write output file: {e}"
    
    return result


def inject_into_evaluation(
    metrics: dict[str, float],
    precision: int = 2
) -> Optional[InjectionResult]:
    """
    Inject metrics into evaluation.tex if it contains placeholders.
    Creates evaluation_final.tex as output.
    """
    eval_source = PAPER_DIR / "evaluation.tex"
    eval_output = PAPER_DIR / "evaluation_final.tex"
    
    if not eval_source.exists():
        return None
    
    content = eval_source.read_text(encoding="utf-8")
    if not VAR_PATTERN.search(content):
        print(f"\n  No placeholders found in {eval_source.name}")
        return None
    
    return process_latex_file(eval_source, metrics, eval_output, precision)


# =============================================================================
# Main Execution
# =============================================================================

def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    width = 70
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def print_available_metrics(metrics: dict[str, float]) -> None:
    """Print all available metrics for reference."""
    print("\n  Available metrics for use in \\VAR{...} placeholders:")
    for key, value in sorted(metrics.items()):
        if not key.endswith("_unit"):
            formatted = format_value(value)
            print(f"    {key}: {formatted}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aeon LaTeX Benchmark Injector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Benchmark results JSON file (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output LaTeX file (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--precision", "-p",
        type=int,
        default=2,
        help="Decimal precision for formatted values (default: 2)"
    )
    parser.add_argument(
        "--list-metrics", "-l",
        action="store_true",
        help="List available metrics and exit"
    )
    
    args = parser.parse_args()
    
    print_header("AEON LATEX BENCHMARK INJECTOR")
    
    # Load benchmark results
    if not args.input.exists():
        print(f"\n  ERROR: Benchmark results not found: {args.input}")
        print("  Run `python scripts/verify_benchmarks.py` first.")
        return 1
    
    try:
        with open(args.input) as f:
            benchmark_data = json.load(f)
        print(f"\n  Loaded benchmark data from: {args.input}")
        print(f"  Timestamp: {benchmark_data.get('timestamp', 'unknown')}")
    except Exception as e:
        print(f"\n  ERROR: Failed to load benchmark data: {e}")
        return 1
    
    # Extract metrics
    metrics = extract_metrics(benchmark_data)
    print(f"  Extracted {len(metrics)} metrics")
    
    if args.list_metrics:
        print_available_metrics(metrics)
        return 0
    
    # Find source file
    source_file = find_source_file()
    if not source_file:
        print(f"\n  ERROR: No source LaTeX file found in {PAPER_DIR}")
        print(f"  Expected one of: {[f.name for f in SOURCE_FILES]}")
        return 1
    
    print(f"\n  Source file: {source_file}")
    
    # Process main paper file
    print_header("Processing Main Paper", "-")
    main_result = process_latex_file(
        source_file,
        metrics,
        args.output,
        args.precision
    )
    
    # Also process evaluation.tex if it has placeholders
    print_header("Processing Evaluation Section", "-")
    eval_result = inject_into_evaluation(metrics, args.precision)
    
    # Summary
    print_header("SUMMARY")
    
    results = [main_result]
    if eval_result:
        results.append(eval_result)
    
    total_replacements = sum(r.replacements_made for r in results)
    all_unresolved = []
    for r in results:
        all_unresolved.extend(r.unresolved_vars)
    
    print(f"  Files processed:    {len(results)}")
    print(f"  Total replacements: {total_replacements}")
    print(f"  Unresolved vars:    {len(set(all_unresolved))}")
    
    if all_unresolved:
        print("\n  Unresolved placeholders (add to LaTeX or run benchmarks):")
        for var in sorted(set(all_unresolved)):
            print(f"    - \\VAR{{{var}}}")
    
    # Check for errors
    all_success = all(r.success for r in results)
    if all_success:
        print(f"\n  ✓ All files processed successfully!")
        print(f"\n  Output files:")
        for r in results:
            print(f"    - {r.output_file}")
    else:
        print(f"\n  ❌ Some files failed to process:")
        for r in results:
            if not r.success:
                print(f"    - {r.source_file}: {r.error}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
