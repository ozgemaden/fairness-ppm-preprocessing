"""
Quick analysis script to print summary statistics for all configured datasets.

Usage (from project root):
    python src/analyze_all_7_datasets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset_config import DATASET_PATHS
from src.dataset_loader import load_and_convert_dataset
from src.dataset_analysis import analyze_dataset

def main():
    results = []
    for path in DATASET_PATHS:
        path = Path(path)
        if not path.exists():
            print(f"[SKIP] Missing dataset: {path}")
            continue
        name = path.stem
        print(f"\nLoading: {name} ...")
        try:
            try:
                log = load_and_convert_dataset(str(path), separator=";")
            except Exception:
                log = load_and_convert_dataset(str(path), separator=",")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": name, "error": str(e)})
            continue
        try:
            char = analyze_dataset(log, name)
            results.append(char)
            print(f"  Traces: {char['num_traces']:,}, Events: {char['num_events']:,}, "
                  f"Activities: {char['num_activities']}, Trace len: {char['trace_length_min']}-{char['trace_length_max']} (avg {char['trace_length_avg']})")
        except Exception as e:
            print(f"  Analysis ERROR: {e}")
            results.append({"dataset_name": name, "error": str(e)})

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (all datasets)")
    print("=" * 100)
    header = f"{'Dataset':<45} {'Cases':>10} {'Events':>12} {'Min':>6} {'Avg':>6} {'Max':>6} {'Activities':>10} {'Variants':>10}"
    print(header)
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r.get('dataset_name', r.get('name', '?')):<45} ERROR: {r['error'][:40]}")
            continue
        print(f"{r['dataset_name']:<45} {r['num_traces']:>10,} {r['num_events']:>12,} "
              f"{r['trace_length_min']:>6} {r['trace_length_avg']:>6} {r['trace_length_max']:>6} "
              f"{r['num_activities']:>10} {r['num_variants']:>10,}")
    print("=" * 100)

    # Details (for each dataset)
    print("\n" + "=" * 100)
    print("DETAILS (per dataset)")
    print("=" * 100)
    for r in results:
        if "error" in r:
            continue
        print(f"\n--- {r['dataset_name']} ---")
        print(f"  Traces (cases):     {r['num_traces']:,}")
        print(f"  Events (total):     {r['num_events']:,}")
        print(f"  Trace length:       min={r['trace_length_min']}, avg={r['trace_length_avg']}, max={r['trace_length_max']}, std={r['trace_length_std']:.1f}")
        print(f"  Activities (unique): {r['num_activities']}")
        print(f"  Variants (unique trace pattern): {r['num_variants']:,}")
        print(f"  Variance ratio (variants/traces): {r['variance_ratio']:.3f}")
        print(f"  Trace attributes:   {r['trace_attributes']}, Event attributes: {r['event_attributes']}")

if __name__ == "__main__":
    main()
