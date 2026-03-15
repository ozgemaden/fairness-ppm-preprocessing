"""
Final pipeline entrypoint for running all experiments.

Runs the comprehensive fairness/bias pipeline for all configured real-world
datasets using the implementation in `src/`, and writes results to the
`results/` directory.

Usage (from the project root):

    python run_final_pipeline_all_datasets.py

This script does not contain any modelling logic; it only wires together:
- `src/dataset_config.py`  → list of datasets
- `src/comprehensive_pipeline.py` → ComprehensivePipeline implementation
"""

import sys
from pathlib import Path


def _add_project_root_to_path() -> Path:
    """
    Ensure the main project root (where `src/` lives) and the local
    `nirdizati-light` package are on sys.path, so that the thesis-specific
    copy of `nirdizati_light` is used instead of any globally installed one.

    File layout assumption:
        masterarbeit/  ← project root
          src/
          datasets/
          results/
          nirdizati-light/
          run_final_pipeline_all_datasets.py
    """
    this_file = Path(__file__).resolve()
    project_root = this_file.parent

    # Add project root itself
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Prefer the local `nirdizati-light` copy over any global installation
    local_nirdizati = project_root / "nirdizati-light"
    if local_nirdizati.exists() and str(local_nirdizati) not in sys.path:
        sys.path.insert(0, str(local_nirdizati))

    return project_root


PROJECT_ROOT = _add_project_root_to_path()

# Imports from the existing implementation
from nirdizati_light.encoding.common import EncodingType  # type: ignore  # noqa: E402
from nirdizati_light.encoding.constants import TaskGenerationType  # type: ignore  # noqa: E402

from src.dataset_config import DATASET_PATHS  # type: ignore  # noqa: E402
from src.comprehensive_pipeline import ComprehensivePipeline  # type: ignore  # noqa: E402


def run_for_dataset(dataset_path: Path, output_dir: Path) -> None:
    """Run ComprehensivePipeline for a single dataset with the thesis design."""
    if not dataset_path.exists():
        print(f"[SKIP] Missing dataset: {dataset_path}")
        return

    print(f"\n=== FINAL PIPELINE: {dataset_path.stem} ===")
    pipeline = ComprehensivePipeline(str(dataset_path), output_dir=str(output_dir), seed=42)

    # 1) BASELINE: ONLY_THIS, prefix = max, encoding = simple
    print("  -> Baseline (ONLY_THIS, prefix=max, encoding=simple) without mitigation ...")
    pipeline.run_full_pipeline(
        encoding_types=[EncodingType.SIMPLE.value],
        generation_types=[TaskGenerationType.ONLY_THIS.value],
        apply_preprocessing=False,
        prefix_length_names=["max"],
    )

    print("  -> Baseline (ONLY_THIS, prefix=max, encoding=simple) with mitigation ...")
    pipeline.run_full_pipeline(
        encoding_types=[EncodingType.SIMPLE.value],
        generation_types=[TaskGenerationType.ONLY_THIS.value],
        apply_preprocessing=True,
        prefix_length_names=["max"],
    )

    # 2) VARIATIONS: ALL_IN_ONE, prefix ∈ {min, avg, max}, encoding ∈ {simple, frequency, complex}
    print("  -> Variations (ALL_IN_ONE, 3 prefix × 3 encoding) without mitigation ...")
    pipeline.run_full_pipeline(
        encoding_types=[
            EncodingType.SIMPLE.value,
            EncodingType.FREQUENCY.value,
            EncodingType.COMPLEX.value,
        ],
        generation_types=[TaskGenerationType.ALL_IN_ONE.value],
        apply_preprocessing=False,
        prefix_length_names=["min", "avg", "max"],
    )

    print("  -> Variations (ALL_IN_ONE, 3 prefix × 3 encoding) with mitigation ...")
    pipeline.run_full_pipeline(
        encoding_types=[
            EncodingType.SIMPLE.value,
            EncodingType.FREQUENCY.value,
            EncodingType.COMPLEX.value,
        ],
        generation_types=[TaskGenerationType.ALL_IN_ONE.value],
        apply_preprocessing=True,
        prefix_length_names=["min", "avg", "max"],
    )


def main() -> None:
    # Final output directory for all experiments
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds in DATASET_PATHS:
        # DATASET_PATHS is relative to project root in the original implementation
        dataset_path = (PROJECT_ROOT / ds).resolve()
        run_for_dataset(dataset_path, output_dir=output_dir)

    print(f"\n=== Pipeline finished. Results in: {output_dir} ===\n")


if __name__ == "__main__":
    main()

