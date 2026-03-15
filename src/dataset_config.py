"""
Central configuration of datasets used in the experiments.

All CSV event logs listed here are used by the pipeline; no manual limiting
to a subset is applied.
"""

from pathlib import Path

DATASET_PATHS = [
    "datasets/DomesticDeclarations.csv",
    "datasets/InternationalDeclarations.csv",
    "datasets/PermitLog.csv",
    "datasets/PrepaidTravelCost.csv",
    "datasets/RequestForPayment.csv",
    "datasets/Road_Traffic_Fine_Management_Process.csv",
    "datasets/SepsisCases.csv",
]

# Dataset names without file extensions
DATASET_NAMES = [Path(p).stem for p in DATASET_PATHS]
