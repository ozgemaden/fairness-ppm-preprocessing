# fairness-ppm-preprocessing

Code and result summaries for the master thesis:

> **Fairness in Predictive Process Monitoring: The Impact of Pre-processing
> Design Choices on Bias and Implications for Mitigation**  
> (Özge Maden, Humboldt-Universität zu Berlin, 2026)

This repository contains the experimental pipeline used in the thesis and the
main result tables for all datasets and configurations.

## Repository structure

- `run_final_pipeline_all_datasets.py`  
  Entry script that runs the full experiment for all event logs.

- `src/`  
  Python modules for:
  - loading event logs and applying a fixed case-level train/validation/test split  
  - prefix generation:
    - **baseline** – one vector per trace (longest prefix only, `only-this`)
    - **all-in-one** – multiple vectors per trace (all prefixes up to \(k\))
  - encoding:
    - `simple`, `frequency`, `complex` (includes all case-level attributes present in the log
      plus the standard resource event attribute `org:resource`/`Resource`; the attribute set
      therefore differs per dataset — e.g., `Resource` only for DomesticDeclarations,
      clinical attributes such as `Age` and `Leucocytes` for SepsisCases, and financial
      attributes such as `TotalDeclared` and `RfpNumber` for the BPI 2020 logs)
  - bias metrics:
    - outcome distribution shift (original vs. encoded)
    - trace-level representation (multiplication ratio)
    - attribute–outcome statistics (for complex encoding)
  - preprocessing-based mitigation:
    - duplicate and conflict removal (same prefix, different outcome)
  - model training and evaluation:
    - Random Forest and Decision Tree classifiers

- `results/`  
  - `*_results_summary.csv`: for each of the seven event logs,
    20 configurations per dataset  
    (1 baseline + 9 variations × with/without mitigation), including  
    `total_feature_vectors`, `multiplication_ratio`, `f1_rf`, `f1_dt`,
    and outcome distributions (original / encoded / change).  
  - `*_results_detailed.json` (for some datasets) and
    `attribute_analysis/*.json` provide additional statistics used in the thesis.  
  - Large encoded CSV files and full outcome-distribution JSON files are included
    here for reproducibility; they are generated automatically by the pipeline.

## Data

The original event logs are **not** included in this repository.  
They can be obtained from 4TU.ResearchData and the IEEE Task Force on Process Mining
resource list, as referenced in the thesis (Sepsis Cases, Road Traffic Fine Management
and the BPI 2020 logs).

Update the dataset paths in `src/dataset_config.py` (`DATASET_PATHS` list)
to point to your local copies of the CSV event logs.

## Installation

Tested with Python 3.10+.

```bash
git clone https://github.com/ozgemaden/fairness-ppm-preprocessing.git
cd fairness-ppm-preprocessing

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
# plus nirdizati-light if not already installed:
# pip install nirdizati-light
```

## Running the experiments

After configuring the dataset paths, run:

```bash
python run_final_pipeline_all_datasets.py
```

For each event log, this will:

1. Create a fixed case-level train/validation/test split.
2. Generate encoded datasets for:
   - prefix lengths: min / avg / max trace length per log,
   - encodings: simple / frequency / complex,
   - prefix-generation strategies: baseline (one vector per trace, longest prefix only)
     vs. all-in-one (multiple vectors per trace, all prefixes up to \(k\)),
   - with and without mitigation (duplicate and conflict removal).
3. Compute bias metrics and model performance and write the summary tables
   to `results/*_results_summary.csv`.

## Citation

If you use this code or the result tables, please cite:

> Özge Maden, *Fairness in Predictive Process Monitoring: The Impact of
> Pre-processing Design Choices on Bias and Implications for Mitigation*,
> Master thesis, Humboldt-Universität zu Berlin, 2026.
