"""
Dataset characteristics summary table utilities.

Builds compact tables summarizing key statistics per dataset.
"""

import pandas as pd
from typing import List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_summary_table(characteristics_list: List[Dict]) -> pd.DataFrame:
    """
    Create a summary table from per-dataset characteristic dictionaries.

    Args:
        characteristics_list: List of characteristic dicts, one per dataset.

    Returns:
        DataFrame with columns: Dataset, Cases, Min, Avg, Max, Variance, Activities.
    """
    rows = []
    
    for char in characteristics_list:
        dataset_name = char['dataset_name']
        
        # Shorten dataset names for display
        if 'Sepsis' in dataset_name or 'sepsis' in dataset_name.lower():
            display_name = 'Sepsis'
        elif 'Traffic' in dataset_name or 'traffic' in dataset_name.lower():
            display_name = 'Traffic Fine'
        elif 'BPI_Challenge_2017' in dataset_name or 'BPI 2017' in dataset_name:
            display_name = 'BPI 2017'
        elif 'BPI_Challenge_2019' in dataset_name or 'BPI 2019' in dataset_name:
            display_name = 'BPI 2019'
        elif 'BPIC11' in dataset_name or 'BPI 11' in dataset_name:
            display_name = 'BPIC11'
        else:
            display_name = dataset_name
        
        # Coarse variance classification
        variance_ratio = char['variance_ratio']
        if variance_ratio > 0.5:
            variance_label = 'High'
        elif variance_ratio > 0.2:
            variance_label = 'Med'
        else:
            variance_label = 'Low'
        
        # Show variance ratio with three decimal places
        variance_display = f"{variance_label} ({variance_ratio:.3f})"
        
        # Single table row
        row = {
            'Dataset': display_name,
            'Cases': char['num_traces'],
            'Min': char['trace_length_min'],
            'Avg': int(char['trace_length_avg']),
            'Max': char['trace_length_max'],
            'Variance': variance_display,
            'Activities': char['num_activities']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by number of cases (descending)
    df = df.sort_values('Cases', ascending=False)
    
    return df


def print_summary_table(characteristics_list: List[Dict], title: str = "Dataset Characteristics Summary"):
    """
    Print a formatted summary table for dataset characteristics.

    Args:
        characteristics_list: List of characteristic dictionaries, one per dataset.
        title: Title string for the printed table.
    """
    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            safe_text = str(text).encode('ascii', 'replace').decode('ascii')
            print(safe_text)
    
    df = create_summary_table(characteristics_list)
    
    safe_print("\n" + "="*100)
    safe_print(title)
    safe_print("="*100)
    
    # Configure table display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # Format: show Cases with thousands separators
    df_display = df.copy()
    df_display['Cases'] = df_display['Cases'].apply(lambda x: f"{x:,}")
    
    safe_print("\n" + df_display.to_string(index=False))
    safe_print("\n" + "="*100)
    
    return df


def save_summary_table(characteristics_list: List[Dict], 
                      output_path: str = 'results/dataset_characteristics_summary.csv'):
    """
    Save the dataset-characteristics summary table as CSV.

    Args:
        characteristics_list: List of characteristic dictionaries, one per dataset.
        output_path: Output file path.
    """
    df = create_summary_table(characteristics_list)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Summary table saved to {output_path}")
    
    return df
