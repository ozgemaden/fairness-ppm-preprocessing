"""
Dataset loader utilities.

Load event logs from CSV/XES and convert them into a PM4Py EventLog in the
format expected by Nirdizati-light (case id, activity, timestamp).
"""

import pandas as pd
import pm4py
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_and_convert_dataset(filepath: str, separator: str = ',') -> pm4py.objects.log.obj.EventLog:
    """
    Load a dataset and convert it to a Nirdizati-light-compatible EventLog.

    Args:
        filepath: Dataset file path.
        separator: CSV separator (default: ',').

    Returns:
        PM4Py EventLog instance.
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.xes':
        # Directly load XES files
        log = pm4py.read_xes(str(filepath))
        return log
    
    # Load CSV (low_memory=False to avoid mixed-type warnings)
    df = pd.read_csv(filepath, sep=separator, low_memory=False)
    
    # Column mapping – support slightly different CSV schemas
    column_mapping = {}
    
    # Case ID mapping
    if 'Case ID' in df.columns:
        column_mapping['Case ID'] = 'case:concept:name'
    elif 'case:concept:name' not in df.columns:
        # Fallback: treat the first column as case ID if it looks like it
        first_col = df.columns[0]
        if 'case' in first_col.lower() or 'id' in first_col.lower():
            column_mapping[first_col] = 'case:concept:name'
    
    # Activity mapping
    if 'Activity' in df.columns:
        column_mapping['Activity'] = 'concept:name'
    elif 'concept:name' not in df.columns:
        # Try to find a column that looks like an activity/event name
        for col in df.columns:
            if 'activity' in col.lower() or 'event' in col.lower():
                column_mapping[col] = 'concept:name'
                break
    
    # Timestamp mapping
    if 'Complete Timestamp' in df.columns:
        column_mapping['Complete Timestamp'] = 'time:timestamp'
    elif 'time:timestamp' not in df.columns:
        # Try to find a column that looks like a timestamp
        for col in df.columns:
            if 'timestamp' in col.lower() or 'time' in col.lower():
                column_mapping[col] = 'time:timestamp'
                break
    
    # Rename columns to the standard keys
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logger.info(f"Renamed columns: {column_mapping}")
    
    # Check that all required columns are present
    required_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}. "
            f"Please ensure your dataset has columns for case ID, activity, and timestamp."
        )
    
    # Convert timestamp to datetime
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    
    # Convert case ID to string
    df['case:concept:name'] = df['case:concept:name'].astype(str)
    
    # Convert to EventLog
    log = pm4py.convert_to_event_log(df, case_id_key='case:concept:name')
    
    logger.info(f"Successfully loaded dataset: {len(log)} traces, {sum(len(t) for t in log)} events")
    
    return log
