"""
Fixed test set methodology.

Implements a case-based train/validation/test split where the test set is
kept fixed across all experimental configurations.
"""

import random
import logging
from typing import Tuple, List
from pm4py.objects.log.obj import EventLog
import pm4py

logger = logging.getLogger(__name__)


def get_case_ids(log: EventLog) -> List[str]:
    """Extract all case IDs from an event log."""
    case_ids = []
    for trace in log:
        case_id = trace.attributes.get('concept:name', None)
        if case_id:
            case_ids.append(str(case_id))
    return case_ids


def split_cases(case_ids: List[str], 
                train_size: float = 0.8, 
                val_size: float = 0.1, 
                test_size: float = 0.1,
                shuffle: bool = False,
                seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split case IDs into train/validation/test sets.

    Args:
        case_ids: List of all case IDs.
        train_size: Proportion of cases in the training split (default: 0.8).
        val_size: Proportion of cases in the validation split (default: 0.1).
        test_size: Proportion of cases in the test split (default: 0.1).
        shuffle: Whether to shuffle case IDs (for a fixed test set, typically False).
        seed: Random seed.

    Returns:
        Tuple of (train_cases, val_cases, test_cases).
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Train, val, test sizes must sum to 1.0"
    
    if shuffle:
        random.seed(seed)
        case_ids = case_ids.copy()
        random.shuffle(case_ids)
    
    total_cases = len(case_ids)
    train_end = int(train_size * total_cases)
    val_end = train_end + int(val_size * total_cases)
    
    train_cases = case_ids[:train_end]
    val_cases = case_ids[train_end:val_end]
    test_cases = case_ids[val_end:]
    
    logger.info(f"Split: Train={len(train_cases)}, Val={len(val_cases)}, Test={len(test_cases)}")
    
    return train_cases, val_cases, test_cases


def filter_log_by_cases(log: EventLog, case_ids: List[str]) -> EventLog:
    """Filter an event log to keep only traces whose case IDs are in the given list."""
    filtered_traces = []
    case_ids_set = set(case_ids)
    
    for trace in log:
        case_id = str(trace.attributes.get('concept:name', ''))
        if case_id in case_ids_set:
            filtered_traces.append(trace)
    
    # Build a new EventLog instance with the selected traces
    filtered_log = pm4py.objects.log.obj.EventLog(filtered_traces)
    return filtered_log


def create_fixed_test_set(log: EventLog,
                          train_size: float = 0.8,
                          val_size: float = 0.1,
                          test_size: float = 0.1,
                          shuffle: bool = False,
                          seed: int = 42) -> Tuple[EventLog, EventLog, EventLog, List[str]]:
    """
    Create a fixed train/validation/test split following the fixed-test-set methodology.

    Important: for a reproducible fixed test set, `shuffle=False` should be used.

    Args:
        log: PM4Py EventLog instance.
        train_size: Train set proportion.
        val_size: Validation set proportion.
        test_size: Test set proportion.
        shuffle: Whether to shuffle case IDs before splitting.
        seed: Random seed.

    Returns:
        Tuple of (train_log, val_log, test_log, test_case_ids), where test_case_ids
        contains the case IDs used in the fixed test set.
    """
    logger.info("Creating fixed test set split...")
    
    # Get case IDs
    case_ids = get_case_ids(log)
    logger.info(f"Total cases: {len(case_ids)}")
    
    # Split
    train_cases, val_cases, test_cases = split_cases(
        case_ids, train_size, val_size, test_size, shuffle, seed
    )
    
    # Filter logs based on the selected case ID splits
    train_log = filter_log_by_cases(log, train_cases)
    val_log = filter_log_by_cases(log, val_cases)
    test_log = filter_log_by_cases(log, test_cases)
    
    logger.info(f"Fixed test set created: {len(test_cases)} test cases")
    
    return train_log, val_log, test_log, test_cases


def save_test_case_ids(test_case_ids: List[str], filepath: str):
    """Save test-set case IDs to disk (for reproducibility)."""
    import json
    with open(filepath, 'w') as f:
        json.dump(test_case_ids, f, indent=2)
    logger.info(f"Test case IDs saved to {filepath}")


def load_test_case_ids(filepath: str) -> List[str]:
    """Load test-set case IDs from disk."""
    import json
    with open(filepath, 'r') as f:
        test_case_ids = json.load(f)
    logger.info(f"Test case IDs loaded from {filepath}")
    return test_case_ids
