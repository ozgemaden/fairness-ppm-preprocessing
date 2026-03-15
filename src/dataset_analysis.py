"""
Dataset characteristic analysis.

Computes core statistics (trace lengths, variants, activities, attributes)
and simple variance/homogeneity indicators per event log.
"""

import pandas as pd
import numpy as np
from pm4py.objects.log.obj import EventLog
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_trace_lengths(log: EventLog) -> List[int]:
    """Return a list with the length of each trace."""
    return [len(trace) for trace in log]


def calculate_trace_statistics(log: EventLog) -> Dict[str, float]:
    """Compute basic statistics over trace lengths."""
    trace_lengths = calculate_trace_lengths(log)
    
    return {
        'min': int(min(trace_lengths)),
        'max': int(max(trace_lengths)),
        'avg': int(np.mean(trace_lengths)),
        'median': int(np.median(trace_lengths)),
        'std': float(np.std(trace_lengths))
    }


def count_variants(log: EventLog) -> int:
    """Count the number of unique trace patterns (variants)."""
    variants = set()
    for trace in log:
        # Represent trace as an activity-sequence tuple
        trace_str = tuple([event['concept:name'] for event in trace])
        variants.add(trace_str)
    return len(variants)


def count_activities(log: EventLog) -> int:
    """Count the number of unique activities in the log."""
    activities = set()
    for trace in log:
        for event in trace:
            activities.add(event['concept:name'])
    return len(activities)


def count_attributes(log: EventLog) -> Dict[str, int]:
    """Count the number of distinct trace and event attributes."""
    trace_attributes = set()
    event_attributes = set()
    
    for trace in log:
        # Trace attributes
        for attr in trace.attributes.keys():
            if attr not in ['concept:name', 'time:timestamp']:
                trace_attributes.add(attr)
        
        # Event attributes
        for event in trace:
            for attr in event.keys():
                if attr not in ['concept:name', 'time:timestamp']:
                    event_attributes.add(attr)
    
    return {
        'trace_attributes': len(trace_attributes),
        'event_attributes': len(event_attributes),
        'total_attributes': len(trace_attributes) + len(event_attributes)
    }


def analyze_dataset(log: EventLog, dataset_name: str = "Unknown") -> Dict:
    """
    Analyse an event log and compute its main characteristics.

    Args:
        log: PM4Py EventLog instance.
        dataset_name: Human-readable dataset name.

    Returns:
        Dictionary containing all characteristics.
    """
    logger.info(f"Analyzing dataset: {dataset_name}")
    
    # Basic statistics
    num_traces = len(log)
    num_events = sum(len(trace) for trace in log)
    
    # Trace-length statistics
    trace_stats = calculate_trace_statistics(log)
    
    # Number of variants
    num_variants = count_variants(log)
    
    # Number of activities
    num_activities = count_activities(log)
    
    # Attribute counts
    attr_counts = count_attributes(log)
    
    # Variant ratio (variants per trace)
    trace_lengths = calculate_trace_lengths(log)
    variance_ratio = num_variants / num_traces if num_traces > 0 else 0
    
    # Homogeneity indicator based on trace-length std/avg
    trace_length_std = trace_stats['std']
    trace_length_avg = trace_stats['avg']
    homogeneity_ratio = trace_length_std / trace_length_avg if trace_length_avg > 0 else 0
    
    characteristics = {
        'dataset_name': dataset_name,
        'num_traces': num_traces,
        'num_events': num_events,
        'num_variants': num_variants,
        'variance_ratio': variance_ratio,
        'num_activities': num_activities,
        'trace_length_min': trace_stats['min'],
        'trace_length_max': trace_stats['max'],
        'trace_length_avg': trace_stats['avg'],
        'trace_length_median': trace_stats['median'],
        'trace_length_std': trace_stats['std'],
        'homogeneity_ratio': homogeneity_ratio,
        'trace_attributes': attr_counts['trace_attributes'],
        'event_attributes': attr_counts['event_attributes'],
        'total_attributes': attr_counts['total_attributes'],
        'prefix_lengths': {
            'min': trace_stats['min'],
            'avg': trace_stats['avg'],
            'max': trace_stats['max']
        }
    }
    
    logger.info(f"Analysis complete for {dataset_name}")
    return characteristics


def classify_dataset(characteristics: Dict) -> Dict[str, str]:
    """
    Classify a dataset into coarse variance and homogeneity categories.

    Returns:
        Dictionary with string classifications.
    """
    variance_ratio = characteristics['variance_ratio']
    homogeneity_ratio = characteristics['homogeneity_ratio']
    
    # Variance classification
    if variance_ratio > 0.5:
        variance_class = 'high'
    elif variance_ratio > 0.2:
        variance_class = 'medium'
    else:
        variance_class = 'low'
    
    # Homogeneity classification
    if homogeneity_ratio < 0.3:
        homogeneity_class = 'homogeneous'
    elif homogeneity_ratio < 0.7:
        homogeneity_class = 'moderate'
    else:
        homogeneity_class = 'heterogeneous'
    
    return {
        'variance_class': variance_class,
        'homogeneity_class': homogeneity_class
    }


def print_dataset_summary(characteristics: Dict):
    """Print a human-readable summary of dataset characteristics."""
    import sys
    import io
    
    # Safe printing helper for terminals that may have encoding issues
    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            safe_text = str(text).encode('ascii', 'replace').decode('ascii')
            print(safe_text)
    
    safe_print(f"\n{'='*60}")
    safe_print(f"Dataset: {characteristics['dataset_name']}")
    safe_print(f"{'='*60}")
    safe_print(f"Traces: {characteristics['num_traces']:,}")
    safe_print(f"Events: {characteristics['num_events']:,}")
    safe_print(f"Variants: {characteristics['num_variants']:,} (Ratio: {characteristics['variance_ratio']:.3f})")
    safe_print(f"Activities: {characteristics['num_activities']}")
    safe_print(f"\nTrace Length Statistics:")
    safe_print(f"  Min: {characteristics['trace_length_min']}")
    safe_print(f"  Max: {characteristics['trace_length_max']}")
    safe_print(f"  Avg: {characteristics['trace_length_avg']:.2f}")
    safe_print(f"  Median: {characteristics['trace_length_median']}")
    safe_print(f"  Std: {characteristics['trace_length_std']:.2f}")
    safe_print(f"\nPrefix Lengths (for experiments):")
    safe_print(f"  Min: {characteristics['prefix_lengths']['min']}")
    safe_print(f"  Avg: {characteristics['prefix_lengths']['avg']}")
    safe_print(f"  Max: {characteristics['prefix_lengths']['max']}")
    safe_print(f"\nAttributes:")
    safe_print(f"  Trace attributes: {characteristics['trace_attributes']}")
    safe_print(f"  Event attributes: {characteristics['event_attributes']}")
    safe_print(f"  Total: {characteristics['total_attributes']}")
    
    classifications = classify_dataset(characteristics)
    safe_print(f"\nClassifications:")
    safe_print(f"  Variance: {classifications['variance_class']}")
    safe_print(f"  Homogeneity: {classifications['homogeneity_class']}")
    safe_print(f"{'='*60}\n")
