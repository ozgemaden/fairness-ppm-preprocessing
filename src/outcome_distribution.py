"""
Outcome distribution analysis.

Provides utilities to analyse and compare outcome distributions for
original event logs and encoded datasets.
"""

import pandas as pd
import numpy as np
from pm4py.objects.log.obj import EventLog, Trace
from typing import Dict, List
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def get_outcome_from_trace(trace: Trace) -> str:
    """
    Extract an outcome label from a trace.

    The label may be stored as a trace attribute, an event attribute, or
    implicitly as the last activity.
    """
    # First look for a trace-level label
    if 'label' in trace.attributes:
        return str(trace.attributes['label'])
    
    # Then look for an event-level label on the first event
    if len(trace) > 0 and 'label' in trace[0]:
        return str(trace[0]['label'])
    
    # Finally, use the last activity name as outcome (common convention)
    if len(trace) > 0:
        return str(trace[-1]['concept:name'])
    
    return None


def analyze_original_outcome_distribution(log: EventLog) -> Dict:
    """
    Analyse outcome distribution in the original event log.

    Returns:
        Dictionary with outcome distribution statistics.
    """
    outcomes = []
    trace_lengths = []
    outcome_by_length = {}
    
    for trace in log:
        outcome = get_outcome_from_trace(trace)
        trace_length = len(trace)
        
        if outcome:
            outcomes.append(outcome)
            trace_lengths.append(trace_length)
            
            if outcome not in outcome_by_length:
                outcome_by_length[outcome] = []
            outcome_by_length[outcome].append(trace_length)
    
    if not outcomes:
        logger.warning("No outcomes found in event log")
        return None
    
    # Outcome counts
    outcome_counts = Counter(outcomes)
    total_traces = len(outcomes)
    
    # Outcome distribution in percentage
    outcome_distribution = {
        outcome: (count / total_traces) * 100 
        for outcome, count in outcome_counts.items()
    }
    
    # Trace-length statistics per outcome
    outcome_length_stats = {}
    for outcome, lengths in outcome_by_length.items():
        outcome_length_stats[outcome] = {
            'count': len(lengths),
            'avg_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths)
        }
    
    return {
        'total_traces': total_traces,
        'outcome_counts': dict(outcome_counts),
        'outcome_distribution': outcome_distribution,
        'outcome_length_stats': outcome_length_stats,
        'unique_outcomes': list(outcome_counts.keys())
    }


def analyze_encoded_outcome_distribution(encoded_df: pd.DataFrame) -> Dict:
    """
    Analyse outcome distribution in an encoded DataFrame.

    Args:
        encoded_df: Encoded DataFrame produced by Nirdizati-light.

    Returns:
        Dictionary with outcome distribution statistics.
    """
    if 'label' not in encoded_df.columns:
        logger.error("'label' column not found in encoded DataFrame")
        return None
    
    # Outcome counts
    outcome_counts = Counter(encoded_df['label'])
    total_vectors = len(encoded_df)
    
    # Outcome distribution in percentage
    outcome_distribution = {
        str(outcome): (count / total_vectors) * 100 
        for outcome, count in outcome_counts.items()
    }
    
    # If available, compute feature vectors per trace_id
    if 'trace_id' in encoded_df.columns:
        vectors_per_trace = encoded_df.groupby('trace_id').size()
        trace_vector_stats = {
            'min_vectors_per_trace': int(vectors_per_trace.min()),
            'max_vectors_per_trace': int(vectors_per_trace.max()),
            'avg_vectors_per_trace': float(vectors_per_trace.mean()),
            'total_traces': len(vectors_per_trace)
        }
    else:
        trace_vector_stats = None
    
    return {
        'total_feature_vectors': total_vectors,
        'outcome_counts': {str(k): v for k, v in outcome_counts.items()},
        'outcome_distribution': outcome_distribution,
        'trace_vector_stats': trace_vector_stats,
        'unique_outcomes': [str(k) for k in outcome_counts.keys()]
    }


def compare_outcome_distributions(original_stats: Dict, encoded_stats: Dict) -> Dict:
    """
    Compare outcome distributions of original vs encoded data.

    Returns:
        Dictionary with comparison results.
    """
    if not original_stats or not encoded_stats:
        return None
    
    comparison = {
        'trace_count_change': {
            'original': original_stats['total_traces'],
            'encoded': encoded_stats.get('trace_vector_stats', {}).get('total_traces', 0),
            'change': encoded_stats.get('trace_vector_stats', {}).get('total_traces', 0) - original_stats['total_traces']
        },
        'feature_vector_count': encoded_stats['total_feature_vectors'],
        'multiplication_ratio': encoded_stats['total_feature_vectors'] / original_stats['total_traces'] if original_stats['total_traces'] > 0 else 0,
        'outcome_distribution_change': {}
    }
    
    # Outcome-distribution changes
    original_dist = original_stats['outcome_distribution']
    encoded_dist = encoded_stats['outcome_distribution']
    
    all_outcomes = set(list(original_dist.keys()) + list(encoded_dist.keys()))
    
    for outcome in all_outcomes:
        original_pct = original_dist.get(outcome, 0)
        encoded_pct = encoded_dist.get(outcome, 0)
        change = encoded_pct - original_pct
        
        comparison['outcome_distribution_change'][outcome] = {
            'original': original_pct,
            'encoded': encoded_pct,
            'change': change,
            'change_percentage': (change / original_pct * 100) if original_pct > 0 else 0
        }
    
    return comparison


def print_outcome_distribution_summary(stats: Dict, title: str = "Outcome Distribution"):
    """Print a human-readable summary of an outcome distribution."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    if 'total_traces' in stats:
        print(f"Total Traces: {stats['total_traces']:,}")
    if 'total_feature_vectors' in stats:
        print(f"Total Feature Vectors: {stats['total_feature_vectors']:,}")
    
    print(f"\nOutcome Distribution:")
    for outcome, percentage in stats['outcome_distribution'].items():
        count = stats['outcome_counts'].get(outcome, 0)
        print(f"  {outcome}: {count:,} ({percentage:.2f}%)")
    
    if 'trace_vector_stats' in stats and stats['trace_vector_stats']:
        print(f"\nFeature Vectors per Trace:")
        print(f"  Min: {stats['trace_vector_stats']['min_vectors_per_trace']}")
        print(f"  Max: {stats['trace_vector_stats']['max_vectors_per_trace']}")
        print(f"  Avg: {stats['trace_vector_stats']['avg_vectors_per_trace']:.2f}")
    
    print(f"{'='*60}\n")


def print_comparison_summary(comparison: Dict):
    """Print a human-readable summary of an outcome-distribution comparison."""
    print(f"\n{'='*60}")
    print(f"Outcome Distribution Comparison")
    print(f"{'='*60}")
    
    print(f"\nTrace Count:")
    print(f"  Original: {comparison['trace_count_change']['original']:,}")
    print(f"  Encoded: {comparison['trace_count_change']['encoded']:,}")
    print(f"  Change: {comparison['trace_count_change']['change']:,}")
    
    print(f"\nFeature Vectors: {comparison['feature_vector_count']:,}")
    print(f"Multiplication Ratio: {comparison['multiplication_ratio']:.2f}x")
    
    print(f"\nOutcome Distribution Changes:")
    for outcome, change_info in comparison['outcome_distribution_change'].items():
        print(f"  {outcome}:")
        print(f"    Original: {change_info['original']:.2f}%")
        print(f"    Encoded: {change_info['encoded']:.2f}%")
        print(f"    Change: {change_info['change']:+.2f}% ({change_info['change_percentage']:+.2f}%)")
    
    print(f"{'='*60}\n")
