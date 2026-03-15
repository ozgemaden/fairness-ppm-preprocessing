"""
Bias metrics: sampling bias and group-fairness indicators.

In this project, sampling bias is defined at the trace level: after prefix
generation, all feature vectors have the same dimensionality, but longer
traces produce more feature vectors. The `multiplication_ratio` metric
quantifies this trace-level representation bias (feature vectors per trace
as a function of trace length).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def calculate_multiplication_ratio(encoded_df: pd.DataFrame, 
                                   original_log_trace_lengths: Dict[str, int]) -> Dict:
    """
    Compute the multiplication ratio (trace-level representation bias).

    This is defined as:
        avg_vectors_for_longest_traces / avg_vectors_for_shortest_traces

    Args:
        encoded_df: Encoded DataFrame.
        original_log_trace_lengths: Mapping {trace_id: trace_length}.

    Returns:
        Dictionary with multiplication-ratio statistics.
    """
    logger.info("Calculating multiplication ratio...")
    
    if 'trace_id' not in encoded_df.columns:
        logger.warning("trace_id column not found, cannot calculate multiplication ratio")
        return None
    
    # Count feature vectors per trace_id
    vectors_per_trace = encoded_df.groupby('trace_id').size().to_dict()
    
    # Group feature vectors by trace length
    length_to_vectors = {}
    for trace_id, vector_count in vectors_per_trace.items():
        trace_length = original_log_trace_lengths.get(trace_id, 0)
        if trace_length not in length_to_vectors:
            length_to_vectors[trace_length] = []
        length_to_vectors[trace_length].append(vector_count)
    
    # Compute average feature vectors per trace length
    avg_vectors_by_length = {
        length: np.mean(counts) 
        for length, counts in length_to_vectors.items()
    }
    
    if len(avg_vectors_by_length) < 2:
        logger.warning("Not enough trace length diversity for multiplication ratio")
        return None
    
    # Find min and max lengths
    min_length = min(avg_vectors_by_length.keys())
    max_length = max(avg_vectors_by_length.keys())
    
    min_vectors = avg_vectors_by_length[min_length]
    max_vectors = avg_vectors_by_length[max_length]
    
    # Multiplication ratio
    multiplication_ratio = max_vectors / min_vectors if min_vectors > 0 else 0
    
    stats = {
        'min_length': min_length,
        'max_length': max_length,
        'min_avg_vectors': min_vectors,
        'max_avg_vectors': max_vectors,
        'multiplication_ratio': multiplication_ratio,
        'length_to_avg_vectors': avg_vectors_by_length
    }
    
    logger.info(f"Multiplication ratio: {multiplication_ratio:.2f}x "
                f"(min length {min_length}: {min_vectors:.2f} vectors, "
                f"max length {max_length}: {max_vectors:.2f} vectors)")
    
    return stats


def calculate_disparate_impact(encoded_df: pd.DataFrame,
                               protected_attribute: str,
                               positive_outcome: str = '1') -> Dict:
    """
    Compute Disparate Impact (DI).

    DI = P(Y=1 | A=protected) / P(Y=1 | A=non-protected)
    DI ≈ 1   → fair
    DI < 0.8 → protected group disadvantaged
    DI > 1.25 → protected group advantaged

    Args:
        encoded_df: Encoded DataFrame.
        protected_attribute: Protected attribute column name.
        positive_outcome: Positive outcome value (default: '1').

    Returns:
        Dictionary with DI statistics, or None if groups are empty.
    """
    logger.info(f"Calculating Disparate Impact for attribute: {protected_attribute}")
    
    if protected_attribute not in encoded_df.columns:
        logger.error(f"Protected attribute '{protected_attribute}' not found in DataFrame")
        return None
    
    if 'label' not in encoded_df.columns:
        logger.error("'label' column not found in DataFrame")
        return None
    
    # Split into protected and non-protected groups
    protected_group = encoded_df[encoded_df[protected_attribute] == 1]
    non_protected_group = encoded_df[encoded_df[protected_attribute] == 0]
    
    if len(protected_group) == 0 or len(non_protected_group) == 0:
        logger.warning("One of the groups is empty")
        return None
    
    # Positive outcome rates
    protected_positive = (protected_group['label'] == positive_outcome).sum()
    protected_total = len(protected_group)
    protected_rate = protected_positive / protected_total if protected_total > 0 else 0
    
    non_protected_positive = (non_protected_group['label'] == positive_outcome).sum()
    non_protected_total = len(non_protected_group)
    non_protected_rate = non_protected_positive / non_protected_total if non_protected_total > 0 else 0
    
    # Disparate Impact
    if non_protected_rate == 0:
        di = float('inf') if protected_rate > 0 else 0
    else:
        di = protected_rate / non_protected_rate
    
    # Bias status
    if di < 0.8:
        bias_status = 'protected_disadvantaged'
    elif di > 1.25:
        bias_status = 'protected_advantaged'
    else:
        bias_status = 'fair'
    
    stats = {
        'disparate_impact': di,
        'bias_status': bias_status,
        'protected_group': {
            'size': protected_total,
            'positive_count': protected_positive,
            'positive_rate': protected_rate
        },
        'non_protected_group': {
            'size': non_protected_total,
            'positive_count': non_protected_positive,
            'positive_rate': non_protected_rate
        }
    }
    
    logger.info(f"Disparate Impact: {di:.3f} ({bias_status})")
    
    return stats


def calculate_equalized_odds(encoded_df: pd.DataFrame,
                             protected_attribute: str,
                             predicted_label_col: str = 'predicted_label',
                             positive_outcome: str = '1') -> Dict:
    """
    Compute Equalized Odds (EO).

    EO focuses on equality of:
      - TPR = True Positive Rate
      - FPR = False Positive Rate

    Args:
        encoded_df: Encoded DataFrame (must contain actual and predicted labels).
        protected_attribute: Protected attribute column name.
        predicted_label_col: Predicted label column name.
        positive_outcome: Positive outcome value.

    Returns:
        Dictionary with EO statistics, or None if attributes/labels are missing.
    """
    logger.info(f"Calculating Equalized Odds for attribute: {protected_attribute}")
    
    if protected_attribute not in encoded_df.columns:
        logger.error(f"Protected attribute '{protected_attribute}' not found")
        return None
    
    if 'label' not in encoded_df.columns or predicted_label_col not in encoded_df.columns:
        logger.error("'label' or predicted label column not found")
        return None
    
    # Split into protected and non-protected groups
    protected_group = encoded_df[encoded_df[protected_attribute] == 1]
    non_protected_group = encoded_df[encoded_df[protected_attribute] == 0]
    
    def calculate_tpr_fpr(group):
        """Compute TPR and FPR for a group."""
        actual_positive = group[group['label'] == positive_outcome]
        actual_negative = group[group['label'] != positive_outcome]
        
        if len(actual_positive) == 0:
            tpr = 0
        else:
            true_positives = (actual_positive[predicted_label_col] == positive_outcome).sum()
            tpr = true_positives / len(actual_positive)
        
        if len(actual_negative) == 0:
            fpr = 0
        else:
            false_positives = (actual_negative[predicted_label_col] == positive_outcome).sum()
            fpr = false_positives / len(actual_negative)
        
        return tpr, fpr
    
    protected_tpr, protected_fpr = calculate_tpr_fpr(protected_group)
    non_protected_tpr, non_protected_fpr = calculate_tpr_fpr(non_protected_group)
    
    # Differences in TPR/FPR between protected and non-protected groups
    tpr_difference = abs(protected_tpr - non_protected_tpr)
    fpr_difference = abs(protected_fpr - non_protected_fpr)
    
    stats = {
        'protected_group': {
            'tpr': protected_tpr,
            'fpr': protected_fpr
        },
        'non_protected_group': {
            'tpr': non_protected_tpr,
            'fpr': non_protected_fpr
        },
        'tpr_difference': tpr_difference,
        'fpr_difference': fpr_difference,
        'equalized_odds_achieved': tpr_difference < 0.05 and fpr_difference < 0.05
    }
    
    logger.info(f"TPR difference: {tpr_difference:.3f}, FPR difference: {fpr_difference:.3f}")
    
    return stats


def calculate_sampling_bias_metrics(encoded_df: pd.DataFrame,
                                    original_log_trace_lengths: Dict[str, int]) -> Dict:
    """
    Compute sampling-bias metrics at the trace level (longer traces = more vectors).

    Args:
        encoded_df: Encoded DataFrame.
        original_log_trace_lengths: Mapping {trace_id: trace_length}.

    Returns:
        Dictionary with sampling-bias metrics.
    """
    logger.info("Calculating sampling bias metrics...")
    
    if 'trace_id' not in encoded_df.columns:
        logger.warning("trace_id column not found")
        return None
    
    # Multiplication ratio
    mult_ratio = calculate_multiplication_ratio(encoded_df, original_log_trace_lengths)
    
    # Distribution of feature vectors per trace
    vectors_per_trace = encoded_df.groupby('trace_id').size()
    
    stats = {
        'multiplication_ratio': mult_ratio,
        'vectors_per_trace': {
            'min': int(vectors_per_trace.min()),
            'max': int(vectors_per_trace.max()),
            'mean': float(vectors_per_trace.mean()),
            'std': float(vectors_per_trace.std()),
            'median': float(vectors_per_trace.median())
        },
        'total_traces': len(vectors_per_trace),
        'total_feature_vectors': len(encoded_df)
    }
    
    return stats


def print_bias_metrics_summary(metrics: Dict):
    """Print a human-readable summary of bias metrics."""
    print(f"\n{'='*60}")
    print(f"Bias Metrics Summary")
    print(f"{'='*60}")
    
    if 'multiplication_ratio' in metrics and metrics['multiplication_ratio']:
        mr = metrics['multiplication_ratio']
        print(f"\nMultiplication Ratio: {mr['multiplication_ratio']:.2f}x")
        print(f"  Min length ({mr['min_length']}): {mr['min_avg_vectors']:.2f} vectors")
        print(f"  Max length ({mr['max_length']}): {mr['max_avg_vectors']:.2f} vectors")
    
    if 'vectors_per_trace' in metrics:
        vpt = metrics['vectors_per_trace']
        print(f"\nFeature Vectors per Trace:")
        print(f"  Min: {vpt['min']}")
        print(f"  Max: {vpt['max']}")
        print(f"  Mean: {vpt['mean']:.2f}")
        print(f"  Median: {vpt['median']:.2f}")
        print(f"  Std: {vpt['std']:.2f}")
    
    print(f"\nTotal: {metrics.get('total_traces', 0):,} traces, "
          f"{metrics.get('total_feature_vectors', 0):,} feature vectors")
    print(f"{'='*60}\n")
