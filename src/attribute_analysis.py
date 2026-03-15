"""
Attribute-level analysis utilities.

Used together with complex encoding to analyse how trace- and event-level
attributes relate to outcomes and how their distributions change after
pre-processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def analyze_attribute_distribution(encoded_df: pd.DataFrame, 
                                  attribute_name: str) -> Dict:
    """
    Analyse the marginal distribution of a single attribute.

    Args:
        encoded_df: Encoded DataFrame (typically with complex encoding).
        attribute_name: Name of the attribute column to analyse.

    Returns:
        Dictionary with attribute distribution statistics, or None if missing.
    """
    if attribute_name not in encoded_df.columns:
        logger.warning(f"Attribute '{attribute_name}' not found in DataFrame")
        return None
    
    attribute_counts = Counter(encoded_df[attribute_name])
    total = len(encoded_df)
    
    distribution = {
        value: (count / total) * 100 
        for value, count in attribute_counts.items()
    }
    
    return {
        'attribute_name': attribute_name,
        'total_count': total,
        'unique_values': len(attribute_counts),
        'value_counts': dict(attribute_counts),
        'distribution': distribution
    }


def analyze_attribute_vs_outcome(encoded_df: pd.DataFrame,
                                 attribute_name: str,
                                 outcome_col: str = 'label') -> Dict:
    """
    Analyse the relationship between a single attribute and the outcome.

    Typical examples: resource vs outcome, age vs outcome, diagnosis vs outcome.

    Args:
        encoded_df: Encoded DataFrame.
        attribute_name: Name of the attribute column to analyse.
        outcome_col: Outcome column name.

    Returns:
        Dictionary with attribute vs outcome statistics, or None if missing.
    """
    if attribute_name not in encoded_df.columns:
        logger.warning(f"Attribute '{attribute_name}' not found")
        return None
    
    if outcome_col not in encoded_df.columns:
        logger.warning(f"Outcome column '{outcome_col}' not found")
        return None
    
    # If the selection returns a DataFrame (duplicate column name), take the first column
    attr_series = encoded_df[attribute_name]
    if isinstance(attr_series, pd.DataFrame):
        attr_series = attr_series.iloc[:, 0]
    outcome_series = encoded_df[outcome_col]
    if isinstance(outcome_series, pd.DataFrame):
        outcome_series = outcome_series.iloc[:, 0]
    
    # Cross-tabulation
    crosstab = pd.crosstab(attr_series, outcome_series, margins=True)
    
    # For each attribute value, compute outcome distribution
    attribute_outcome_stats = {}
    for attr_value in attr_series.unique():
        attr_data = encoded_df[attr_series == attr_value]
        oc = attr_data[outcome_col]
        outcome_vals = oc.iloc[:, 0] if isinstance(oc, pd.DataFrame) else oc
        outcome_counts = Counter(outcome_vals)
        total = len(attr_data)
        
        attribute_outcome_stats[attr_value] = {
            'count': total,
            'outcome_counts': dict(outcome_counts),
            'outcome_distribution': {
                outcome: (count / total) * 100 
                for outcome, count in outcome_counts.items()
            }
        }
    
    return {
        'attribute_name': attribute_name,
        'crosstab': crosstab,
        'attribute_outcome_stats': attribute_outcome_stats
    }


def compare_attribute_distributions(original_log_attributes: Dict,
                                   encoded_df: pd.DataFrame,
                                   attribute_name: str) -> Dict:
    """
    Compare attribute distributions between original log and encoded data.

    Args:
        original_log_attributes: Distribution of the attribute in the original log.
        encoded_df: Encoded DataFrame.
        attribute_name: Name of the attribute to compare.

    Returns:
        Dictionary with comparison results, or None if the attribute is missing.
    """
    if attribute_name not in encoded_df.columns:
        return None
    
    encoded_stats = analyze_attribute_distribution(encoded_df, attribute_name)
    
    if not encoded_stats:
        return None
    
    comparison = {
        'attribute_name': attribute_name,
        'original_distribution': original_log_attributes.get('distribution', {}),
        'encoded_distribution': encoded_stats['distribution'],
        'distribution_change': {}
    }
    
    # Compute change in distribution
    all_values = set(
        list(comparison['original_distribution'].keys()) + 
        list(comparison['encoded_distribution'].keys())
    )
    
    for value in all_values:
        original_pct = comparison['original_distribution'].get(value, 0)
        encoded_pct = comparison['encoded_distribution'].get(value, 0)
        change = encoded_pct - original_pct
        
        comparison['distribution_change'][value] = {
            'original': original_pct,
            'encoded': encoded_pct,
            'change': change,
            'change_percentage': (change / original_pct * 100) if original_pct > 0 else 0
        }
    
    return comparison


def identify_static_vs_dynamic_attributes(encoded_df: pd.DataFrame,
                                          trace_id_col: str = 'trace_id') -> Dict:
    """
    Identify static vs dynamic attributes.

    Static: trace-level attributes (same value across a trace).
    Dynamic: event-level attributes (may vary within a trace).

    Args:
        encoded_df: Encoded DataFrame.
        trace_id_col: Trace ID column name.

    Returns:
        Dictionary with static and dynamic attribute lists.
    """
    if trace_id_col not in encoded_df.columns:
        logger.warning(f"Trace ID column '{trace_id_col}' not found")
        return None
    
    # Consider all columns except trace_id and label
    attribute_cols = [col for col in encoded_df.columns 
                     if col not in [trace_id_col, 'label']]
    
    static_attributes = []
    dynamic_attributes = []
    
    for col in attribute_cols:
        # For each trace, count how many distinct values this attribute takes
        unique_per_trace = encoded_df.groupby(trace_id_col)[col].nunique()
        
        # If each trace has exactly one unique value → static, otherwise dynamic
        if (unique_per_trace == 1).all():
            static_attributes.append(col)
        else:
            dynamic_attributes.append(col)
    
    return {
        'static_attributes': static_attributes,
        'dynamic_attributes': dynamic_attributes,
        'total_attributes': len(attribute_cols)
    }


def analyze_static_attribute_bias(encoded_df: pd.DataFrame,
                                  static_attribute: str,
                                  outcome_col: str = 'label') -> Dict:
    """
    Analyse bias for a static (trace-level) attribute vs outcome.

    Args:
        encoded_df: Encoded DataFrame.
        static_attribute: Name of the static attribute.
        outcome_col: Outcome column name.

    Returns:
        Dictionary with static attribute bias statistics.
    """
    return analyze_attribute_vs_outcome(encoded_df, static_attribute, outcome_col)


def analyze_dynamic_attribute_bias(encoded_df: pd.DataFrame,
                                   dynamic_attribute: str,
                                   outcome_col: str = 'label') -> Dict:
    """
    Analyse bias for a dynamic (event-level) attribute vs outcome.

    Args:
        encoded_df: Encoded DataFrame.
        dynamic_attribute: Name of the dynamic attribute.
        outcome_col: Outcome column name.

    Returns:
        Dictionary with dynamic attribute bias statistics.
    """
    return analyze_attribute_vs_outcome(encoded_df, dynamic_attribute, outcome_col)


def print_attribute_analysis_summary(analysis: Dict):
    """Print a human-readable summary of an attribute analysis result."""
    print(f"\n{'='*60}")
    print(f"Attribute Analysis: {analysis.get('attribute_name', 'Unknown')}")
    print(f"{'='*60}")
    
    if 'crosstab' in analysis:
        print(f"\nCross-tabulation:")
        print(analysis['crosstab'])
    
    if 'attribute_outcome_stats' in analysis:
        print(f"\nAttribute Value vs Outcome:")
        for attr_value, stats in analysis['attribute_outcome_stats'].items():
            print(f"\n  {attr_value}:")
            print(f"    Count: {stats['count']:,}")
            print(f"    Outcome Distribution:")
            for outcome, pct in stats['outcome_distribution'].items():
                count = stats['outcome_counts'].get(outcome, 0)
                print(f"      {outcome}: {count:,} ({pct:.2f}%)")
    
    print(f"{'='*60}\n")
