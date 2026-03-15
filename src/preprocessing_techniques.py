"""
Pre-processing techniques for encoded datasets.

Implements duplicate and conflicting-duplicate removal as described in
prior fairness-aware predictive process monitoring work.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def remove_duplicates(df: pd.DataFrame, keep: str = 'first') -> Tuple[pd.DataFrame, Dict]:
    """
    Remove exact duplicate feature vectors.

    Args:
        df: Encoded DataFrame.
        keep: 'first' (keep first occurrence) or 'last' (keep last occurrence).

    Returns:
        Tuple of (cleaned_df, removal_stats).
    """
    logger.info("Removing exact duplicates...")
    
    original_count = len(df)
    
    # Exclude trace_id and label when checking for duplicates (compare only features)
    feature_cols = [col for col in df.columns if col not in ['trace_id', 'label']]
    
    # Identify and drop duplicate feature vectors
    cleaned_df = df.drop_duplicates(subset=feature_cols, keep=keep)
    
    removed_count = original_count - len(cleaned_df)
    
    stats = {
        'original_count': original_count,
        'cleaned_count': len(cleaned_df),
        'removed_count': removed_count,
        'removal_percentage': (removed_count / original_count * 100) if original_count > 0 else 0
    }
    
    logger.info(f"Removed {removed_count:,} duplicate rows ({stats['removal_percentage']:.2f}%)")
    
    return cleaned_df, stats


def remove_conflicting(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove conflicting duplicates (identical features, different labels).

    Args:
        df: Encoded DataFrame.

    Returns:
        Tuple of (cleaned_df, removal_stats).
    """
    logger.info("Removing conflicting duplicates...")
    
    original_count = len(df)
    
    # Exclude trace_id and label when checking for duplicates (compare only features)
    feature_cols = [col for col in df.columns if col not in ['trace_id', 'label']]
    
    # Find rows that share the same features but differ in label
    duplicates = df[df.duplicated(subset=feature_cols, keep=False)]
    
    if len(duplicates) == 0:
        logger.info("No conflicting duplicates found")
        return df.copy(), {
            'original_count': original_count,
            'cleaned_count': original_count,
            'removed_count': 0,
            'removal_percentage': 0
        }
    
    # For each unique feature combination, check whether labels disagree
    conflicting_indices = []
    
    for feature_tuple, group in duplicates.groupby(feature_cols):
        unique_labels = group['label'].unique()
        if len(unique_labels) > 1:
            # Conflicting: same features, different labels
            conflicting_indices.extend(group.index.tolist())
    
    # Remove conflicting rows
    cleaned_df = df.drop(index=conflicting_indices)
    
    removed_count = len(conflicting_indices)
    
    stats = {
        'original_count': original_count,
        'cleaned_count': len(cleaned_df),
        'removed_count': removed_count,
        'removal_percentage': (removed_count / original_count * 100) if original_count > 0 else 0,
        'conflicting_groups': len(set([tuple(row) for row in duplicates[feature_cols].values]))
    }
    
    logger.info(f"Removed {removed_count:,} conflicting rows ({stats['removal_percentage']:.2f}%)")
    
    return cleaned_df, stats


def apply_preprocessing_pipeline(df: pd.DataFrame, 
                                 remove_dups: bool = True,
                                 remove_conflicts: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply a pre-processing pipeline (duplicate/conflicting removal) to an encoded dataset.

    Args:
        df: Encoded DataFrame.
        remove_dups: Whether to apply duplicate removal.
        remove_conflicts: Whether to apply conflicting-duplicate removal.

    Returns:
        Tuple of (cleaned_df, all_stats).
    """
    logger.info("Applying preprocessing pipeline...")
    
    all_stats = {
        'original_count': len(df),
        'steps': []
    }
    
    current_df = df.copy()
    
    # Step 1: Duplicate removal
    if remove_dups:
        current_df, dup_stats = remove_duplicates(current_df)
        all_stats['steps'].append({
            'step': 'duplicate_removal',
            'stats': dup_stats
        })
        all_stats['after_duplicate_removal'] = len(current_df)
    
    # Step 2: Conflicting removal
    if remove_conflicts:
        current_df, conflict_stats = remove_conflicting(current_df)
        all_stats['steps'].append({
            'step': 'conflicting_removal',
            'stats': conflict_stats
        })
        all_stats['after_conflicting_removal'] = len(current_df)
    
    all_stats['final_count'] = len(current_df)
    all_stats['total_removed'] = all_stats['original_count'] - all_stats['final_count']
    all_stats['total_removal_percentage'] = (
        all_stats['total_removed'] / all_stats['original_count'] * 100 
        if all_stats['original_count'] > 0 else 0
    )
    
    logger.info(f"Preprocessing complete: {all_stats['original_count']:,} → {all_stats['final_count']:,} "
                f"({all_stats['total_removal_percentage']:.2f}% removed)")
    
    return current_df, all_stats


def print_preprocessing_stats(stats: Dict):
    """Print a human-readable summary of pre-processing statistics."""
    print(f"\n{'='*60}")
    print(f"Pre-processing Statistics")
    print(f"{'='*60}")
    print(f"Original count: {stats['original_count']:,}")
    
    for step_info in stats['steps']:
        step = step_info['step']
        step_stats = step_info['stats']
        print(f"\n{step.replace('_', ' ').title()}:")
        print(f"  Removed: {step_stats['removed_count']:,} ({step_stats['removal_percentage']:.2f}%)")
        print(f"  Remaining: {step_stats['cleaned_count']:,}")
    
    print(f"\nFinal count: {stats['final_count']:,}")
    print(f"Total removed: {stats['total_removed']:,} ({stats['total_removal_percentage']:.2f}%)")
    print(f"{'='*60}\n")
