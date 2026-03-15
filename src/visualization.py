"""
Visualization utilities for outcome distributions, bias metrics and model results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Matplotlib font configuration
plt.rcParams['font.family'] = 'DejaVu Sans'


def plot_outcome_distribution(
    original_stats: Dict,
    encoded_stats: Dict,
    title: str = "Outcome Distribution Comparison",
    save_path: Optional[str] = None,
):
    """
    Visualize outcome distributions for original event log vs encoded data.

    Args:
        original_stats: Outcome distribution statistics for the original event log.
        encoded_stats: Outcome distribution statistics for the encoded data.
        title: Plot title.
        save_path: Optional path to save the figure as PNG.
    """
    if not original_stats or not encoded_stats:
        logger.warning("Missing statistics for visualization")
        return
    
    original_dist = original_stats['outcome_distribution']
    original_counts = original_stats['outcome_counts']
    encoded_dist = encoded_stats['outcome_distribution']
    encoded_counts = encoded_stats['outcome_counts']
    n_outcomes = max(len(original_dist), len(encoded_dist))
    # If there are many outcomes, enlarge the figure to avoid overlapping x-labels
    fig_width = max(14, min(n_outcomes * 1.8, 28))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6))
    
    # Original distribution
    axes[0].bar(original_dist.keys(), original_dist.values(), color='steelblue')
    axes[0].set_title('Original Event Log', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Outcome', fontsize=12)
    axes[0].set_ylabel('Percentage (%)', fontsize=12)
    axes[0].set_ylim([0, 100])
    axes[0].tick_params(axis='x', rotation=45)
    plt.setp(axes[0].get_xticklabels(), ha='right', rotation_mode='anchor')
    
    # Also show counts
    for outcome, pct in original_dist.items():
        count = original_counts.get(outcome, 0)
        axes[0].text(outcome, pct + 1, f'{count:,}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontsize=10)
    
    # Encoded distribution
    axes[1].bar(encoded_dist.keys(), encoded_dist.values(), color='coral')
    axes[1].set_title('After Prefix Extraction', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Outcome', fontsize=12)
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].set_ylim([0, 100])
    axes[1].tick_params(axis='x', rotation=45)
    plt.setp(axes[1].get_xticklabels(), ha='right', rotation_mode='anchor')
    
    # Also show counts
    for outcome, pct in encoded_dist.items():
        count = encoded_counts.get(outcome, 0)
        axes[1].text(outcome, pct + 1, f'{count:,}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()  # Only show when no save_path is provided


def plot_outcome_distribution_change(
    comparison: Dict,
    title: str = "Outcome Distribution Change",
    save_path: Optional[str] = None,
):
    """
    Visualize changes in outcome distribution (original vs encoded).

    Args:
        comparison: Output dictionary from compare_outcome_distributions().
        title: Plot title.
        save_path: Optional path to save the figure as PNG.
    """
    if not comparison:
        logger.warning("No comparison data for visualization")
        return
    
    outcomes = list(comparison['outcome_distribution_change'].keys())
    original_pcts = [comparison['outcome_distribution_change'][o]['original'] for o in outcomes]
    encoded_pcts = [comparison['outcome_distribution_change'][o]['encoded'] for o in outcomes]
    changes = [comparison['outcome_distribution_change'][o]['change'] for o in outcomes]
    
    x = np.arange(len(outcomes))
    width = 0.35
    
    # If there are many outcomes, enlarge the figure to avoid overlapping x-labels
    n_outcomes = len(outcomes)
    fig_width = max(10, min(n_outcomes * 1.5, 32))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    bars1 = ax.bar(x - width/2, original_pcts, width, label='Original', color='steelblue')
    bars2 = ax.bar(x + width/2, encoded_pcts, width, label='Encoded', color='coral')
    
    # Show change values above bars
    for i, (orig, enc, change) in enumerate(zip(original_pcts, encoded_pcts, changes)):
        ax.text(i, max(orig, enc) + 2, f'{change:+.1f}%', 
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               color='red' if change > 0 else 'green')
    
    ax.set_xlabel('Outcome', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, rotation=45, ha='right', rotation_mode='anchor')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()  # Only show when no save_path is provided


def plot_multiplication_ratio(
    mult_ratio_stats: Dict,
    title: str = "Multiplication Ratio by Trace Length",
    save_path: Optional[str] = None,
):
    """
    Visualize the multiplication ratio as a function of trace length
    (trace-level representation bias).

    Longer traces produce more feature vectors; the higher the ratio,
    the more pronounced the sampling bias.

    Args:
        mult_ratio_stats: Output dictionary from calculate_multiplication_ratio().
        title: Plot title.
        save_path: Optional path to save the figure as PNG.
    """
    if not mult_ratio_stats:
        logger.warning("No multiplication ratio data for visualization")
        return
    
    lengths = sorted(mult_ratio_stats['length_to_avg_vectors'].keys())
    avg_vectors = [mult_ratio_stats['length_to_avg_vectors'][l] for l in lengths]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(lengths, avg_vectors, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Trace Length', fontsize=12)
    ax.set_ylabel('Average Feature Vectors per Trace', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Highlight min and max points
    min_length = mult_ratio_stats['min_length']
    max_length = mult_ratio_stats['max_length']
    min_vectors = mult_ratio_stats['min_avg_vectors']
    max_vectors = mult_ratio_stats['max_avg_vectors']
    
    ax.scatter([min_length, max_length], [min_vectors, max_vectors], 
              color='red', s=200, zorder=5, label='Min/Max')
    ax.annotate(f'Min: {min_length}\n{min_vectors:.1f} vectors', 
               xy=(min_length, min_vectors), xytext=(10, 10),
               textcoords='offset points', fontsize=10, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    ax.annotate(f'Max: {max_length}\n{max_vectors:.1f} vectors', 
               xy=(max_length, max_vectors), xytext=(10, -30),
               textcoords='offset points', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()  # Only show when no save_path is provided


def plot_heatmap_f1_scores(
    results_df: pd.DataFrame,
    title: str = "F1 Scores Heatmap",
    save_path: Optional[str] = None,
):
    """
    Visualize F1 scores as a heatmap for different prefix/encoding/generation settings.

    Args:
        results_df: DataFrame with columns: prefix_length, generation_type,
                    encoding_type, f1_score.
        title: Plot title.
        save_path: Optional path to save the figure as PNG.
    """
    if results_df.empty:
        logger.warning("No results data for heatmap")
        return
    
    # Map internal generation type identifiers to human-readable labels
    label_map = {
        "only_this": "Single-prefix (one vector per trace)",
        "all_in_one": "Multi-prefix (all cumulative prefixes)",
    }
    df = results_df.copy()
    if "generation_type" in df.columns:
        df["generation_type"] = df["generation_type"].astype(str).map(
            lambda x: label_map.get(x, x)
        )
    
    # Create pivot table for the heatmap
    pivot = df.pivot_table(
        values='f1_score',
        index='generation_type',
        columns=['prefix_length', 'encoding_type'],
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
               cbar_kws={'label': 'F1 Score'}, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Prefix Length × Encoding Type', fontsize=12)
    ax.set_ylabel('Prefix generation strategy', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()  # Only show when no save_path is provided
