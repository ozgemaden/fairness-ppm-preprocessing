"""
Dataset characteristic-based comparison utilities.

Used to compare event logs with different structural characteristics and to
relate those characteristics to bias metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import json

from src.dataset_analysis import analyze_dataset, classify_dataset
from src.comprehensive_pipeline import ComprehensivePipeline

logger = logging.getLogger(__name__)


def compare_datasets_by_characteristics(dataset_paths: List[str], 
                                       output_dir: str = 'results/comparison') -> pd.DataFrame:
    """
    Compare datasets based on their structural characteristics.

    Args:
        dataset_paths: List of dataset file paths.
        output_dir: Directory where comparison artefacts will be stored.

    Returns:
        DataFrame with per-dataset characteristics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_characteristics = []
    all_classifications = []
    
    logger.info("="*60)
    logger.info("Dataset Characteristic Comparison")
    logger.info("="*60)
    
    # Compute characteristics for each dataset
    for dataset_path in dataset_paths:
        dataset_name = Path(dataset_path).stem
        logger.info(f"\nAnalyzing {dataset_name}...")
        
        try:
            from src.dataset_loader import load_and_convert_dataset
            log = load_and_convert_dataset(dataset_path, separator=',')
            
            characteristics = analyze_dataset(log, dataset_name)
            classifications = classify_dataset(characteristics)
            
            all_characteristics.append(characteristics)
            all_classifications.append({
                'dataset_name': dataset_name,
                **classifications
            })
            
        except Exception as e:
            logger.error(f"Error analyzing {dataset_name}: {str(e)}")
            continue
    
    if not all_characteristics:
        logger.error("No datasets were successfully analyzed!")
        return pd.DataFrame()
    
    # Build DataFrames with raw characteristics and coarse classifications
    df_characteristics = pd.DataFrame(all_characteristics)
    df_classifications = pd.DataFrame(all_classifications)
    
    # Comparison analysis
    comparison_results = {
        'high_variance_datasets': [],
        'low_variance_datasets': [],
        'homogeneous_datasets': [],
        'heterogeneous_datasets': [],
        'comparison_summary': {}
    }
    
    # Group by variance class (if available)
    if not df_classifications.empty and 'variance_class' in df_classifications.columns:
        high_variance = df_classifications[df_classifications['variance_class'] == 'high']
        low_variance = df_classifications[df_classifications['variance_class'] == 'low']
        
        comparison_results['high_variance_datasets'] = high_variance['dataset_name'].tolist() if not high_variance.empty else []
        comparison_results['low_variance_datasets'] = low_variance['dataset_name'].tolist() if not low_variance.empty else []
    
    # Group by homogeneity class (if available)
    if not df_classifications.empty and 'homogeneity_class' in df_classifications.columns:
        homogeneous = df_classifications[df_classifications['homogeneity_class'] == 'homogeneous']
        heterogeneous = df_classifications[df_classifications['homogeneity_class'] == 'heterogeneous']
        
        comparison_results['homogeneous_datasets'] = homogeneous['dataset_name'].tolist() if not homogeneous.empty else []
        comparison_results['heterogeneous_datasets'] = heterogeneous['dataset_name'].tolist() if not heterogeneous.empty else []
    
    # Summary statistics
    comparison_results['comparison_summary'] = {
        'total_datasets': len(df_characteristics),
        'avg_traces': float(df_characteristics['num_traces'].mean()),
        'avg_variants': float(df_characteristics['num_variants'].mean()),
        'avg_variance_ratio': float(df_characteristics['variance_ratio'].mean()),
        'avg_homogeneity_ratio': float(df_characteristics['homogeneity_ratio'].mean()),
        'high_variance_count': len(high_variance),
        'low_variance_count': len(low_variance),
        'homogeneous_count': len(homogeneous),
        'heterogeneous_count': len(heterogeneous)
    }
    
    # Persist artefacts
    df_characteristics.to_csv(output_path / 'dataset_characteristics_comparison.csv', index=False)
    df_classifications.to_csv(output_path / 'dataset_classifications.csv', index=False)
    
    with open(output_path / 'comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Print comparison summary
    print_comparison_summary(comparison_results, df_characteristics)
    
    return df_characteristics


def compare_bias_by_characteristics(results_dir: str = 'results') -> pd.DataFrame:
    """
    Compare bias metrics across datasets with different characteristics.

    Args:
        results_dir: Directory where per-dataset results are stored.

    Returns:
        DataFrame with bias comparison enriched with characteristic columns.
    """
    results_path = Path(results_dir)
    
    # Locate all summary CSV files
    summary_files = list(results_path.glob('*_results_summary.csv'))
    
    if not summary_files:
        logger.warning("No results summary files found")
        return pd.DataFrame()
    
    all_results = []
    
    for summary_file in summary_files:
        try:
            df = pd.read_csv(summary_file)
            dataset_name = summary_file.stem.replace('_results_summary', '')
            df['dataset'] = dataset_name
            all_results.append(df)
        except Exception as e:
            logger.error(f"Error reading {summary_file}: {str(e)}")
            continue
    
    if not all_results:
        return pd.DataFrame()
    
    # Concatenate all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Load per-dataset characteristic JSON files
    characteristics_files = list(results_path.glob('dataset_characteristics/*_characteristics.json'))
    characteristics_dict = {}
    
    for char_file in characteristics_files:
        try:
            with open(char_file, 'r') as f:
                char_data = json.load(f)
                dataset_name = char_file.stem.replace('_characteristics', '')
                characteristics_dict[dataset_name] = char_data
        except Exception as e:
            logger.error(f"Error reading {char_file}: {str(e)}")
            continue
    
    # Attach characteristics to the combined results
    for dataset_name, char_data in characteristics_dict.items():
        mask = combined_df['dataset'] == dataset_name
        combined_df.loc[mask, 'variance_ratio'] = char_data.get('variance_ratio', np.nan)
        combined_df.loc[mask, 'homogeneity_ratio'] = char_data.get('homogeneity_ratio', np.nan)
        combined_df.loc[mask, 'num_variants'] = char_data.get('num_variants', np.nan)
    
    # Characteristic-based analysis
    comparison_path = results_path / 'comparison'
    comparison_path.mkdir(parents=True, exist_ok=True)
    
    # Variance-based comparison
    if 'variance_ratio' in combined_df.columns:
        high_variance = combined_df[combined_df['variance_ratio'] > 0.5]
        low_variance = combined_df[combined_df['variance_ratio'] <= 0.2]
        
        if len(high_variance) > 0 and len(low_variance) > 0:
            variance_comparison = {
                'high_variance_avg_multiplication_ratio': float(high_variance['multiplication_ratio'].mean()),
                'low_variance_avg_multiplication_ratio': float(low_variance['multiplication_ratio'].mean()),
                'high_variance_avg_feature_vectors': float(high_variance['total_feature_vectors'].mean()),
                'low_variance_avg_feature_vectors': float(low_variance['total_feature_vectors'].mean())
            }
            
            with open(comparison_path / 'variance_bias_comparison.json', 'w') as f:
                json.dump(variance_comparison, f, indent=2)
    
    # Homogeneity-based comparison
    if 'homogeneity_ratio' in combined_df.columns:
        homogeneous = combined_df[combined_df['homogeneity_ratio'] < 0.3]
        heterogeneous = combined_df[combined_df['homogeneity_ratio'] > 0.7]
        
        if len(homogeneous) > 0 and len(heterogeneous) > 0:
            homogeneity_comparison = {
                'homogeneous_avg_multiplication_ratio': float(homogeneous['multiplication_ratio'].mean()),
                'heterogeneous_avg_multiplication_ratio': float(heterogeneous['multiplication_ratio'].mean()),
                'homogeneous_avg_feature_vectors': float(homogeneous['total_feature_vectors'].mean()),
                'heterogeneous_avg_feature_vectors': float(heterogeneous['total_feature_vectors'].mean())
            }
            
            with open(comparison_path / 'homogeneity_bias_comparison.json', 'w') as f:
                json.dump(homogeneity_comparison, f, indent=2)
    
    # Persist combined bias-by-characteristics table
    combined_df.to_csv(comparison_path / 'bias_by_characteristics.csv', index=False)
    
    return combined_df


def print_comparison_summary(comparison_results: Dict, df_characteristics: pd.DataFrame):
    """Print a human-readable summary of dataset-characteristic comparisons."""
    print(f"\n{'='*60}")
    print("DATASET CHARACTERISTIC COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    summary = comparison_results['comparison_summary']
    print(f"\nTotal Datasets: {summary['total_datasets']}")
    print(f"Average Traces: {summary['avg_traces']:,.0f}")
    print(f"Average Variants: {summary['avg_variants']:,.0f}")
    print(f"Average Variance Ratio: {summary['avg_variance_ratio']:.3f}")
    print(f"Average Homogeneity Ratio: {summary['avg_homogeneity_ratio']:.3f}")
    
    print(f"\nVariance Classification:")
    print(f"  High Variance: {summary['high_variance_count']} datasets")
    if comparison_results['high_variance_datasets']:
        print(f"    - {', '.join(comparison_results['high_variance_datasets'])}")
    print(f"  Low Variance: {summary['low_variance_count']} datasets")
    if comparison_results['low_variance_datasets']:
        print(f"    - {', '.join(comparison_results['low_variance_datasets'])}")
    
    print(f"\nHomogeneity Classification:")
    print(f"  Homogeneous: {summary['homogeneous_count']} datasets")
    if comparison_results['homogeneous_datasets']:
        print(f"    - {', '.join(comparison_results['homogeneous_datasets'])}")
    print(f"  Heterogeneous: {summary['heterogeneous_count']} datasets")
    if comparison_results['heterogeneous_datasets']:
        print(f"    - {', '.join(comparison_results['heterogeneous_datasets'])}")
    
    print(f"\n{'='*60}\n")
