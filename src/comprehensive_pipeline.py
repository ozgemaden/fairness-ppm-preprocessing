"""
Comprehensive pipeline for exploring the impact of pre-processing design
decisions (prefix length, encoding type, prefix-generation strategy, and
mitigation) on data characteristics, bias metrics, and model performance.
"""

import os
import json
import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from nirdizati_light.log.common import get_log
from src.dataset_loader import load_and_convert_dataset
from nirdizati_light.encoding.common import EncodingType
from nirdizati_light.encoding.constants import TaskGenerationType
from nirdizati_light.labeling.common import LabelTypes
from pm4py.objects.log.obj import EventLog

from src.dataset_analysis import analyze_dataset, print_dataset_summary
from src.fixed_test_set import create_fixed_test_set, save_test_case_ids, get_case_ids
from src.encoding_pipeline import encode_log
from src.encoding_pipeline import save_encoded_data
from src.label_detection import (
    determine_labeling_strategy,
    print_label_strategy_info,
    check_label_availability
)
from src.outcome_distribution import (
    analyze_original_outcome_distribution,
    analyze_encoded_outcome_distribution,
    compare_outcome_distributions
)
from src.bias_metrics import (
    calculate_sampling_bias_metrics,
    calculate_disparate_impact
)
from src.preprocessing_techniques import apply_preprocessing_pipeline
from src.model_training import train_rf_dt_and_evaluate
from src.visualization import (
    plot_outcome_distribution,
    plot_outcome_distribution_change,
    plot_multiplication_ratio
)
from src.attribute_analysis import (
    identify_static_vs_dynamic_attributes,
    analyze_static_attribute_bias,
    analyze_dynamic_attribute_bias,
    analyze_attribute_vs_outcome
)


def _generation_type_display_label(generation_type: str) -> str:
    """
    Map internal generation type identifiers to human-readable labels for plots.
    (Implementation parameters remain unchanged: `all_in_one`, `only_this`.)
    """
    if generation_type == TaskGenerationType.ONLY_THIS.value:
        return "One vector per trace (longest prefix only)"
    if generation_type == TaskGenerationType.ALL_IN_ONE.value:
        return "Multiple vectors per trace (all prefix lengths)"
    return generation_type

logger = logging.getLogger(__name__)


class ComprehensivePipeline:
    """
    End-to-end pipeline that:
    - loads and analyses an event log,
    - defines a fixed test set,
    - applies different pre-processing configurations,
    - computes outcome and sampling-bias metrics,
    - optionally applies mitigation,
    - trains simple models (RF/DT) and aggregates results.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 output_dir: str = 'results',
                 seed: int = 42):
        """
        Args:
            dataset_path: Path to the dataset file.
            output_dir: Directory where results will be stored.
            seed: Random seed.
        """
        self.dataset_path = dataset_path
        self.dataset_name = Path(dataset_path).stem
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Create output sub-directories
        self._create_output_dirs()
        
        # Results storage
        self.results = []
        self.dataset_characteristics = None
        self.original_outcome_stats = None
        self.test_cases = None
        self.labeling_type = None
        self.label_strategy_info = None
        
    def _create_output_dirs(self):
        """Create all required output sub-directories."""
        dirs = [
            self.output_dir / 'dataset_characteristics',
            self.output_dir / 'outcome_distributions',
            self.output_dir / 'bias_metrics',
            self.output_dir / 'encoded_data',
            self.output_dir / 'visualizations',
            self.output_dir / 'preprocessing_results',
            self.output_dir / 'attribute_analysis'
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run_full_pipeline(self,
                          encoding_types: List[str] = None,
                          generation_types: List[str] = None,
                          apply_preprocessing: bool = False,
                          prefix_length_names: List[str] = None):
        """
        Run the full pipeline for the given configuration space.

        Args:
            encoding_types: Encoding types to test (None = use default set).
            generation_types: Prefix-generation strategies to test (None = use default set).
            apply_preprocessing: Whether to apply the mitigation pre-processing pipeline.
            prefix_length_names: Subset of prefix length keys to use (e.g. ['min', 'avg', 'max']);
                                 if None, all available prefix lengths are used.
        """
        logger.info("="*60)
        logger.info(f"Starting Comprehensive Pipeline for {self.dataset_name}")
        logger.info("="*60)
        
        # 1. Load and analyse the dataset
        # First try the standard Nirdizati-light format; fall back to custom loader if needed
        try:
            log = get_log(self.dataset_path, separator=';')
        except (ValueError, KeyError) as e:
            # If the log is not in the standard Nirdizati-light format, use the custom loader
            logger.info(f"Standard format failed, trying custom loader: {str(e)}")
            log = load_and_convert_dataset(self.dataset_path, separator=',')
        self.dataset_characteristics = analyze_dataset(log, self.dataset_name)
        print_dataset_summary(self.dataset_characteristics)
        
        # Save dataset characteristics
        self._save_dataset_characteristics()
        
        # 1.5. Determine labeling strategy
        self.labeling_type, self.label_strategy_info = determine_labeling_strategy(log)
        print_label_strategy_info(self.label_strategy_info)
        
        # 1.6. If fallback_to_last_activity=True, add last activity as a trace-level label
        if self.label_strategy_info.get('fallback_to_last_activity', False):
            log = self._add_last_activity_as_label(log)
            # Sanity check: ensure the label has been added correctly
            if len(log) > 0 and 'label' in log[0].attributes:
                sample_label = log[0].attributes['label']
                logger.info(f"Added last activity as 'label' attribute to traces (fallback). Sample label: {sample_label}")
            else:
                logger.error("Label was not added successfully! Encoding may fail.")
                raise ValueError("Failed to add label attribute to traces. Cannot proceed with encoding.")
        
        # 2. Create a fixed test set (all configurations evaluated on the same test cases)
        train_log, val_log, test_log, test_cases = create_fixed_test_set(
            log, shuffle=False, seed=self.seed
        )
        self.test_cases = test_cases
        self.train_cases = get_case_ids(train_log)
        self.full_log = log  # Full log: encoding is performed on all cases, then split by case_id

        # Save test case IDs
        test_case_path = self.output_dir / 'encoded_data' / f'{self.dataset_name}_test_cases.json'
        save_test_case_ids(test_cases, str(test_case_path))
        
        # 3. Analyse original outcome distribution on the training portion
        self.original_outcome_stats = analyze_original_outcome_distribution(train_log)
        if self.original_outcome_stats:
            self._save_outcome_stats('original', self.original_outcome_stats)
        
        # 4. Get prefix lengths
        prefix_lengths = {
            'min': self.dataset_characteristics['prefix_lengths']['min'],
            'avg': self.dataset_characteristics['prefix_lengths']['avg'],
            'max': self.dataset_characteristics['prefix_lengths']['max']
        }
        
        # 5. Encoding and generation types
        if encoding_types is None:
            encoding_types = [
                EncodingType.SIMPLE.value,
                EncodingType.FREQUENCY.value,
                EncodingType.COMPLEX.value
            ]
        
        if generation_types is None:
            generation_types = [
                TaskGenerationType.ONLY_THIS.value,
                TaskGenerationType.ALL_IN_ONE.value
            ]
        
        # 6. Evaluate all combinations
        if prefix_length_names is None:
            selected_prefixes = list(prefix_lengths.items())
        else:
            # Use only the requested prefix length keys if provided
            selected_prefixes = [
                (name, prefix_lengths[name])
                for name in prefix_length_names
                if name in prefix_lengths
            ]
        
        total_combinations = len(selected_prefixes) * len(encoding_types) * len(generation_types)
        current_combination = 0
        
        logger.info(f"\nTotal combinations to test: {total_combinations}")
        logger.info("="*60)
        
        for prefix_length_name, prefix_length in selected_prefixes:
            for encoding_type in encoding_types:
                for generation_type in generation_types:
                    current_combination += 1
                    logger.info(f"\n[{current_combination}/{total_combinations}] Testing: "
                              f"prefix={prefix_length_name}({prefix_length}), "
                              f"encoding={encoding_type}, generation={generation_type}")
                    logger.info(f"Progress: {current_combination}/{total_combinations} "
                              f"({100*current_combination/total_combinations:.1f}%)")
                    
                    try:
                        result = self._test_combination(
                            self.full_log,
                            train_log,
                            self.train_cases,
                            prefix_length,
                            prefix_length_name,
                            encoding_type,
                            generation_type,
                            apply_preprocessing
                        )
                        if result:
                            self.results.append(result)
                            logger.info(f"[OK] Completed: {current_combination}/{total_combinations}")
                    except Exception as e:
                        logger.error(f"[ERROR] Error in combination {prefix_length_name}/{encoding_type}/{generation_type}: {str(e)}")
        
        # 7. Save aggregated results
        self._save_results()
        
        logger.info("="*60)
        logger.info("Pipeline completed!")
        logger.info("="*60)
    
    def _test_combination(self,
                         full_log,
                         train_log,
                         train_cases: List[str],
                         prefix_length: int,
                         prefix_length_name: str,
                         encoding_type: str,
                         generation_type: str,
                         apply_preprocessing: bool) -> Dict:
        """
        Evaluate a single pre-processing and encoding combination.

        The full log is encoded once; train/test are then separated by case_id.
        Outcome and bias analyses are performed on the training portion, and model
        evaluation uses the fixed test cases.

        Returns:
            Dictionary with results.
        """
        logger.info(f"\nTesting: prefix={prefix_length_name}({prefix_length}), "
                   f"encoding={encoding_type}, generation={generation_type}")
        
        # Label presence check before encoding (for ATTRIBUTE_STRING)
        if self.labeling_type == LabelTypes.ATTRIBUTE_STRING.value:
            sample_trace = train_log[0] if len(train_log) > 0 else None
            if sample_trace and 'label' not in sample_trace.attributes:
                logger.error("Label attribute not found in train_log before encoding!")
                logger.error(f"Labeling type: {self.labeling_type}, but label attribute is missing.")
                raise ValueError(
                    f"Label attribute must be present for {self.labeling_type} labeling. "
                    "Check if _add_last_activity_as_label() was called correctly."
                )
            elif sample_trace:
                logger.debug(f"Label check passed. Sample label: {sample_trace.attributes.get('label', 'N/A')}")
        
        # Encode the full log so that all predictors are evaluated on the same test set
        encoder, encoded_df = encode_log(
            log=full_log,
            prefix_length=prefix_length,
            encoding_type=encoding_type,
            generation_type=generation_type,
            labeling_type=self.labeling_type
        )
        encoded_df = encoded_df.copy()
        if 'trace_id' in encoded_df.columns:
            encoded_df['trace_id'] = encoded_df['trace_id'].astype(str)
        
        # Apply pre-processing (optional)
        if apply_preprocessing:
            encoded_df, prep_stats = apply_preprocessing_pipeline(encoded_df)
            preprocessing_applied = True
        else:
            prep_stats = None
            preprocessing_applied = False
        
        # Restrict to the training portion for outcome/bias analysis
        train_cases_set = set(str(c) for c in train_cases)
        train_encoded = encoded_df[encoded_df['trace_id'].isin(train_cases_set)] if 'trace_id' in encoded_df.columns else encoded_df
        
        # Analyse encoded outcome distribution (training part; compared to original training log)
        encoded_stats = analyze_encoded_outcome_distribution(train_encoded)
        
        # If no label column is present, warn but still record basic stats
        if encoded_stats is None:
            logger.warning(f"Label column not found in encoded DataFrame. Columns: {list(encoded_df.columns)}")
            logger.warning("Continuing without outcome distribution analysis...")
            # Still save encoded DataFrame stats
            encoded_stats = {
                'total_feature_vectors': len(encoded_df),
                'outcome_counts': {},
                'outcome_distribution': {},
                'trace_vector_stats': None,
                'unique_outcomes': []
            }
        
        # Compare original vs encoded outcome distributions
        comparison = None
        if self.original_outcome_stats and encoded_stats:
            comparison = compare_outcome_distributions(
                self.original_outcome_stats, encoded_stats
            )
        
        # Sampling-bias metrics on the training portion (vectors per trace vs trace length)
        trace_lengths = {}
        for trace in train_log:
            case_id = str(trace.attributes.get('concept:name', ''))
            if case_id:
                trace_lengths[case_id] = len(trace)
        
        if not trace_lengths:
            logger.warning("No trace IDs found in train_log. Cannot calculate sampling bias metrics.")
            bias_metrics = None
        else:
            bias_metrics = calculate_sampling_bias_metrics(train_encoded, trace_lengths)
        
        # Attribute-level analysis (only for complex encoding)
        attribute_analysis = None
        if encoding_type == EncodingType.COMPLEX.value:
            logger.info("Performing attribute-level analysis (complex encoding)...")
            try:
                attr_classification = identify_static_vs_dynamic_attributes(train_encoded)
                
                if attr_classification is not None:
                    attribute_analysis = {
                        'static_attributes': attr_classification['static_attributes'],
                        'dynamic_attributes': attr_classification['dynamic_attributes'],
                        'total_attributes': attr_classification['total_attributes'],
                        'static_analyses': {},
                        'dynamic_analyses': {}
                    }
                    
                    # Bias analysis for a subset of static attributes
                    for static_attr in attr_classification['static_attributes'][:5]:
                        if 'label' in train_encoded.columns:
                            static_analysis = analyze_static_attribute_bias(train_encoded, static_attr)
                            if static_analysis:
                                attribute_analysis['static_analyses'][static_attr] = static_analysis
                    
                    for dynamic_attr in attr_classification['dynamic_attributes'][:5]:
                        if 'label' in train_encoded.columns:
                            dynamic_analysis = analyze_dynamic_attribute_bias(train_encoded, dynamic_attr)
                            if dynamic_analysis:
                                attribute_analysis['dynamic_analyses'][dynamic_attr] = dynamic_analysis
                    
                    # Persist attribute-analysis results
                    if attribute_analysis is not None:
                        attr_analysis_path = (
                            self.output_dir / 'attribute_analysis' / 
                            f"{self.dataset_name}_prefix{prefix_length_name}_{prefix_length}_"
                            f"{encoding_type}_{generation_type}_attribute_analysis.json"
                        )
                        attr_analysis_serializable = self._make_json_serializable(attribute_analysis)
                        with open(attr_analysis_path, 'w') as f:
                            json.dump(attr_analysis_serializable, f, indent=2)
                        logger.info(f"Attribute analysis saved to {attr_analysis_path}")
            except Exception as e:
                logger.warning(f"Attribute analysis failed: {str(e)}")
                attribute_analysis = None
        
        # Visualization: outcome distribution plots
        if self.original_outcome_stats and encoded_stats and comparison:
            try:
                # Outcome distribution comparison plot
                generation_label = _generation_type_display_label(generation_type)
                plot_title = (
                    f"{self.dataset_name} - Outcome Distribution\n"
                    f"Prefix: {prefix_length_name}({prefix_length}), "
                    f"Encoding: {encoding_type}, Prefix generation: {generation_label}"
                )
                plot_path = (
                    (self.output_dir / 'visualizations' /
                     f"{self.dataset_name}_prefix{prefix_length_name}_{prefix_length}_"
                     f"{encoding_type}_{generation_type}_outcome_distribution.png").resolve()
                )
                plot_outcome_distribution(
                    self.original_outcome_stats,
                    encoded_stats,
                    title=plot_title,
                    save_path=str(plot_path.resolve())
                )
                plt.close('all')  # Avoid memory leaks when running many combinations
                
                # Outcome distribution change plot
                change_plot_path = (
                    self.output_dir / 'visualizations' /
                    f"{self.dataset_name}_prefix{prefix_length_name}_{prefix_length}_"
                    f"{encoding_type}_{generation_type}_outcome_change.png"
                ).resolve()
                plot_outcome_distribution_change(
                    comparison,
                    title=f"{self.dataset_name} - Outcome Distribution Change",
                    save_path=str(change_plot_path)
                )
                plt.close('all')  # Avoid memory leaks when running many combinations
                
                logger.info(f"Visualizations saved to {self.output_dir / 'visualizations'}")
            except Exception as e:
                import traceback
                logger.warning(f"Visualization failed: {str(e)}")
                logger.debug(traceback.format_exc())
                # Best-effort attempt to create the directory even if plotting failed
                try:
                    (self.output_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
                except Exception as mkdir_e:
                    logger.warning(f"Visualizations dir create failed: {mkdir_e}")
        
        # Visualization: multiplication-ratio plot
        if bias_metrics and bias_metrics.get('multiplication_ratio'):
            try:
                mult_ratio_plot_path = (
                    self.output_dir / 'visualizations' /
                    f"{self.dataset_name}_prefix{prefix_length_name}_{prefix_length}_"
                    f"{encoding_type}_{generation_type}_multiplication_ratio.png"
                ).resolve()
                generation_label_mr = _generation_type_display_label(generation_type)
                plot_title = (
                    f"{self.dataset_name} - Multiplication Ratio\n"
                    f"Prefix: {prefix_length_name}({prefix_length}), "
                    f"Encoding: {encoding_type}, Prefix generation: {generation_label_mr}"
                )
                plot_multiplication_ratio(
                    bias_metrics['multiplication_ratio'],
                    title=plot_title,
                    save_path=str(mult_ratio_plot_path)
                )
                plt.close('all')  # Avoid memory leaks when running many combinations
            except Exception as e:
                import traceback
                logger.warning(f"Multiplication ratio plot failed: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Persist encoded data to CSV (for reproducibility)
        csv_filename = (
            f"{self.dataset_name}_"
            f"prefix{prefix_length_name}_{prefix_length}_"
            f"{encoding_type}_{generation_type}"
            f"{'_preprocessed' if preprocessing_applied else ''}.csv"
        )
        csv_path = self.output_dir / 'encoded_data' / csv_filename
        save_encoded_data(encoded_df, str(csv_path))  # Full encoded (train+test) for reproducibility
        
        # Model performance: evaluated on the same fixed test set (when available)
        model_metrics = None
        if self.test_cases and encoded_df is not None:
            model_metrics = train_rf_dt_and_evaluate(
                encoded_df, self.test_cases,
                label_col='label', trace_id_col='trace_id',
                max_features_importance=15, random_state=self.seed
            )
        
        # Aggregate results into a single dictionary
        result = {
            'dataset': self.dataset_name,
            'prefix_length': prefix_length,
            'prefix_length_name': prefix_length_name,
            'encoding_type': encoding_type,
            'generation_type': generation_type,
            'preprocessing_applied': preprocessing_applied,
            'encoded_stats': encoded_stats,
            'comparison': comparison,
            'bias_metrics': bias_metrics,
            'preprocessing_stats': prep_stats,
            'attribute_analysis': attribute_analysis,
            'model_metrics': model_metrics,
            'csv_path': str(csv_path)
        }
        
        # Save outcome stats
        if encoded_stats:
            stats_filename = (
                f"{self.dataset_name}_"
                f"prefix{prefix_length_name}_{prefix_length}_"
                f"{encoding_type}_{generation_type}_outcome_stats.json"
            )
            self._save_outcome_stats(stats_filename.replace('.json', ''), encoded_stats)
        
        return result
    
    def _add_last_activity_as_label(self, log: EventLog) -> EventLog:
        """
        Add the last activity of each trace as a trace-level 'label' attribute
        (fallback strategy when no explicit label is present).

        Args:
            log: PM4Py EventLog instance.

        Returns:
            EventLog with a 'label' attribute attached to each trace.
        """
        for trace in log:
            if len(trace) > 0:
                # Add last activity as label
                last_activity = trace[-1]['concept:name']
                trace.attributes['label'] = last_activity
        return log
    
    def _save_dataset_characteristics(self):
        """Persist dataset characteristics to JSON."""
        filepath = self.output_dir / 'dataset_characteristics' / f'{self.dataset_name}_characteristics.json'
        with open(filepath, 'w') as f:
            json.dump(self.dataset_characteristics, f, indent=2)
        logger.info(f"Dataset characteristics saved to {filepath}")
    
    def _save_outcome_stats(self, name: str, stats: Dict):
        """Persist outcome statistics to JSON."""
        filepath = self.output_dir / 'outcome_distributions' / f'{self.dataset_name}_{name}_outcome_stats.json'
        # Make JSON serializable
        stats_serializable = self._make_json_serializable(stats)
        with open(filepath, 'w') as f:
            json.dump(stats_serializable, f, indent=2)
    
    def _save_results(self):
        """Persist summary CSV and detailed JSON with all results."""
        # Results DataFrame
        results_data = []
        for result in self.results:
            mm = result.get('model_metrics')
            row = {
                'dataset': result['dataset'],
                'prefix_length': result['prefix_length'],
                'prefix_length_name': result['prefix_length_name'],
                'encoding_type': result['encoding_type'],
                'generation_type': result['generation_type'],
                'preprocessing_applied': result['preprocessing_applied'],
                'total_feature_vectors': result['encoded_stats']['total_feature_vectors'] if result['encoded_stats'] else 0,
                'multiplication_ratio': result['bias_metrics']['multiplication_ratio']['multiplication_ratio'] if result['bias_metrics'] and result['bias_metrics'].get('multiplication_ratio') else None,
                'f1_rf': mm.get('f1_rf') if mm else None,
                'f1_dt': mm.get('f1_dt') if mm else None,
                'accuracy_rf': mm.get('accuracy_rf') if mm else None,
                'accuracy_dt': mm.get('accuracy_dt') if mm else None,
            }
            
            # Outcome-distribution changes
            if result['comparison']:
                for outcome, change_info in result['comparison']['outcome_distribution_change'].items():
                    row[f'{outcome}_original'] = change_info['original']
                    row[f'{outcome}_encoded'] = change_info['encoded']
                    row[f'{outcome}_change'] = change_info['change']
            
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        results_path = self.output_dir / f'{self.dataset_name}_results_summary.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results summary saved to {results_path}")
        
        # Detailed results JSON
        results_json_path = self.output_dir / f'{self.dataset_name}_results_detailed.json'
        results_serializable = [self._make_json_serializable(r) for r in self.results]
        with open(results_json_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        logger.info(f"Detailed results saved to {results_json_path}")
    
    def _make_json_serializable(self, obj):
        """Convert nested objects to a JSON-serializable structure."""
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(k, (np.integer, np.int32, np.int64)):
                    key = int(k)
                elif isinstance(k, (np.floating, np.float32, np.float64)):
                    key = float(k)
                elif not isinstance(k, (str, int, float, bool, type(None))):
                    key = str(k)
                else:
                    key = k
                out[key] = self._make_json_serializable(v)
            return out
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='split')  # index, columns, data
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj


def run_pipeline_for_dataset(dataset_path: str,
                            output_dir: str = 'results',
                            seed: int = 42,
                            apply_preprocessing: bool = False):
    """
    Convenience wrapper to run the comprehensive pipeline for a single dataset.

    Args:
        dataset_path: Path to the dataset file.
        output_dir: Output directory.
        seed: Random seed.
        apply_preprocessing: Whether to apply mitigation pre-processing.
    """
    pipeline = ComprehensivePipeline(dataset_path, output_dir, seed)
    pipeline.run_full_pipeline(apply_preprocessing=apply_preprocessing)
    return pipeline
