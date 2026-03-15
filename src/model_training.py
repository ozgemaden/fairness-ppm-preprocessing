"""
Model training utilities (optional, secondary focus).
Provides Random Forest and Decision Tree training to support interpretation
of which features become important under different pre-processing settings.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

from nirdizati_light.predictive_model.common import ClassificationMethods
from nirdizati_light.predictive_model.predictive_model import PredictiveModel
from nirdizati_light.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from nirdizati_light.evaluation.common import evaluate_classifier

logger = logging.getLogger(__name__)


def train_and_evaluate_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: str = ClassificationMethods.RANDOM_FOREST.value,
    prefix_length: int = 10,
    max_evaluations: int = 10,
    target_metric: str = HyperoptTarget.F1.value,
) -> Dict:
    """
    Train a predictive model using Nirdizati-light's PredictiveModel abstraction
    and evaluate it on a held-out test set.

    Note: In this project, the primary focus is on data and bias analysis;
    this function is a secondary utility for additional model-level evaluation.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame (fixed test set).
        model_type: Model type (e.g., RANDOM_FOREST, LSTM).
        prefix_length: Prefix length used in the encoding.
        max_evaluations: Maximum number of hyperparameter evaluations.
        target_metric: Metric to optimize (e.g., F1, ACCURACY).

    Returns:
        Dictionary with model performance metrics.
    """
    logger.info(f"Training {model_type} model...")
    
    try:
        # Instantiate model
        predictive_models = [
            PredictiveModel(model_type, train_df, val_df, test_df, prefix_length=prefix_length)
        ]
        
        # Hyperparameter optimization
        best_candidates, best_model_idx, best_model_model, best_model_config = retrieve_best_model(
            predictive_models,
            max_evaluations=max_evaluations,
            target=target_metric
        )
        
        # Set best model
        best_model = predictive_models[best_model_idx]
        best_model.model = best_model_model
        best_model.config = best_model_config
        
        logger.info(f"Best model: {best_model.model_type}")
        
        # Predict on test set
        predicted, scores = best_model.predict(test=True)
        actual = test_df['label']
        
        # Evaluate
        results = evaluate_classifier(actual, predicted, scores)
        
        logger.info(f"Model evaluation complete: F1={results.get('f1_score', 0):.4f}")
        
        return {
            'model_type': model_type,
            'prefix_length': prefix_length,
            'performance': results,
            'best_config': best_model_config
        }
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return None


def train_rf_dt_and_evaluate(
    encoded_df: pd.DataFrame,
    test_case_ids: Optional[List[str]] = None,
    label_col: str = 'label',
    trace_id_col: str = 'trace_id',
    max_features_importance: int = 15,
    random_state: int = 42,
    test_ratio: float = 0.2,
) -> Optional[Dict]:
    """
    Train Random Forest and Decision Tree classifiers and return F1/accuracy
    together with simple feature-importance summaries (to support interpretation).

    If `test_case_ids` (fixed test set case IDs) are present in `encoded_df`,
    the train/test split is based on these IDs. Otherwise, a random split by
    `trace_id` is used with the given `test_ratio`.

    Args:
        encoded_df: Encoded DataFrame (must contain `trace_id`, `label` and feature columns).
        test_case_ids: Optional list of fixed test-set case IDs.
        label_col: Name of the label column.
        trace_id_col: Name of the trace identifier column.
        max_features_importance: Number of top features to keep in the importance lists.
        random_state: Random seed.
        test_ratio: Fraction of traces used for test if `test_case_ids` is not provided.

    Returns:
        Dict with f1_rf, f1_dt, accuracy_rf, accuracy_dt, feature_importance_rf,
        feature_importance_dt, top_features, or None if evaluation is not possible.
    """
    if trace_id_col not in encoded_df.columns or label_col not in encoded_df.columns:
        logger.warning("trace_id or label column missing for model evaluation")
        return None
    
    feature_cols = [c for c in encoded_df.columns if c not in (trace_id_col, label_col)]
    if not feature_cols:
        logger.warning("No feature columns for model evaluation")
        return None
    
    test_ids_set = set(str(c) for c in (test_case_ids or []))
    in_df = set(encoded_df[trace_id_col].astype(str).unique())
    use_fixed_test = test_ids_set and (test_ids_set & in_df)
    
    if use_fixed_test:
        train_mask = ~encoded_df[trace_id_col].astype(str).isin(test_ids_set)
        test_mask = encoded_df[trace_id_col].astype(str).isin(test_ids_set)
    else:
        rng = np.random.default_rng(random_state)
        unique_traces = encoded_df[trace_id_col].astype(str).unique()
        n_test = max(1, int(len(unique_traces) * test_ratio))
        test_traces = set(rng.choice(unique_traces, size=n_test, replace=False))
        train_mask = ~encoded_df[trace_id_col].astype(str).isin(test_traces)
        test_mask = encoded_df[trace_id_col].astype(str).isin(test_traces)
    
    train_df = encoded_df.loc[train_mask]
    test_df = encoded_df.loc[test_mask]
    
    if len(train_df) < 2 or len(test_df) < 1:
        logger.warning("Insufficient train/test size for model evaluation")
        return None
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[label_col]
    X_test = test_df[feature_cols].copy()
    y_test = test_df[label_col]
    
    # Drop non-numeric feature columns (simple baseline: use only numeric features)
    for col in feature_cols:
        if col not in X_train.columns:
            continue
        if not np.issubdtype(X_train[col].dtype, np.number):
            X_train = X_train.drop(columns=[col], errors='ignore')
            X_test = X_test.drop(columns=[col], errors='ignore')
    if X_train.shape[1] == 0:
        logger.warning("No numeric features for model evaluation")
        return None
    
    feature_cols = list(X_train.columns)
    
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        f1_rf = f1_score(y_test, pred_rf, average='weighted', zero_division=0)
        acc_rf = accuracy_score(y_test, pred_rf)
        
        dt = DecisionTreeClassifier(random_state=random_state)
        dt.fit(X_train, y_train)
        pred_dt = dt.predict(X_test)
        f1_dt = f1_score(y_test, pred_dt, average='weighted', zero_division=0)
        acc_dt = accuracy_score(y_test, pred_dt)
        
        imp_rf = list(zip(feature_cols, rf.feature_importances_.tolist()))
        imp_rf.sort(key=lambda x: -x[1])
        imp_dt = list(zip(feature_cols, dt.feature_importances_.tolist()))
        imp_dt.sort(key=lambda x: -x[1])
        
        return {
            'f1_rf': float(f1_rf),
            'f1_dt': float(f1_dt),
            'accuracy_rf': float(acc_rf),
            'accuracy_dt': float(acc_dt),
            'feature_importance_rf': imp_rf[:max_features_importance],
            'feature_importance_dt': imp_dt[:max_features_importance],
            'top_features_rf': [x[0] for x in imp_rf[:5]],
            'top_features_dt': [x[0] for x in imp_dt[:5]],
        }
    except Exception as e:
        logger.warning(f"RF/DT evaluation failed: {e}")
        return None
