"""
Encoding pipeline using Nirdizati-light.

Provides helpers to perform prefix extraction and feature encoding for
predictive process monitoring experiments.
"""

import logging
from typing import Tuple, Optional
from pm4py.objects.log.obj import EventLog
import pandas as pd

from nirdizati_light.log.common import get_log
from nirdizati_light.encoding.common import get_encoded_df, EncodingType
from nirdizati_light.encoding.constants import (
    TaskGenerationType, 
    PrefixLengthStrategy,
    EncodingTypeAttribute
)
from nirdizati_light.encoding.time_encoding import TimeEncodingType
from nirdizati_light.labeling.common import LabelTypes

logger = logging.getLogger(__name__)


def encode_log(
    log: EventLog,
    prefix_length: int,
    encoding_type: str = EncodingType.SIMPLE.value,
    generation_type: str = TaskGenerationType.ONLY_THIS.value,
    padding: bool = True,
    labeling_type: str = LabelTypes.ATTRIBUTE_STRING.value,
    attribute_encoding: str = EncodingTypeAttribute.LABEL.value,
    time_encoding: str = TimeEncodingType.NONE.value,
    prefix_length_strategy: str = PrefixLengthStrategy.FIXED.value,
    target_event: Optional[str] = None
) -> Tuple[object, pd.DataFrame]:
    """
    Encode an event log using Nirdizati-light.

    Args:
        log: PM4Py EventLog instance.
        prefix_length: Prefix length (e.g. min, avg, or max trace length).
        encoding_type: Encoding type (SIMPLE, FREQUENCY, COMPLEX, etc.).
        generation_type: Prefix-generation type (ONLY_THIS or ALL_IN_ONE).
        padding: Whether to use padding.
        labeling_type: Labeling type (ATTRIBUTE_STRING, NEXT_ACTIVITY, etc.).
        attribute_encoding: Attribute encoding type (LABEL, ONEHOT).
        time_encoding: Time encoding type (NONE, DATE, DURATION, etc.).
        prefix_length_strategy: Prefix-length strategy (FIXED, PERCENTAGE, etc.).
        target_event: Optional target event.

    Returns:
        Tuple of (encoder, encoded_dataframe).
    """
    logger.info(f"Encoding log with: prefix_length={prefix_length}, "
                f"encoding_type={encoding_type}, generation_type={generation_type}")
    
    try:
        encoder, encoded_df = get_encoded_df(
            log=log,
            feature_encoding_type=encoding_type,
            prefix_length=prefix_length,
            prefix_length_strategy=prefix_length_strategy,
            time_encoding_type=time_encoding,
            attribute_encoding=attribute_encoding,
            padding=padding,
            labeling_type=labeling_type,
            task_generation_type=generation_type,
            target_event=target_event
        )
        
        logger.info(f"Encoding complete: {len(encoded_df):,} feature vectors created")
        return encoder, encoded_df
        
    except KeyError as e:
        # Specific handling when the label attribute is missing
        if 'label' in str(e).lower() or labeling_type == LabelTypes.ATTRIBUTE_STRING.value:
            logger.error(
                f"Label attribute not found in dataset. "
                f"Current labeling_type: {labeling_type}. "
                f"Consider using LabelTypes.NEXT_ACTIVITY or adding label attributes to your dataset."
            )
            raise ValueError(
                f"Label attribute not found. Cannot use {labeling_type}. "
                f"Please check your dataset or use a different labeling_type."
            ) from e
        raise
    except Exception as e:
        logger.error(f"Encoding failed: {str(e)}")
        raise


def get_encoding_config(
    prefix_length: int,
    encoding_type: str = EncodingType.SIMPLE.value,
    generation_type: str = TaskGenerationType.ONLY_THIS.value,
    **kwargs
) -> dict:
    """
    Build a configuration dictionary for encoding settings.

    Returns:
        Dictionary with encoding configuration.
    """
    config = {
        'prefix_length': prefix_length,
        'encoding_type': encoding_type,
        'generation_type': generation_type,
        'padding': kwargs.get('padding', True),
        'labeling_type': kwargs.get('labeling_type', LabelTypes.ATTRIBUTE_STRING.value),
        'attribute_encoding': kwargs.get('attribute_encoding', EncodingTypeAttribute.LABEL.value),
        'time_encoding': kwargs.get('time_encoding', TimeEncodingType.NONE.value),
        'prefix_length_strategy': kwargs.get('prefix_length_strategy', PrefixLengthStrategy.FIXED.value),
        'target_event': kwargs.get('target_event', None)
    }
    return config


def generate_all_combinations(prefix_lengths: list, encoding_types: list, generation_types: list) -> list:
    """
    Generate all combinations of prefix lengths, encoding types and generation types.

    Args:
        prefix_lengths: List of prefix lengths (e.g. [min, avg, max]).
        encoding_types: List of encoding types (SIMPLE, FREQUENCY, COMPLEX).
        generation_types: List of generation types (ONLY_THIS, ALL_IN_ONE).

    Returns:
        List of dictionaries, one per combination.
    """
    combinations = []
    for prefix_length in prefix_lengths:
        for encoding_type in encoding_types:
            for generation_type in generation_types:
                combinations.append({
                    'prefix_length': prefix_length,
                    'encoding_type': encoding_type,
                    'generation_type': generation_type
                })
    return combinations


def save_encoded_data(encoded_df: pd.DataFrame, filepath: str):
    """Save an encoded DataFrame as CSV (for reproducibility)."""
    encoded_df.to_csv(filepath, index=False)
    logger.info(f"Encoded data saved to {filepath}")


def load_encoded_data(filepath: str) -> pd.DataFrame:
    """Load an encoded DataFrame from a CSV file."""
    encoded_df = pd.read_csv(filepath)
    logger.info(f"Encoded data loaded from {filepath}")
    return encoded_df
