"""
Label detection and labeling-strategy selection utilities.

Inspects an event log to determine whether explicit labels are present and
selects an appropriate labeling strategy for Nirdizati-light.
"""

import logging
from typing import Dict, Optional, Tuple
from pm4py.objects.log.obj import EventLog, Trace
from nirdizati_light.labeling.common import LabelTypes

logger = logging.getLogger(__name__)


def check_label_availability(log: EventLog) -> Dict:
    """
    Check whether a label attribute is present in the event log.

    Args:
        log: PM4Py EventLog instance.

    Returns:
        Dictionary with label availability information.
    """
    has_trace_label = False
    has_event_label = False
    label_values = set()
    sample_traces_checked = min(100, len(log))  # Check up to the first 100 traces
    
    for i, trace in enumerate(log[:sample_traces_checked]):
        # Check for labels in trace attributes
        if 'label' in trace.attributes:
            has_trace_label = True
            label_values.add(str(trace.attributes['label']))
        
        # Check for labels in event attributes
        if len(trace) > 0:
            if 'label' in trace[0]:
                has_event_label = True
                label_values.add(str(trace[0]['label']))
    
    # If labels are present, collect all unique label values across the log
    if has_trace_label or has_event_label:
        label_values = set()
        for trace in log:
            if 'label' in trace.attributes:
                label_values.add(str(trace.attributes['label']))
            elif len(trace) > 0 and 'label' in trace[0]:
                label_values.add(str(trace[0]['label']))
    
    return {
        'has_label': has_trace_label or has_event_label,
        'has_trace_label': has_trace_label,
        'has_event_label': has_event_label,
        'unique_label_values': sorted(list(label_values)) if label_values else [],
        'num_unique_labels': len(label_values) if label_values else 0
    }


def determine_labeling_strategy(log: EventLog, 
                                preferred_strategy: Optional[str] = None) -> Tuple[str, Dict]:
    """
    Determine an appropriate labeling strategy for a given event log.

    Args:
        log: PM4Py EventLog instance.
        preferred_strategy: Optional user-specified strategy (None = auto-detect).

    Returns:
        Tuple of (labeling_type, strategy_info), where strategy_info contains
        label availability and the reasoning behind the recommendation.
    """
    label_info = check_label_availability(log)
    
    strategy_info = {
        'label_available': label_info['has_label'],
        'label_info': label_info,
        'recommended_strategy': None,
        'strategy_reason': None
    }
    
    # If the user specified a preferred strategy and labels are available (or NEXT_ACTIVITY is requested), use it
    if preferred_strategy:
        if label_info['has_label'] or preferred_strategy == LabelTypes.NEXT_ACTIVITY.value:
            strategy_info['recommended_strategy'] = preferred_strategy
            strategy_info['strategy_reason'] = f"User specified: {preferred_strategy}"
            return preferred_strategy, strategy_info
        else:
            logger.warning(f"Preferred strategy {preferred_strategy} requires label, but label not found. Using fallback.")
    
    # Automatic strategy selection
    if label_info['has_label']:
        # If labels exist, use ATTRIBUTE_STRING
        strategy_info['recommended_strategy'] = LabelTypes.ATTRIBUTE_STRING.value
        strategy_info['strategy_reason'] = (
            f"Label found in {'trace attributes' if label_info['has_trace_label'] else 'event attributes'}. "
            f"Found {label_info['num_unique_labels']} unique label values: {label_info['unique_label_values'][:5]}"
        )
        return LabelTypes.ATTRIBUTE_STRING.value, strategy_info
    
    else:
        # No explicit label found – consider using the last activity as outcome
        last_activities = set()
        for trace in log[:min(1000, len(log))]:  # Inspect up to the first 1000 traces
            if len(trace) > 0:
                last_activities.add(trace[-1]['concept:name'])
        
        num_unique_last_activities = len(last_activities)
        
        if num_unique_last_activities <= 10:
            # Use last activity as outcome (fallback) – limited number of distinct outcomes
            strategy_info['recommended_strategy'] = LabelTypes.ATTRIBUTE_STRING.value
            strategy_info['strategy_reason'] = (
                f"No label attribute found. Using last activity as outcome (fallback). "
                f"Found {num_unique_last_activities} unique last activities: {sorted(list(last_activities))[:5]}. "
                f"Will add last activity as 'label' attribute before encoding."
            )
            strategy_info['fallback_to_last_activity'] = True
            logger.info(
                "No label attribute found in dataset. "
                "Will use last activity as outcome (fallback). "
                "Adding last activity as 'label' attribute to traces before encoding."
            )
            return LabelTypes.ATTRIBUTE_STRING.value, strategy_info
        else:
            # Even if there are many distinct last activities, fall back to using
            # the last activity as outcome and add it as a 'label' attribute.
            strategy_info['recommended_strategy'] = LabelTypes.ATTRIBUTE_STRING.value
            strategy_info['strategy_reason'] = (
                f"No label attribute found. Using last activity as outcome (fallback) for outcome prediction. "
                f"Found {num_unique_last_activities} unique last activities (many, but will use as outcome). "
                f"Will add last activity as 'label' attribute before encoding."
            )
            strategy_info['fallback_to_last_activity'] = True
            logger.info(
                "No label attribute found in dataset. "
                "Will use last activity as outcome (fallback) for outcome prediction. "
                "Adding last activity as 'label' attribute to traces before encoding."
            )
            return LabelTypes.ATTRIBUTE_STRING.value, strategy_info


def print_label_strategy_info(strategy_info: Dict):
    """Print a human-readable summary of the chosen label strategy."""
    # Safe printing helper for Windows terminals
    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            safe_text = str(text).encode('ascii', 'replace').decode('ascii')
            print(safe_text)
    
    safe_print(f"\n{'='*60}")
    safe_print("Label Strategy Information")
    safe_print(f"{'='*60}")
    
    label_info = strategy_info['label_info']
    safe_print(f"\nLabel Availability:")
    safe_print(f"  Has Label: {label_info['has_label']}")
    if label_info['has_label']:
        safe_print(f"  Location: {'Trace attributes' if label_info['has_trace_label'] else 'Event attributes'}")
        safe_print(f"  Unique Labels: {label_info['num_unique_labels']}")
        if label_info['unique_label_values']:
            safe_print(f"  Label Values: {label_info['unique_label_values'][:10]}")
            if len(label_info['unique_label_values']) > 10:
                safe_print(f"    ... and {len(label_info['unique_label_values']) - 10} more")
    
    safe_print(f"\nRecommended Strategy:")
    safe_print(f"  Labeling Type: {strategy_info['recommended_strategy']}")
    safe_print(f"  Reason: {strategy_info['strategy_reason']}")
    
    if strategy_info.get('fallback_to_last_activity', False):
        safe_print(f"\nWARNING: Using last activity as outcome fallback.")
        safe_print(f"   Encoding may fail if nirdizati-light expects label attribute.")
        safe_print(f"   Consider adding label attributes to your dataset.")
    
    safe_print(f"{'='*60}\n")
