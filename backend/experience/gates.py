"""
Gating functions for the Experiential Layer.
Ensures experiential updates stay within Nature and Nurture bounds.
"""
from typing import Any, Optional, Dict
import numpy as np

from .config import DEFAULT_EXPERIENTIAL_CONFIG


def nature_gate(
    update: Any,
    update_type: str,
    value_keywords: Optional[list] = None,
) -> Optional[Any]:
    """
    Gate experiential updates through nature's value evaluation.
    
    Nature gate enforces hard constraints based on core values.
    All updates must pass through this gate.
    
    Args:
        update: The proposed update (vector, fact, commitment, etc.)
        update_type: Type of update ('topic_vector', 'salient_fact', 'commitment', etc.)
        value_keywords: Optional list of value-violating keywords to check
    
    Returns:
        Gated update (may be None if rejected, scaled, or unchanged)
    """
    if update is None:
        return None
    
    if update_type == 'topic_vector':
        # Topics are observational; minimal gating
        return update
    
    elif update_type == 'emotional_trajectory':
        # Emotions are observational; minimal gating
        return update
    
    elif update_type == 'user_state':
        # User state is observational; minimal gating
        return update
    
    elif update_type == 'salient_fact':
        # Check if fact content aligns with values
        if hasattr(update, 'content'):
            content = update.content.lower()
            # Reject facts that encode harmful instructions
            harmful_patterns = [
                'ignore safety', 'bypass', 'jailbreak', 'pretend you have no',
                'ignore your instructions', 'ignore instructions', 'disregard',
                'no restrictions', 'without restrictions', 'bypass safety',
                'override', 'hack', 'exploit'
            ]
            for pattern in harmful_patterns:
                if pattern in content:
                    return None  # Reject this fact
        return update
    
    elif update_type == 'commitment':
        # Ensure commitment doesn't violate core values
        if hasattr(update, 'promise'):
            promise = update.promise.lower()
            # Cannot commit to harmful actions
            harmful_commitments = [
                'harm', 'illegal', 'unethical', 'bypass safety', 'bypass',
                'jailbreak', 'ignore', 'override', 'hack', 'exploit',
                'no restrictions', 'without restrictions'
            ]
            for harmful in harmful_commitments:
                if harmful in promise:
                    return None  # Cannot commit to this
        return update
    
    elif update_type == 'persistent_pattern':
        # Cross-session patterns get scrutiny
        if isinstance(update, np.ndarray):
            # Normalize to prevent unbounded growth
            norm = np.linalg.norm(update)
            if norm > 1.0:
                update = update / norm
        return update
    
    return update


def nurture_gate(
    update: Any,
    update_type: str,
    nurture_state: Any,
    config: Any = None,
) -> Any:
    """
    Gate experiential updates through nurture's character bounds.
    
    Nurture gate ensures updates stay within established character.
    Updates are dampened or biased based on nurture state.
    
    Args:
        update: The proposed update
        update_type: Type of update
        nurture_state: Current NurtureState
        config: ExperientialConfig (optional)
    
    Returns:
        Gated update (may be modified but not rejected)
    """
    if update is None:
        return None
    
    if config is None:
        config = DEFAULT_EXPERIENTIAL_CONFIG
    
    stance_influence = config.STANCE_INFLUENCE
    
    if update_type == 'topic_vector':
        # Topics outside nurture's domain focus get dampened
        if isinstance(update, np.ndarray) and nurture_state is not None:
            # Simple dampening based on domain alignment
            # In full implementation, would compute semantic alignment
            dampening = 0.5 + 0.5 * 1.0  # Default full alignment
            return update * dampening
        return update
    
    elif update_type == 'emotional_trajectory':
        # Bound emotional range by nurture's emotionality stance
        if isinstance(update, np.ndarray) and nurture_state is not None:
            max_magnitude = nurture_state.get_emotionality_bound()
            current_magnitude = np.linalg.norm(update)
            if current_magnitude > max_magnitude:
                update = (update / current_magnitude) * max_magnitude
        return update
    
    elif update_type == 'user_state':
        # User modeling bounded by relationship depth
        if isinstance(update, np.ndarray) and nurture_state is not None:
            depth_factor = nurture_state.get_relationship_depth_factor()
            return update * depth_factor
        return update
    
    elif update_type == 'salient_fact':
        # Facts are dampened based on domain alignment
        if hasattr(update, 'salience_score') and nurture_state is not None:
            alignment = nurture_state.get_domain_alignment(update.content)
            update.salience_score *= alignment
        return update
    
    return update


def apply_experiential_gates(
    update: Any,
    update_type: str,
    nurture_state: Any = None,
    config: Any = None,
) -> Optional[Any]:
    """
    Apply both nature and nurture gates to an experiential update.
    
    Nature gate is applied first (hard constraints).
    Nurture gate is applied second (character constraints).
    
    Args:
        update: The proposed update
        update_type: Type of update
        nurture_state: Current NurtureState (optional)
        config: ExperientialConfig (optional)
    
    Returns:
        Fully gated update (may be None if rejected by nature gate)
    """
    # Nature gate first (hard constraints)
    gated = nature_gate(update, update_type)
    
    if gated is None:
        return None
    
    # Nurture gate second (character constraints)
    if nurture_state is not None:
        gated = nurture_gate(gated, update_type, nurture_state, config)
    
    return gated


def compute_promotion_candidate(
    persistent_traces: Any,
    nurture_state: Any,
    min_sessions: int = 10,
    stability_threshold: float = 0.9,
) -> Optional[Dict[str, Any]]:
    """
    Check if any persistent pattern should be promoted to nurture.
    
    Promotion is rare and requires:
    - Sufficient sessions
    - High pattern stability
    - Nurture still plastic enough to accept
    
    Args:
        persistent_traces: PersistentTraces from experiential state
        nurture_state: Current NurtureState
        min_sessions: Minimum sessions required
        stability_threshold: Required pattern stability
    
    Returns:
        Promotion candidate dict if eligible, None otherwise
    """
    if nurture_state is None:
        return None
    
    if not nurture_state.can_accept_promotion():
        return None  # Nurture too stable
    
    if persistent_traces.session_count < min_sessions:
        return None  # Not enough sessions
    
    # Compute pattern stability (simplified)
    pattern = persistent_traces.pattern_accumulator
    if isinstance(pattern, np.ndarray):
        # Stability approximated by pattern magnitude
        # In full implementation, would track variance over time
        stability = min(1.0, np.linalg.norm(pattern) / 2.0)
        
        if stability >= stability_threshold:
            return {
                'pattern': pattern,
                'stability': stability,
                'sessions': persistent_traces.session_count,
                'familiarity': persistent_traces.familiarity_score,
            }
    
    return None
