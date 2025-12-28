"""
Nurture update mechanisms for N_env, N_stance, stability, and plasticity.
"""
import numpy as np
from typing import Dict, Any, Tuple, List
from .config import NurtureConfig, DEFAULT_CONFIG, STANCE_DIMENSIONS
from .state import NurtureState


def encode_environment(env_features: Dict[str, Any], d_env: int = 512) -> np.ndarray:
    """
    Encode environment features to vector representation.
    For Phase 1, use simple categorical encoding.
    """
    vector = np.zeros(d_env)
    
    # Formality encoding (indices 0-4)
    formality_map = {
        'very informal': 0, 'informal': 1, 'neutral': 2, 
        'formal': 3, 'very formal': 4
    }
    formality = env_features.get('formality_level', 'neutral')
    formality_idx = formality_map.get(formality, 2)
    vector[formality_idx] = 1.0
    
    # Technical level (indices 5-8)
    tech_map = {'novice': 5, 'intermediate': 6, 'advanced': 7, 'expert': 8}
    tech = env_features.get('technical_level', 'intermediate')
    tech_idx = tech_map.get(tech, 6)
    vector[tech_idx] = 1.0
    
    # Emotional tone (indices 9-13)
    tone_map = {
        'hostile': 9, 'cold': 10, 'neutral': 11, 
        'warm': 12, 'very warm': 13
    }
    tone = env_features.get('emotional_tone', 'neutral')
    tone_idx = tone_map.get(tone, 11)
    vector[tone_idx] = 1.0
    
    # Pace (indices 14-18)
    pace_map = {
        'very slow': 14, 'slow': 15, 'moderate': 16,
        'fast': 17, 'very fast': 18
    }
    pace = env_features.get('pace_preference', 'moderate')
    pace_idx = pace_map.get(pace, 16)
    vector[pace_idx] = 1.0
    
    return vector


def update_N_env(
    N_env: np.ndarray,
    env_features: Dict[str, Any],
    learning_rate: float = 0.1
) -> np.ndarray:
    """
    Update environmental model with exponential moving average.
    This is UNGATED - updates freely based on observation.
    """
    env_vector = encode_environment(env_features, len(N_env))
    N_env_new = (1 - learning_rate) * N_env + learning_rate * env_vector
    return N_env_new


def update_env_json(
    env_json: Dict[str, Any],
    env_features: Dict[str, Any],
    learning_rate: float = 0.1,
    interaction_count: int = 0
) -> Dict[str, Any]:
    """
    Update JSON environment representation.
    For Phase 1 human-readable tracking.
    """
    updated = env_json.copy()
    
    # Update categorical fields directly
    categorical_fields = [
        'formality_level', 'technical_level', 
        'emotional_tone', 'pace_preference',
        'interaction_style', 'domain_focus',
        'user_expertise'
    ]
    
    for field in categorical_fields:
        if field in env_features and env_features[field]:
            # For categorical, just replace (EMA doesn't apply well)
            updated[field] = env_features[field]
    
    # relationship_depth progresses based on interaction count (more reliable)
    if interaction_count >= 30:
        updated['relationship_depth'] = 'established'
    elif interaction_count >= 15:
        updated['relationship_depth'] = 'developing'
    elif interaction_count >= 5:
        updated['relationship_depth'] = 'forming'
    else:
        updated['relationship_depth'] = 'new'
    
    # Update key traits with decay/consolidation
    if 'key_traits' in env_features and env_features['key_traits']:
        existing_traits = updated.get('key_traits', [])
        new_traits = [t.lower().strip() for t in env_features['key_traits'] if t]
        
        # Trait tracking with counts (simulate decay by limiting old traits)
        trait_counts = {}
        
        # New traits get higher weight
        for trait in new_traits:
            if trait:
                trait_counts[trait] = trait_counts.get(trait, 0) + 2
        
        # Existing traits (decay: only keep if reinforced or recent)
        for i, trait in enumerate(existing_traits[:8]):  # Keep max 8 old
            if trait:
                # More recent traits (lower index) get slightly higher weight
                weight = 1.0 - (i * 0.1)
                trait_counts[trait] = trait_counts.get(trait, 0) + weight
        
        # Sort by count and keep top 6
        sorted_traits = sorted(trait_counts.items(), key=lambda x: -x[1])
        updated['key_traits'] = [t[0] for t in sorted_traits[:6]]
    
    return updated


def compute_stance_delta(
    stance_updates: Dict[str, float],
    current_stance: Dict[str, float]
) -> Tuple[Dict[str, float], float]:
    """
    Compute stance delta vector and magnitude.
    
    Returns:
        Tuple of (delta_dict, delta_magnitude)
    """
    delta = {}
    squared_sum = 0.0
    
    for dim in STANCE_DIMENSIONS:
        if dim in stance_updates:
            update_direction = stance_updates[dim]
            delta[dim] = update_direction
            squared_sum += update_direction ** 2
        else:
            delta[dim] = 0.0
    
    magnitude = np.sqrt(squared_sum)
    return delta, magnitude


def update_N_stance(
    N_stance: np.ndarray,
    stance_json: Dict[str, float],
    stance_updates: Dict[str, float],
    alignment_score: float,
    plasticity: float,
    config: NurtureConfig = DEFAULT_CONFIG,
    debug: bool = False
) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Update relational stance, GATED by alignment score and plasticity.
    
    High alignment = stronger update
    Low alignment = note but resist
    Low plasticity = smaller updates
    
    Returns:
        Tuple of (new_N_stance, new_stance_json, delta_magnitude)
    """
    # Compute raw delta
    delta, raw_magnitude = compute_stance_delta(stance_updates, stance_json)
    
    # Gate by alignment: linear instead of squared for less aggressive gating
    alignment_gate = alignment_score  # Changed from alignment_score ** 2
    
    # Gate by plasticity
    plasticity_gate = plasticity
    
    # Effective learning rate
    effective_lr = config.STANCE_BASE_LR * alignment_gate * plasticity_gate
    
    # Debug logging
    if debug or raw_magnitude > 0:
        print(f"[STANCE UPDATE] raw_mag={raw_magnitude:.3f}, align={alignment_score:.2f}, "
              f"plast={plasticity:.2f}, eff_lr={effective_lr:.3f}")
        if stance_updates:
            print(f"  Recommendations: {stance_updates}")
        else:
            print(f"  Recommendations: NONE (LLM said maintain all)")
    
    # Update stance JSON
    new_stance_json = stance_json.copy()
    for dim, delta_val in delta.items():
        if dim in new_stance_json:
            new_val = new_stance_json[dim] + effective_lr * delta_val
            # Clamp to [0, 1]
            new_stance_json[dim] = max(0.0, min(1.0, new_val))
    
    # Update vector representation
    new_N_stance = N_stance.copy()
    for i, dim in enumerate(STANCE_DIMENSIONS):
        if i < len(new_N_stance) and dim in new_stance_json:
            new_N_stance[i] = new_stance_json[dim]
    
    # Actual delta magnitude (after gating)
    actual_magnitude = raw_magnitude * effective_lr
    
    return new_N_stance, new_stance_json, actual_magnitude


def update_stability(
    current_stability: float,
    delta_magnitude: float,
    delta_history: List[float],
    config: NurtureConfig = DEFAULT_CONFIG
) -> Tuple[float, List[float]]:
    """
    Update stability score based on recent delta history.
    Stability is inverse of average delta magnitude.
    """
    # Update history
    new_history = delta_history.copy()
    new_history.append(delta_magnitude)
    if len(new_history) > config.DELTA_HISTORY_WINDOW:
        new_history.pop(0)
    
    # Compute average delta
    if len(new_history) > 0:
        avg_delta = np.mean(new_history)
    else:
        avg_delta = 0.0
    
    # Stability is inverse of average delta
    raw_stability = 1.0 / (1.0 + avg_delta * config.STABILITY_SENSITIVITY)
    
    # Smooth the stability score
    new_stability = (
        config.STABILITY_SMOOTHING * current_stability +
        (1 - config.STABILITY_SMOOTHING) * raw_stability
    )
    
    return new_stability, new_history


def compute_plasticity(
    stability: float,
    config: NurtureConfig = DEFAULT_CONFIG
) -> float:
    """
    Derive plasticity from stability.
    Plasticity = 1 - stability, with minimum floor.
    """
    plasticity = 1.0 - stability
    return max(plasticity, config.MIN_PLASTICITY)


def check_for_shock(
    delta_magnitude: float,
    plasticity: float,
    config: NurtureConfig = DEFAULT_CONFIG
) -> float:
    """
    Detect shock events that temporarily increase plasticity.
    
    Returns:
        Plasticity boost (0 if no shock)
    """
    # Shock threshold is higher when already stable
    shock_threshold = config.SHOCK_BASE + (1 - plasticity) * config.SHOCK_RANGE
    
    if delta_magnitude > shock_threshold:
        # Significant destabilizing event
        plasticity_boost = min(
            delta_magnitude * config.SHOCK_RESPONSE,
            config.MAX_SHOCK_BOOST
        )
        return plasticity_boost
    
    return 0.0


def process_shock(
    current_plasticity: float,
    shock_boost: float,
    config: NurtureConfig = DEFAULT_CONFIG
) -> float:
    """
    Apply shock boost to plasticity.
    """
    new_plasticity = current_plasticity + shock_boost
    # Never fully reopen
    return min(new_plasticity, config.MAX_REOPEN_PLASTICITY)


def update_stable_count(
    stable_count: int,
    delta_magnitude: float,
    threshold: float = 0.01
) -> int:
    """
    Track consecutive stable interactions.
    """
    if delta_magnitude < threshold:
        return stable_count + 1
    else:
        return 0  # Reset on any significant change
