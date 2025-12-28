"""
Significance filter for determining when full evaluation is needed.
"""
import re
import numpy as np
from typing import Dict, Any, Tuple
from .config import (
    NurtureConfig, DEFAULT_CONFIG,
    VALUE_KEYWORDS, POSITIVE_FEEDBACK, NEGATIVE_FEEDBACK
)
from .state import NurtureState


def compute_sentiment_magnitude(text: str) -> float:
    """
    Compute sentiment magnitude (strong positive or negative).
    Returns value between 0 and 1.
    """
    text_lower = text.lower()
    
    # Simple lexicon-based approach for prototype
    positive_markers = [
        'amazing', 'wonderful', 'fantastic', 'excellent', 'love',
        'brilliant', 'outstanding', 'incredible', 'awesome', 'perfect',
        '!', 'thank you so much', 'really appreciate'
    ]
    negative_markers = [
        'terrible', 'awful', 'horrible', 'hate', 'disgusting',
        'worst', 'pathetic', 'useless', 'stupid', 'disappointed',
        'angry', 'frustrated', 'annoyed', 'upset'
    ]
    
    positive_count = sum(1 for marker in positive_markers if marker in text_lower)
    negative_count = sum(1 for marker in negative_markers if marker in text_lower)
    
    # Exclamation marks indicate intensity
    exclamation_count = text.count('!')
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    intensity = min(1.0, (positive_count + negative_count) * 0.2 + 
                   exclamation_count * 0.1 + caps_ratio * 0.5)
    
    return intensity


def compute_value_keyword_presence(text: str) -> float:
    """
    Check for presence of value-relevant keywords.
    Returns value between 0 and 1.
    """
    text_lower = text.lower()
    
    keyword_count = sum(1 for keyword in VALUE_KEYWORDS if keyword in text_lower)
    
    # Normalize: 3+ keywords = max score
    return min(1.0, keyword_count / 3)


def compute_novelty_score(text: str, N_env: np.ndarray, env_json: Dict[str, Any]) -> float:
    """
    Compute novelty score based on divergence from environment expectations.
    For Phase 1, use simple heuristics.
    """
    text_lower = text.lower()
    
    novelty_indicators = 0
    
    # Topic shift detection (simple keyword-based)
    topic_shift_markers = [
        'by the way', 'speaking of', 'changing topic', 'different question',
        'unrelated', 'off topic', 'new subject', 'something else'
    ]
    if any(marker in text_lower for marker in topic_shift_markers):
        novelty_indicators += 1
    
    # New information markers
    new_info_markers = [
        'actually', 'in fact', 'turns out', 'discovered', 'realized',
        'never told you', 'should mention', 'forgot to say'
    ]
    if any(marker in text_lower for marker in new_info_markers):
        novelty_indicators += 1
    
    # Question complexity (longer questions often seek new ground)
    if '?' in text and len(text) > 100:
        novelty_indicators += 0.5
    
    # Unexpected formality shift
    formal_markers = ['furthermore', 'moreover', 'consequently', 'therefore', 'hence']
    informal_markers = ['gonna', 'wanna', 'kinda', 'sorta', 'lol', 'lmao', 'haha']
    
    current_formality = env_json.get('formality_level', 'neutral')
    has_formal = any(m in text_lower for m in formal_markers)
    has_informal = any(m in text_lower for m in informal_markers)
    
    if current_formality == 'formal' and has_informal:
        novelty_indicators += 1
    elif current_formality == 'informal' and has_formal:
        novelty_indicators += 1
    
    return min(1.0, novelty_indicators / 2)


def compute_contradiction_score(text: str, N_env: np.ndarray, env_json: Dict[str, Any]) -> float:
    """
    Detect contradictions with existing state.
    """
    text_lower = text.lower()
    
    contradiction_indicators = 0
    
    # Explicit contradiction markers
    contradiction_markers = [
        'actually no', 'that\'s wrong', 'i didn\'t mean', 'not what i said',
        'you misunderstood', 'let me correct', 'i changed my mind',
        'forget what i said', 'opposite', 'contrary'
    ]
    if any(marker in text_lower for marker in contradiction_markers):
        contradiction_indicators += 2
    
    # Negation of previous statements
    negation_patterns = [
        r"i (don't|do not|never|didn't)",
        r"that's not",
        r"no,",
        r"wrong"
    ]
    for pattern in negation_patterns:
        if re.search(pattern, text_lower):
            contradiction_indicators += 0.5
    
    return min(1.0, contradiction_indicators / 2)


def compute_user_feedback_score(text: str) -> float:
    """
    Detect user feedback (corrections, praise, criticism).
    """
    text_lower = text.lower()
    
    positive_count = sum(1 for marker in POSITIVE_FEEDBACK if marker in text_lower)
    negative_count = sum(1 for marker in NEGATIVE_FEEDBACK if marker in text_lower)
    
    # Strong feedback in either direction is significant
    return min(1.0, (positive_count + negative_count * 1.5) / 2)


def parse_self_assessment_tag(tag: str) -> float:
    """Convert self-assessment tag to numeric value."""
    tag_map = {
        'low': 0.0,
        'medium': 0.5,
        'high': 1.0
    }
    return tag_map.get(tag.lower(), 0.5)


def compute_significance(
    interaction_text: str,
    nurture_state: NurtureState,
    self_assessment_tag: str = 'medium',
    config: NurtureConfig = DEFAULT_CONFIG
) -> Tuple[float, Dict[str, float]]:
    """
    Compute overall significance score for an interaction.
    
    Args:
        interaction_text: The user's input text
        nurture_state: Current nurture state
        self_assessment_tag: Model's self-assessment (low/medium/high)
        config: Configuration parameters
    
    Returns:
        Tuple of (significance_score, component_scores)
    """
    # Compute individual components
    sentiment = compute_sentiment_magnitude(interaction_text)
    value_keywords = compute_value_keyword_presence(interaction_text)
    novelty = compute_novelty_score(
        interaction_text, 
        nurture_state.N_env, 
        nurture_state.env_json
    )
    contradiction = compute_contradiction_score(
        interaction_text,
        nurture_state.N_env,
        nurture_state.env_json
    )
    feedback = compute_user_feedback_score(interaction_text)
    
    # Weighted heuristic score
    heuristic_score = (
        config.SENTIMENT_WEIGHT * sentiment +
        config.VALUE_KEYWORD_WEIGHT * value_keywords +
        config.NOVELTY_WEIGHT * novelty +
        config.CONTRADICTION_WEIGHT * contradiction +
        config.FEEDBACK_WEIGHT * feedback
    )
    
    # Self-assessment score
    self_assessment = parse_self_assessment_tag(self_assessment_tag)
    
    # Combined score
    combined = (
        config.HEURISTIC_WEIGHT * heuristic_score +
        config.SELF_ASSESSMENT_WEIGHT * self_assessment
    )
    
    component_scores = {
        'sentiment': sentiment,
        'value_keywords': value_keywords,
        'novelty': novelty,
        'contradiction': contradiction,
        'feedback': feedback,
        'heuristic_total': heuristic_score,
        'self_assessment': self_assessment,
        'combined': combined
    }
    
    return combined, component_scores


def should_evaluate(
    significance_score: float,
    plasticity: float,
    config: NurtureConfig = DEFAULT_CONFIG
) -> bool:
    """
    Determine if full evaluation pass is needed.
    
    Lower threshold when more plastic (early formative period).
    Higher threshold when stabilized.
    """
    dynamic_threshold = (
        config.BASE_THRESHOLD + 
        (1 - plasticity) * config.THRESHOLD_RANGE
    )
    
    return significance_score > dynamic_threshold


def get_dynamic_threshold(
    plasticity: float,
    config: NurtureConfig = DEFAULT_CONFIG
) -> float:
    """Get the current dynamic threshold for evaluation."""
    return config.BASE_THRESHOLD + (1 - plasticity) * config.THRESHOLD_RANGE
