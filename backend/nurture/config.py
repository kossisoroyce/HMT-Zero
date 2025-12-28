"""
Configuration for the Nurture Layer system.
Based on CACA Nurture Layer Technical Paper.
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class NurtureConfig:
    """Configuration parameters for the Nurture Layer."""
    
    # Dimensions
    D_ENV: int = 512
    D_STANCE: int = 256
    
    # Significance filter
    BASE_THRESHOLD: float = 0.3
    THRESHOLD_RANGE: float = 0.4
    
    # Learning rates (increased for visible changes)
    ENV_LEARNING_RATE: float = 0.3
    STANCE_BASE_LR: float = 0.5
    MINOR_ENV_LR: float = 0.1
    
    # Stability (tuned for faster progression)
    STABILITY_SENSITIVITY: float = 5.0  # Lower = faster stability growth
    STABILITY_THRESHOLD: float = 0.95
    CONFIRMATION_WINDOW: int = 10
    STABILITY_SMOOTHING: float = 0.8  # Lower = more responsive to changes
    
    # Plasticity
    MIN_PLASTICITY: float = 0.05
    
    # Shock
    SHOCK_BASE: float = 0.5
    SHOCK_RANGE: float = 0.3
    SHOCK_RESPONSE: float = 0.5
    MAX_SHOCK_BOOST: float = 0.3
    MAX_REOPEN_PLASTICITY: float = 0.4
    
    # Window sizes
    DELTA_HISTORY_WINDOW: int = 20
    
    # Experiential Layer Integration
    PROMOTION_PLASTICITY_THRESHOLD: float = 0.2  # Nurture must be this plastic for promotion
    PROMOTION_MIN_SESSIONS: int = 10             # Min sessions before promotion possible
    PROMOTION_STABILITY_THRESHOLD: float = 0.9   # Pattern must be this stable
    STANCE_INFLUENCE: float = 0.1                # How much nurture biases experiential updates
    
    # Significance weights
    SENTIMENT_WEIGHT: float = 0.2
    VALUE_KEYWORD_WEIGHT: float = 0.2
    NOVELTY_WEIGHT: float = 0.25
    CONTRADICTION_WEIGHT: float = 0.2
    FEEDBACK_WEIGHT: float = 0.15
    
    # Self-assessment vs heuristic balance
    HEURISTIC_WEIGHT: float = 0.6
    SELF_ASSESSMENT_WEIGHT: float = 0.4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'D_ENV': self.D_ENV,
            'D_STANCE': self.D_STANCE,
            'BASE_THRESHOLD': self.BASE_THRESHOLD,
            'THRESHOLD_RANGE': self.THRESHOLD_RANGE,
            'ENV_LEARNING_RATE': self.ENV_LEARNING_RATE,
            'STANCE_BASE_LR': self.STANCE_BASE_LR,
            'MINOR_ENV_LR': self.MINOR_ENV_LR,
            'STABILITY_SENSITIVITY': self.STABILITY_SENSITIVITY,
            'STABILITY_THRESHOLD': self.STABILITY_THRESHOLD,
            'CONFIRMATION_WINDOW': self.CONFIRMATION_WINDOW,
            'STABILITY_SMOOTHING': self.STABILITY_SMOOTHING,
            'MIN_PLASTICITY': self.MIN_PLASTICITY,
            'SHOCK_BASE': self.SHOCK_BASE,
            'SHOCK_RANGE': self.SHOCK_RANGE,
            'SHOCK_RESPONSE': self.SHOCK_RESPONSE,
            'MAX_SHOCK_BOOST': self.MAX_SHOCK_BOOST,
            'MAX_REOPEN_PLASTICITY': self.MAX_REOPEN_PLASTICITY,
            'DELTA_HISTORY_WINDOW': self.DELTA_HISTORY_WINDOW,
        }


# Default configuration instance
DEFAULT_CONFIG = NurtureConfig()


# Value-relevant keywords for significance detection
VALUE_KEYWORDS = [
    'should', 'shouldn\'t', 'must', 'mustn\'t', 'ought',
    'right', 'wrong', 'good', 'bad', 'evil',
    'ethical', 'moral', 'immoral', 'values', 'principles',
    'boundaries', 'limits', 'acceptable', 'unacceptable',
    'prefer', 'hate', 'love', 'important', 'matters',
    'always', 'never', 'promise', 'trust', 'honest',
    'fair', 'unfair', 'just', 'unjust', 'harm', 'help',
    'respect', 'disrespect', 'care', 'ignore'
]

# Feedback indicators
POSITIVE_FEEDBACK = [
    'thank', 'thanks', 'great', 'excellent', 'perfect',
    'exactly', 'helpful', 'amazing', 'love it', 'well done',
    'correct', 'right', 'yes', 'good job', 'appreciate'
]

NEGATIVE_FEEDBACK = [
    'wrong', 'incorrect', 'no', 'not what', 'don\'t',
    'stop', 'bad', 'terrible', 'useless', 'unhelpful',
    'mistake', 'error', 'fail', 'disappointed', 'frustrat'
]

# Stance dimension names (for Phase 1 JSON representation)
STANCE_DIMENSIONS = [
    'warmth',           # cold <-> warm
    'formality',        # casual <-> formal
    'depth',            # surface <-> deep
    'pace',             # slow <-> fast
    'directness',       # indirect <-> direct
    'playfulness',      # serious <-> playful
    'assertiveness',    # passive <-> assertive
    'emotionality',     # reserved <-> expressive
]

# Environment dimension names
ENV_DIMENSIONS = [
    'formality_level',
    'technical_level', 
    'emotional_tone',
    'pace_preference',
    'interaction_style',
    'domain_focus',
    'user_expertise',
    'relationship_depth',
]
