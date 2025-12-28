"""
Configuration for the Experiential Layer.
Based on CACA Experiential Layer Technical Paper.
"""
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class ExperientialConfig:
    """Configuration parameters for the Experiential Layer."""
    
    # Dimensions
    D_TRACE: int = 256           # Activation trace dimension
    D_TOPIC: int = 128           # Topic vector dimension
    D_EMOTION: int = 32          # Emotional trajectory dimension
    D_USER: int = 64             # User state model dimension
    D_PATTERN: int = 256         # Cross-session pattern dimension
    
    # Decay rates
    TRACE_DECAY_SHALLOW: float = 0.80
    TRACE_DECAY_DEEP: float = 0.95
    TOPIC_DECAY: float = 0.70
    EMOTION_DECAY: float = 0.85
    USER_STATE_DECAY: float = 0.75
    FACT_DECAY: float = 0.95
    PATTERN_DECAY_CROSS_SESSION: float = 0.90
    
    # Memory limits
    MAX_SALIENT_FACTS: int = 20
    MAX_OPEN_QUESTIONS: int = 10
    MAX_COMMITMENTS: int = 10
    
    # Thresholds
    SALIENCE_THRESHOLD: float = 0.3
    MIN_SALIENCE: float = 0.1
    VALUE_THRESHOLD: float = 0.5
    RESOLUTION_LINGER_SECONDS: int = 300
    FULFILLMENT_LINGER_SECONDS: int = 600
    MAX_QUESTION_ATTEMPTS: int = 5
    
    # Cross-session
    PATTERN_HALF_LIFE_DAYS: float = 7.0
    
    # Gating
    STANCE_INFLUENCE: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'd_trace': self.D_TRACE,
            'd_topic': self.D_TOPIC,
            'd_emotion': self.D_EMOTION,
            'd_user': self.D_USER,
            'd_pattern': self.D_PATTERN,
            'trace_decay_shallow': self.TRACE_DECAY_SHALLOW,
            'trace_decay_deep': self.TRACE_DECAY_DEEP,
            'topic_decay': self.TOPIC_DECAY,
            'emotion_decay': self.EMOTION_DECAY,
            'user_state_decay': self.USER_STATE_DECAY,
            'fact_decay': self.FACT_DECAY,
            'pattern_decay_cross_session': self.PATTERN_DECAY_CROSS_SESSION,
            'max_salient_facts': self.MAX_SALIENT_FACTS,
            'max_open_questions': self.MAX_OPEN_QUESTIONS,
            'max_commitments': self.MAX_COMMITMENTS,
            'salience_threshold': self.SALIENCE_THRESHOLD,
            'min_salience': self.MIN_SALIENCE,
            'value_threshold': self.VALUE_THRESHOLD,
            'stance_influence': self.STANCE_INFLUENCE,
        }


# Default configuration instance
DEFAULT_EXPERIENTIAL_CONFIG = ExperientialConfig()


# Emotion keywords for trajectory tracking
EMOTION_POSITIVE = [
    'happy', 'glad', 'excited', 'grateful', 'thankful', 'love',
    'wonderful', 'amazing', 'great', 'fantastic', 'joy', 'pleased'
]

EMOTION_NEGATIVE = [
    'sad', 'angry', 'frustrated', 'disappointed', 'upset', 'worried',
    'anxious', 'stressed', 'confused', 'hurt', 'annoyed', 'scared'
]

EMOTION_NEUTRAL = [
    'okay', 'fine', 'alright', 'neutral', 'so-so', 'meh'
]

# Commitment indicators
COMMITMENT_PHRASES = [
    'i will', 'i\'ll', 'let me', 'i can', 'i\'m going to',
    'i promise', 'i\'ll make sure', 'count on me', 'i\'ll help'
]

# Question patterns
QUESTION_INDICATORS = ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'which', 'could you', 'can you']
