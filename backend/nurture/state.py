"""
NurtureState data structures for the Nurture Layer system.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import json
import uuid


@dataclass
class NurtureState:
    """
    The complete nurture state for an AI instance.
    
    Contains:
    - N_env: Environmental model (what the world is like)
    - N_stance: Relational stance (how I relate to it)
    - Stability/plasticity tracking
    - Interaction history metadata
    """
    instance_id: str
    N_env: np.ndarray                    # Environmental model vector
    N_stance: np.ndarray                 # Relational stance vector
    stability: float = 0.0              # Current stability score (0-1)
    plasticity: float = 1.0             # Current plasticity (1 - stability)
    interaction_count: int = 0          # Total interactions
    significant_count: int = 0          # Significant interactions evaluated
    stable_count: int = 0               # Consecutive stable interactions
    delta_history: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    
    # Phase tracking
    phase: str = "rapid_formation"      # rapid_formation, consolidation, stabilization, maturity
    
    # JSON-based stance representation for Phase 1 (human-readable)
    stance_json: Dict[str, float] = field(default_factory=dict)
    env_json: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            'instance_id': self.instance_id,
            'N_env': self.N_env.tolist(),
            'N_stance': self.N_stance.tolist(),
            'stability': self.stability,
            'plasticity': self.plasticity,
            'interaction_count': self.interaction_count,
            'significant_count': self.significant_count,
            'stable_count': self.stable_count,
            'delta_history': self.delta_history,
            'last_updated': self.last_updated.isoformat(),
            'created_at': self.created_at.isoformat(),
            'phase': self.phase,
            'stance_json': self.stance_json,
            'env_json': self.env_json,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NurtureState':
        """Deserialize state from dictionary."""
        return cls(
            instance_id=data['instance_id'],
            N_env=np.array(data['N_env']),
            N_stance=np.array(data['N_stance']),
            stability=data['stability'],
            plasticity=data['plasticity'],
            interaction_count=data['interaction_count'],
            significant_count=data['significant_count'],
            stable_count=data.get('stable_count', 0),
            delta_history=data.get('delta_history', []),
            last_updated=datetime.fromisoformat(data['last_updated']),
            created_at=datetime.fromisoformat(data['created_at']),
            phase=data.get('phase', 'rapid_formation'),
            stance_json=data.get('stance_json', {}),
            env_json=data.get('env_json', {}),
        )
    
    def get_phase(self) -> str:
        """Determine current formative phase based on stability."""
        if self.stability < 0.5:
            return "rapid_formation"
        elif self.stability < 0.9:
            return "consolidation"
        elif self.stable_count < 10:  # CONFIRMATION_WINDOW
            return "stabilization"
        else:
            return "maturity"
    
    def update_phase(self):
        """Update the phase based on current stability."""
        self.phase = self.get_phase()
    
    def can_accept_promotion(self, plasticity_threshold: float = 0.2) -> bool:
        """Check if nurture state can accept promotion from experiential layer."""
        return self.plasticity >= plasticity_threshold
    
    def get_stance_bias(self, dimension: str) -> float:
        """Get the stance bias for a given dimension (for experiential gating)."""
        return self.stance_json.get(dimension, 0.5)
    
    def get_domain_alignment(self, topic: str) -> float:
        """Compute alignment between a topic and nurture's domain focus."""
        domain = self.env_json.get('domain_focus', 'general')
        if domain == 'general':
            return 1.0  # General domain accepts all topics
        # Simple keyword matching for now
        if domain.lower() in topic.lower():
            return 1.0
        return 0.5  # Partial alignment for unrelated topics
    
    def get_emotionality_bound(self) -> float:
        """Get the maximum emotional magnitude allowed by nurture."""
        emotionality = self.stance_json.get('emotionality', 0.5)
        return 0.5 + 0.5 * emotionality  # Range: 0.5 to 1.0
    
    def get_relationship_depth_factor(self) -> float:
        """Get depth factor for user state modeling."""
        depth = self.env_json.get('relationship_depth', 'new')
        factors = {'new': 0.5, 'developing': 0.75, 'established': 1.0}
        return factors.get(depth, 0.5)


def initialize_nurture_state(
    instance_id: Optional[str] = None,
    d_env: int = 512,
    d_stance: int = 256
) -> NurtureState:
    """
    Initialize a new nurture state with neutral values.
    
    Args:
        instance_id: Unique identifier for this instance
        d_env: Dimension of environment vector
        d_stance: Dimension of stance vector
    
    Returns:
        Initialized NurtureState with neutral values
    """
    if instance_id is None:
        instance_id = str(uuid.uuid4())
    
    # Initialize vectors to neutral (0.5 for normalized dimensions)
    N_env = np.zeros(d_env)
    N_stance = np.full(d_stance, 0.5)  # Neutral stance: midpoint on all dimensions
    
    # Initialize JSON representations for Phase 1
    stance_json = {
        'warmth': 0.5,
        'formality': 0.5,
        'depth': 0.5,
        'pace': 0.5,
        'directness': 0.5,
        'playfulness': 0.5,
        'assertiveness': 0.5,
        'emotionality': 0.5,
    }
    
    env_json = {
        'formality_level': 'neutral',
        'technical_level': 'intermediate',
        'emotional_tone': 'neutral',
        'pace_preference': 'moderate',
        'interaction_style': 'balanced',
        'domain_focus': 'general',
        'user_expertise': 'unknown',
        'relationship_depth': 'new',
        'key_traits': [],
    }
    
    return NurtureState(
        instance_id=instance_id,
        N_env=N_env,
        N_stance=N_stance,
        stability=0.0,
        plasticity=1.0,
        interaction_count=0,
        significant_count=0,
        stable_count=0,
        delta_history=[],
        stance_json=stance_json,
        env_json=env_json,
    )


@dataclass
class EvaluationResult:
    """Result of an evaluation pass."""
    environment: Dict[str, Any]
    alignment_score: float
    stance_updates: Dict[str, float]
    tensions: List[str]
    raw_evaluation: str


@dataclass
class InteractionMetadata:
    """Metadata about an interaction."""
    significance_score: float
    significance_tag: str  # low, medium, high
    was_evaluated: bool
    delta_magnitude: float
    shock_detected: bool
    phase_before: str
    phase_after: str
