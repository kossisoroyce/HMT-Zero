"""
Experiential Layer - Runtime learning within character bounds.

Part of the CACA (Character-Aware Cognitive Architecture) system.
"""

from .state import (
    ExperientialState,
    ConversationModel,
    WorkingMemory,
    PersistentTraces,
    SalientFact,
    OpenQuestion,
    Commitment,
    initialize_experiential_state,
)
from .config import ExperientialConfig, DEFAULT_EXPERIENTIAL_CONFIG
from .engine import ExperientialEngine
from .gates import nature_gate, nurture_gate, apply_experiential_gates

__all__ = [
    'ExperientialState',
    'ConversationModel',
    'WorkingMemory',
    'PersistentTraces',
    'SalientFact',
    'OpenQuestion',
    'Commitment',
    'initialize_experiential_state',
    'ExperientialConfig',
    'DEFAULT_EXPERIENTIAL_CONFIG',
    'ExperientialEngine',
    'nature_gate',
    'nurture_gate',
    'apply_experiential_gates',
]
