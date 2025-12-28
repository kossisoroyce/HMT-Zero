"""
ExperientialState data structures for the Experiential Layer.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import uuid


@dataclass
class SalientFact:
    """A fact extracted from conversation that's worth remembering."""
    content: str
    source: str  # "user" or "assistant"
    timestamp: datetime
    salience_score: float
    references: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'salience_score': self.salience_score,
            'references': self.references,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SalientFact':
        return cls(
            content=data['content'],
            source=data['source'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            salience_score=data['salience_score'],
            references=data.get('references', 0),
        )


@dataclass
class OpenQuestion:
    """An unresolved question from the conversation."""
    question: str
    context: str
    asked_at: datetime
    attempted_answers: int = 0
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'context': self.context,
            'asked_at': self.asked_at.isoformat(),
            'attempted_answers': self.attempted_answers,
            'resolved': self.resolved,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OpenQuestion':
        return cls(
            question=data['question'],
            context=data['context'],
            asked_at=datetime.fromisoformat(data['asked_at']),
            attempted_answers=data.get('attempted_answers', 0),
            resolved=data.get('resolved', False),
        )


@dataclass
class Commitment:
    """A promise or commitment made during conversation."""
    promise: str
    context: str
    made_at: datetime
    fulfilled: bool = False
    deadline: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'promise': self.promise,
            'context': self.context,
            'made_at': self.made_at.isoformat(),
            'fulfilled': self.fulfilled,
            'deadline': self.deadline.isoformat() if self.deadline else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Commitment':
        return cls(
            promise=data['promise'],
            context=data['context'],
            made_at=datetime.fromisoformat(data['made_at']),
            fulfilled=data.get('fulfilled', False),
            deadline=datetime.fromisoformat(data['deadline']) if data.get('deadline') else None,
        )


@dataclass
class ConversationModel:
    """Tracks the dynamics of the current conversation."""
    topic_vector: np.ndarray          # Current topic embedding
    emotional_trajectory: np.ndarray   # Emotional arc
    user_state_estimate: np.ndarray    # Model of user's current state
    interaction_count: int = 0
    
    # Human-readable summaries
    topic_summary: str = ""
    emotion_summary: str = ""
    user_summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'topic_vector': self.topic_vector.tolist(),
            'emotional_trajectory': self.emotional_trajectory.tolist(),
            'user_state_estimate': self.user_state_estimate.tolist(),
            'interaction_count': self.interaction_count,
            'topic_summary': self.topic_summary,
            'emotion_summary': self.emotion_summary,
            'user_summary': self.user_summary,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationModel':
        return cls(
            topic_vector=np.array(data['topic_vector']),
            emotional_trajectory=np.array(data['emotional_trajectory']),
            user_state_estimate=np.array(data['user_state_estimate']),
            interaction_count=data.get('interaction_count', 0),
            topic_summary=data.get('topic_summary', ''),
            emotion_summary=data.get('emotion_summary', ''),
            user_summary=data.get('user_summary', ''),
        )


@dataclass
class WorkingMemory:
    """Structured working memory for the session."""
    salient_facts: List[SalientFact] = field(default_factory=list)
    open_questions: List[OpenQuestion] = field(default_factory=list)
    commitments: List[Commitment] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'salient_facts': [f.to_dict() for f in self.salient_facts],
            'open_questions': [q.to_dict() for q in self.open_questions],
            'commitments': [c.to_dict() for c in self.commitments],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkingMemory':
        return cls(
            salient_facts=[SalientFact.from_dict(f) for f in data.get('salient_facts', [])],
            open_questions=[OpenQuestion.from_dict(q) for q in data.get('open_questions', [])],
            commitments=[Commitment.from_dict(c) for c in data.get('commitments', [])],
        )


@dataclass
class PersistentTraces:
    """Cross-session pattern accumulator."""
    pattern_accumulator: np.ndarray    # Patterns seen across sessions
    familiarity_score: float = 0.0     # How familiar is this context
    session_count: int = 0             # Sessions with this user
    last_session_end: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_accumulator': self.pattern_accumulator.tolist(),
            'familiarity_score': self.familiarity_score,
            'session_count': self.session_count,
            'last_session_end': self.last_session_end.isoformat() if self.last_session_end else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistentTraces':
        return cls(
            pattern_accumulator=np.array(data['pattern_accumulator']),
            familiarity_score=data.get('familiarity_score', 0.0),
            session_count=data.get('session_count', 0),
            last_session_end=datetime.fromisoformat(data['last_session_end']) if data.get('last_session_end') else None,
        )


@dataclass
class ExperientialState:
    """
    Complete experiential state for a session.
    
    Contains:
    - Conversation model (topic, emotion, user state)
    - Working memory (facts, questions, commitments)
    - Persistent traces (cross-session patterns)
    """
    session_id: str
    conversation_model: ConversationModel
    working_memory: WorkingMemory
    persistent_traces: PersistentTraces
    
    # Metadata
    session_start: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    total_updates: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'conversation_model': self.conversation_model.to_dict(),
            'working_memory': self.working_memory.to_dict(),
            'persistent_traces': self.persistent_traces.to_dict(),
            'session_start': self.session_start.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'total_updates': self.total_updates,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperientialState':
        return cls(
            session_id=data['session_id'],
            conversation_model=ConversationModel.from_dict(data['conversation_model']),
            working_memory=WorkingMemory.from_dict(data['working_memory']),
            persistent_traces=PersistentTraces.from_dict(data['persistent_traces']),
            session_start=datetime.fromisoformat(data['session_start']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            total_updates=data.get('total_updates', 0),
        )
    
    def get_context_string(self) -> str:
        """Generate human-readable context for prompt injection."""
        parts = []
        
        cm = self.conversation_model
        if cm.interaction_count > 0:
            if cm.topic_summary:
                parts.append(f"Current focus: {cm.topic_summary}")
            if cm.emotion_summary:
                parts.append(f"Conversation tone: {cm.emotion_summary}")
            if cm.user_summary:
                parts.append(f"User state: {cm.user_summary}")
        
        wm = self.working_memory
        if wm.salient_facts:
            facts = [f.content for f in wm.salient_facts[:5]]
            parts.append(f"Key points: {'; '.join(facts)}")
        
        if wm.open_questions:
            questions = [q.question for q in wm.open_questions if not q.resolved][:3]
            if questions:
                parts.append(f"Open questions: {'; '.join(questions)}")
        
        if wm.commitments:
            active = [c.promise for c in wm.commitments if not c.fulfilled][:3]
            if active:
                parts.append(f"Commitments: {'; '.join(active)}")
        
        pt = self.persistent_traces
        if pt.session_count > 1:
            if pt.familiarity_score > 0.7:
                parts.append("This context feels familiar from previous sessions.")
            elif pt.familiarity_score > 0.4:
                parts.append("Some patterns here echo previous sessions.")
        
        return "\n".join(parts) if parts else ""


def initialize_experiential_state(
    session_id: Optional[str] = None,
    persistent_traces: Optional[PersistentTraces] = None,
    d_topic: int = 128,
    d_emotion: int = 32,
    d_user: int = 64,
    d_pattern: int = 256,
) -> ExperientialState:
    """
    Initialize a new experiential state for a session.
    
    Args:
        session_id: Unique session identifier
        persistent_traces: Existing persistent traces (for returning users)
        d_topic: Topic vector dimension
        d_emotion: Emotion vector dimension
        d_user: User state dimension
        d_pattern: Pattern accumulator dimension
    
    Returns:
        Initialized ExperientialState
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    conversation_model = ConversationModel(
        topic_vector=np.zeros(d_topic),
        emotional_trajectory=np.zeros(d_emotion),
        user_state_estimate=np.zeros(d_user),
        interaction_count=0,
    )
    
    working_memory = WorkingMemory()
    
    if persistent_traces is None:
        persistent_traces = PersistentTraces(
            pattern_accumulator=np.zeros(d_pattern),
            familiarity_score=0.0,
            session_count=0,
        )
    
    return ExperientialState(
        session_id=session_id,
        conversation_model=conversation_model,
        working_memory=working_memory,
        persistent_traces=persistent_traces,
    )
