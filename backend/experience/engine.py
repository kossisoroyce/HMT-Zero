"""
Experiential Engine - Core processing for the Experiential Layer.
Handles state updates, memory management, and context generation.
"""
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import re

from .state import (
    ExperientialState,
    SalientFact,
    OpenQuestion,
    Commitment,
    initialize_experiential_state,
)
from .config import (
    ExperientialConfig,
    DEFAULT_EXPERIENTIAL_CONFIG,
    EMOTION_POSITIVE,
    EMOTION_NEGATIVE,
    COMMITMENT_PHRASES,
    QUESTION_INDICATORS,
)
from .gates import apply_experiential_gates, compute_promotion_candidate


class ExperientialEngine:
    """
    Engine for processing experiential updates.
    
    Manages:
    - Conversation model updates (topic, emotion, user state)
    - Working memory (facts, questions, commitments)
    - Cross-session persistence
    - Integration with Nurture Layer
    """
    
    def __init__(
        self,
        config: Optional[ExperientialConfig] = None,
        nurture_state: Any = None,
    ):
        self.config = config or DEFAULT_EXPERIENTIAL_CONFIG
        self.nurture_state = nurture_state
        self.state: Optional[ExperientialState] = None
    
    def initialize_session(
        self,
        session_id: Optional[str] = None,
        persistent_traces: Any = None,
    ) -> ExperientialState:
        """Initialize a new session."""
        self.state = initialize_experiential_state(
            session_id=session_id,
            persistent_traces=persistent_traces,
            d_topic=self.config.D_TOPIC,
            d_emotion=self.config.D_EMOTION,
            d_user=self.config.D_USER,
            d_pattern=self.config.D_PATTERN,
        )
        
        # Prime from nurture if available
        if self.nurture_state is not None:
            self._prime_from_nurture()
        
        return self.state
    
    def _prime_from_nurture(self):
        """Initialize experiential state with nurture-derived priors."""
        if self.state is None or self.nurture_state is None:
            return
        
        # Set expected emotional baseline from nurture stance
        emotionality = self.nurture_state.stance_json.get('emotionality', 0.5)
        warmth = self.nurture_state.stance_json.get('warmth', 0.5)
        
        # Create initial emotional trajectory based on nurture
        emotion_dim = self.config.D_EMOTION
        self.state.conversation_model.emotional_trajectory = np.zeros(emotion_dim)
        # Warmth affects positive emotion baseline
        self.state.conversation_model.emotional_trajectory[0] = warmth * 0.3
        # Emotionality affects overall expressiveness
        self.state.conversation_model.emotional_trajectory[1] = emotionality * 0.2
    
    def process_interaction(
        self,
        user_input: str,
        assistant_response: str,
    ) -> Dict[str, Any]:
        """
        Process an interaction and update experiential state.
        
        Args:
            user_input: The user's message
            assistant_response: The assistant's response
        
        Returns:
            Dict with update metadata
        """
        if self.state is None:
            self.initialize_session()
        
        # Update conversation model
        self._update_topic(user_input, assistant_response)
        self._update_emotional_trajectory(user_input, assistant_response)
        self._update_user_state(user_input)
        
        # Update working memory
        self._extract_salient_facts(user_input, assistant_response)
        self._track_questions(user_input, assistant_response)
        self._track_commitments(assistant_response)
        
        # Decay and prune
        self._apply_decay()
        self._prune_memory()
        
        # Update metadata
        self.state.conversation_model.interaction_count += 1
        self.state.total_updates += 1
        self.state.last_updated = datetime.now()
        
        return {
            'interaction_count': self.state.conversation_model.interaction_count,
            'facts_count': len(self.state.working_memory.salient_facts),
            'questions_count': len(self.state.working_memory.open_questions),
            'commitments_count': len(self.state.working_memory.commitments),
            'topic_summary': self.state.conversation_model.topic_summary,
            'emotion_summary': self.state.conversation_model.emotion_summary,
        }
    
    def _update_topic(self, user_input: str, assistant_response: str):
        """Update the topic vector based on current exchange."""
        # Simple keyword-based topic extraction
        combined = f"{user_input} {assistant_response}".lower()
        words = re.findall(r'\b\w+\b', combined)
        
        # Create simple bag-of-words style topic vector
        topic_dim = self.config.D_TOPIC
        current_topic = np.zeros(topic_dim)
        
        for i, word in enumerate(words[:topic_dim]):
            # Hash words to vector positions
            pos = hash(word) % topic_dim
            current_topic[pos] += 1
        
        # Normalize
        norm = np.linalg.norm(current_topic)
        if norm > 0:
            current_topic = current_topic / norm
        
        # Blend with existing topic
        decay = self.config.TOPIC_DECAY
        old_topic = self.state.conversation_model.topic_vector
        new_topic = decay * old_topic + (1 - decay) * current_topic
        
        # Apply gates
        gated = apply_experiential_gates(
            new_topic, 'topic_vector', self.nurture_state, self.config
        )
        if gated is not None:
            self.state.conversation_model.topic_vector = gated
        
        # Generate human-readable summary
        # Extract key nouns/topics from input
        key_words = [w for w in words if len(w) > 4][:5]
        if key_words:
            self.state.conversation_model.topic_summary = ", ".join(set(key_words))
    
    def _update_emotional_trajectory(self, user_input: str, assistant_response: str):
        """Update the emotional trajectory based on current exchange."""
        combined = f"{user_input} {assistant_response}".lower()
        
        # Simple sentiment analysis
        positive_count = sum(1 for word in EMOTION_POSITIVE if word in combined)
        negative_count = sum(1 for word in EMOTION_NEGATIVE if word in combined)
        
        emotion_dim = self.config.D_EMOTION
        current_emotion = np.zeros(emotion_dim)
        
        # Valence (positive-negative)
        valence = (positive_count - negative_count) / max(1, positive_count + negative_count + 1)
        current_emotion[0] = valence
        
        # Arousal (intensity)
        arousal = (positive_count + negative_count) / 10.0  # Normalize
        current_emotion[1] = min(1.0, arousal)
        
        # Blend with existing trajectory
        decay = self.config.EMOTION_DECAY
        old_emotion = self.state.conversation_model.emotional_trajectory
        new_emotion = decay * old_emotion + (1 - decay) * current_emotion
        
        # Apply gates
        gated = apply_experiential_gates(
            new_emotion, 'emotional_trajectory', self.nurture_state, self.config
        )
        if gated is not None:
            self.state.conversation_model.emotional_trajectory = gated
        
        # Generate summary
        if valence > 0.3:
            self.state.conversation_model.emotion_summary = "positive, engaged"
        elif valence < -0.3:
            self.state.conversation_model.emotion_summary = "concerned or frustrated"
        else:
            self.state.conversation_model.emotion_summary = "neutral"
    
    def _update_user_state(self, user_input: str):
        """Update the user state estimate."""
        user_dim = self.config.D_USER
        current_state = np.zeros(user_dim)
        
        input_lower = user_input.lower()
        
        # Engagement signals
        if '?' in user_input:
            current_state[0] = 0.8  # Curious/engaged
        if len(user_input) > 200:
            current_state[1] = 0.7  # Elaborative
        if any(word in input_lower for word in ['please', 'thanks', 'thank you']):
            current_state[2] = 0.9  # Polite
        if any(word in input_lower for word in ['urgent', 'asap', 'quickly']):
            current_state[3] = 0.8  # Time-pressured
        
        # Blend with existing estimate
        decay = self.config.USER_STATE_DECAY
        old_state = self.state.conversation_model.user_state_estimate
        new_state = decay * old_state + (1 - decay) * current_state
        
        # Apply gates
        gated = apply_experiential_gates(
            new_state, 'user_state', self.nurture_state, self.config
        )
        if gated is not None:
            self.state.conversation_model.user_state_estimate = gated
        
        # Generate summary
        summaries = []
        if current_state[0] > 0.5:
            summaries.append("curious")
        if current_state[2] > 0.5:
            summaries.append("appreciative")
        if current_state[3] > 0.5:
            summaries.append("time-sensitive")
        self.state.conversation_model.user_summary = ", ".join(summaries) if summaries else "neutral"
    
    def _extract_salient_facts(self, user_input: str, assistant_response: str):
        """Extract and store salient facts from the exchange."""
        wm = self.state.working_memory
        
        # Extract facts from user input (statements, not questions)
        sentences = re.split(r'[.!]', user_input)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and '?' not in sentence:
                # Score salience based on content
                salience = self._compute_salience(sentence)
                
                if salience >= self.config.SALIENCE_THRESHOLD:
                    fact = SalientFact(
                        content=sentence,
                        source="user",
                        timestamp=datetime.now(),
                        salience_score=salience,
                    )
                    
                    # Apply gates
                    gated = apply_experiential_gates(
                        fact, 'salient_fact', self.nurture_state, self.config
                    )
                    if gated is not None:
                        wm.salient_facts.append(gated)
    
    def _compute_salience(self, text: str) -> float:
        """Compute salience score for a piece of text."""
        score = 0.2  # Base salience
        text_lower = text.lower()
        
        # Personal statements are more salient
        if any(word in text_lower for word in ['i am', 'i have', 'i need', 'i want', 'my ']):
            score += 0.2
        
        # Value statements are salient
        value_words = ['important', 'must', 'should', 'always', 'never', 'believe']
        if any(word in text_lower for word in value_words):
            score += 0.2
        
        # Longer statements tend to be more informative
        if len(text) > 50:
            score += 0.1
        
        # Names and specifics are salient
        if re.search(r'\b[A-Z][a-z]+\b', text):  # Proper nouns
            score += 0.1
        
        return min(1.0, score)
    
    def _track_questions(self, user_input: str, assistant_response: str):
        """Track questions and their resolution."""
        wm = self.state.working_memory
        
        # Detect new questions
        if '?' in user_input:
            questions = re.split(r'\?', user_input)
            for q in questions:
                q = q.strip()
                if len(q) > 5:
                    # Check if this question is already tracked
                    existing = [oq for oq in wm.open_questions if q.lower() in oq.question.lower()]
                    if not existing:
                        wm.open_questions.append(OpenQuestion(
                            question=q + "?",
                            context=user_input[:100],
                            asked_at=datetime.now(),
                        ))
        
        # Check for resolved questions
        response_lower = assistant_response.lower()
        for oq in wm.open_questions:
            if not oq.resolved:
                # Simple heuristic: if response addresses key terms
                q_words = set(re.findall(r'\b\w{4,}\b', oq.question.lower()))
                r_words = set(re.findall(r'\b\w{4,}\b', response_lower))
                overlap = len(q_words & r_words) / max(1, len(q_words))
                
                if overlap > 0.5:
                    oq.attempted_answers += 1
                    if overlap > 0.7:
                        oq.resolved = True
    
    def _track_commitments(self, assistant_response: str):
        """Track commitments made in response."""
        wm = self.state.working_memory
        response_lower = assistant_response.lower()
        
        for phrase in COMMITMENT_PHRASES:
            if phrase in response_lower:
                # Extract the commitment
                idx = response_lower.find(phrase)
                end_idx = response_lower.find('.', idx)
                if end_idx == -1:
                    end_idx = min(idx + 100, len(response_lower))
                
                promise = assistant_response[idx:end_idx].strip()
                if len(promise) > 10:
                    commitment = Commitment(
                        promise=promise,
                        context=assistant_response[:100],
                        made_at=datetime.now(),
                    )
                    
                    # Apply gates
                    gated = apply_experiential_gates(
                        commitment, 'commitment', self.nurture_state, self.config
                    )
                    if gated is not None:
                        wm.commitments.append(gated)
                break  # Only track one commitment per response
    
    def _apply_decay(self):
        """Apply decay to all experiential state."""
        wm = self.state.working_memory
        
        # Decay fact salience
        for fact in wm.salient_facts:
            fact.salience_score *= self.config.FACT_DECAY
    
    def _prune_memory(self):
        """Prune low-salience items from working memory."""
        wm = self.state.working_memory
        config = self.config
        
        # Prune low-salience facts
        wm.salient_facts = [
            f for f in wm.salient_facts
            if f.salience_score >= config.MIN_SALIENCE
        ]
        
        # Keep only top N facts
        wm.salient_facts = sorted(
            wm.salient_facts,
            key=lambda f: f.salience_score,
            reverse=True
        )[:config.MAX_SALIENT_FACTS]
        
        # Prune resolved questions after linger
        now = datetime.now()
        wm.open_questions = [
            q for q in wm.open_questions
            if not q.resolved or (now - q.asked_at).seconds < config.RESOLUTION_LINGER_SECONDS
        ][:config.MAX_OPEN_QUESTIONS]
        
        # Prune fulfilled commitments
        wm.commitments = [
            c for c in wm.commitments
            if not c.fulfilled or (now - c.made_at).seconds < config.FULFILLMENT_LINGER_SECONDS
        ][:config.MAX_COMMITMENTS]
    
    def end_session(self) -> Tuple[Any, Optional[Dict]]:
        """
        Process session end.
        
        Returns:
            Tuple of (persistent_traces, nurture_promotion_candidate)
        """
        if self.state is None:
            return None, None
        
        # Update persistent traces
        self._update_persistent_traces()
        
        # Check for nurture promotion
        promotion_candidate = compute_promotion_candidate(
            self.state.persistent_traces,
            self.nurture_state,
            min_sessions=10,
            stability_threshold=0.9,
        )
        
        return self.state.persistent_traces, promotion_candidate
    
    def _update_persistent_traces(self):
        """Update cross-session pattern accumulator."""
        if self.state is None:
            return
        
        pt = self.state.persistent_traces
        cm = self.state.conversation_model
        
        # Summarize session into pattern vector
        session_pattern = np.concatenate([
            cm.topic_vector[:64],  # Truncate to fit
            cm.emotional_trajectory,
            cm.user_state_estimate,
        ])
        
        # Pad or truncate to pattern dimension
        pattern_dim = self.config.D_PATTERN
        if len(session_pattern) < pattern_dim:
            session_pattern = np.pad(session_pattern, (0, pattern_dim - len(session_pattern)))
        else:
            session_pattern = session_pattern[:pattern_dim]
        
        # Accumulate with decay
        decay = self.config.PATTERN_DECAY_CROSS_SESSION
        pt.pattern_accumulator = decay * pt.pattern_accumulator + (1 - decay) * session_pattern
        
        # Update familiarity
        if np.linalg.norm(pt.pattern_accumulator) > 0 and np.linalg.norm(session_pattern) > 0:
            similarity = np.dot(session_pattern, pt.pattern_accumulator) / (
                np.linalg.norm(session_pattern) * np.linalg.norm(pt.pattern_accumulator)
            )
            pt.familiarity_score = 0.8 * pt.familiarity_score + 0.2 * max(0, similarity)
        
        pt.session_count += 1
        pt.last_session_end = datetime.now()
    
    def get_context_for_prompt(self) -> str:
        """Get experiential context string for prompt injection."""
        if self.state is None:
            return ""
        return self.state.get_context_string()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current experiential state."""
        if self.state is None:
            return {}
        
        return {
            'session_id': self.state.session_id,
            'interaction_count': self.state.conversation_model.interaction_count,
            'topic_summary': self.state.conversation_model.topic_summary,
            'emotion_summary': self.state.conversation_model.emotion_summary,
            'user_summary': self.state.conversation_model.user_summary,
            'facts_count': len(self.state.working_memory.salient_facts),
            'open_questions': len([q for q in self.state.working_memory.open_questions if not q.resolved]),
            'active_commitments': len([c for c in self.state.working_memory.commitments if not c.fulfilled]),
            'session_familiarity': self.state.persistent_traces.familiarity_score,
            'total_sessions': self.state.persistent_traces.session_count,
        }
