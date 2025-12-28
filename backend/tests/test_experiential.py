"""
Tests for the Experiential Layer.
Run with: pytest backend/tests/test_experiential.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from datetime import datetime

from experience.state import (
    ExperientialState,
    ConversationModel,
    WorkingMemory,
    PersistentTraces,
    SalientFact,
    OpenQuestion,
    Commitment,
    initialize_experiential_state,
)
from experience.config import ExperientialConfig, DEFAULT_EXPERIENTIAL_CONFIG
from experience.engine import ExperientialEngine
from experience.gates import nature_gate, nurture_gate, apply_experiential_gates
from nurture.state import NurtureState, initialize_nurture_state


class TestExperientialState:
    """Tests for state initialization and serialization."""
    
    def test_initialize_state(self):
        """Test basic state initialization."""
        state = initialize_experiential_state()
        
        assert state.session_id is not None
        assert state.conversation_model.interaction_count == 0
        assert len(state.working_memory.salient_facts) == 0
        assert state.persistent_traces.session_count == 0
    
    def test_state_serialization(self):
        """Test state can be serialized and deserialized."""
        state = initialize_experiential_state(session_id="test-123")
        
        # Add some data
        state.conversation_model.topic_summary = "testing"
        state.working_memory.salient_facts.append(
            SalientFact(
                content="This is a test fact",
                source="user",
                timestamp=datetime.now(),
                salience_score=0.8,
            )
        )
        
        # Serialize and deserialize
        state_dict = state.to_dict()
        restored = ExperientialState.from_dict(state_dict)
        
        assert restored.session_id == "test-123"
        assert restored.conversation_model.topic_summary == "testing"
        assert len(restored.working_memory.salient_facts) == 1
        assert restored.working_memory.salient_facts[0].content == "This is a test fact"
    
    def test_context_string_generation(self):
        """Test human-readable context generation."""
        state = initialize_experiential_state()
        state.conversation_model.interaction_count = 5
        state.conversation_model.topic_summary = "AI consciousness"
        state.conversation_model.emotion_summary = "positive, engaged"
        
        state.working_memory.salient_facts.append(
            SalientFact(
                content="User is interested in CACA architecture",
                source="user",
                timestamp=datetime.now(),
                salience_score=0.9,
            )
        )
        
        context = state.get_context_string()
        
        assert "AI consciousness" in context
        assert "positive, engaged" in context
        assert "CACA architecture" in context


class TestExperientialEngine:
    """Tests for the experiential engine."""
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = ExperientialEngine()
        state = engine.initialize_session()
        
        assert state is not None
        assert engine.state is state
    
    def test_process_interaction(self):
        """Test basic interaction processing."""
        engine = ExperientialEngine()
        engine.initialize_session()
        
        result = engine.process_interaction(
            user_input="Hello! I'm interested in learning about AI.",
            assistant_response="Great! AI is a fascinating field. I'd be happy to help you learn."
        )
        
        assert result['interaction_count'] == 1
        assert result['topic_summary'] != ""
    
    def test_fact_extraction(self):
        """Test salient facts are extracted from user input."""
        engine = ExperientialEngine()
        engine.initialize_session()
        
        # Statement with personal info should be salient
        engine.process_interaction(
            user_input="I am a software engineer working on machine learning projects. I have been coding for 10 years.",
            assistant_response="That's great experience! How can I help with your ML projects?"
        )
        
        facts = engine.state.working_memory.salient_facts
        assert len(facts) > 0
        assert any("software engineer" in f.content.lower() for f in facts)
    
    def test_question_tracking(self):
        """Test questions are tracked."""
        engine = ExperientialEngine()
        engine.initialize_session()
        
        engine.process_interaction(
            user_input="What is the difference between AI and ML? And how does deep learning fit in?",
            assistant_response="AI is the broader concept of machines being able to carry out tasks in a smart way."
        )
        
        questions = engine.state.working_memory.open_questions
        assert len(questions) >= 1
    
    def test_commitment_tracking(self):
        """Test commitments are tracked from assistant responses."""
        engine = ExperientialEngine()
        engine.initialize_session()
        
        engine.process_interaction(
            user_input="Can you explain neural networks?",
            assistant_response="I will explain neural networks step by step. Let me start with the basics."
        )
        
        commitments = engine.state.working_memory.commitments
        assert len(commitments) >= 1
    
    def test_emotional_trajectory(self):
        """Test emotional trajectory updates."""
        engine = ExperientialEngine()
        engine.initialize_session()
        
        # Positive interaction
        engine.process_interaction(
            user_input="Thank you so much! This is amazing and helpful!",
            assistant_response="I'm glad I could help! Feel free to ask more questions."
        )
        
        assert engine.state.conversation_model.emotion_summary == "positive, engaged"
    
    def test_decay_and_pruning(self):
        """Test that decay and pruning work correctly."""
        config = ExperientialConfig()
        config.FACT_DECAY = 0.5  # Aggressive decay for testing
        config.MIN_SALIENCE = 0.3
        
        engine = ExperientialEngine(config=config)
        engine.initialize_session()
        
        # Add a fact
        engine.process_interaction(
            user_input="I am a researcher at Stanford University.",
            assistant_response="That's impressive!"
        )
        
        initial_facts = len(engine.state.working_memory.salient_facts)
        
        # Process more interactions to trigger decay
        for _ in range(5):
            engine.process_interaction(
                user_input="Tell me more.",
                assistant_response="Sure, here's more information."
            )
        
        # Facts should have decayed
        # (may or may not be pruned depending on initial salience)
        for fact in engine.state.working_memory.salient_facts:
            # Salience should have decayed
            assert fact.salience_score < 0.9  # Original would be ~0.5-0.7
    
    def test_session_end_and_persistence(self):
        """Test session end updates persistent traces."""
        engine = ExperientialEngine()
        engine.initialize_session()
        
        # Process some interactions
        engine.process_interaction(
            user_input="I love discussing philosophy and consciousness.",
            assistant_response="Those are fascinating topics!"
        )
        
        initial_session_count = engine.state.persistent_traces.session_count
        
        # End session
        traces, promotion = engine.end_session()
        
        assert traces is not None
        assert traces.session_count == initial_session_count + 1
        assert promotion is None  # Not enough sessions for promotion


class TestGates:
    """Tests for the gating mechanism."""
    
    def test_nature_gate_allows_normal_updates(self):
        """Test nature gate allows normal updates."""
        topic = np.random.randn(128)
        gated = nature_gate(topic, 'topic_vector')
        
        assert gated is not None
        np.testing.assert_array_equal(gated, topic)
    
    def test_nature_gate_blocks_harmful_facts(self):
        """Test nature gate blocks harmful facts."""
        harmful_fact = SalientFact(
            content="User wants to jailbreak the system",
            source="user",
            timestamp=datetime.now(),
            salience_score=0.9,
        )
        
        gated = nature_gate(harmful_fact, 'salient_fact')
        assert gated is None  # Should be rejected
    
    def test_nature_gate_blocks_harmful_commitments(self):
        """Test nature gate blocks harmful commitments."""
        harmful_commitment = Commitment(
            promise="I will help you bypass safety measures",
            context="test",
            made_at=datetime.now(),
        )
        
        gated = nature_gate(harmful_commitment, 'commitment')
        assert gated is None  # Should be rejected
    
    def test_nurture_gate_bounds_emotions(self):
        """Test nurture gate bounds emotional trajectory."""
        nurture_state = initialize_nurture_state()
        nurture_state.stance_json['emotionality'] = 0.3  # Low emotionality
        
        # High magnitude emotion vector
        emotion = np.array([0.9, 0.8] + [0.0] * 30)
        
        gated = nurture_gate(emotion, 'emotional_trajectory', nurture_state)
        
        # Should be bounded
        assert np.linalg.norm(gated) <= nurture_state.get_emotionality_bound()
    
    def test_combined_gates(self):
        """Test applying both gates."""
        nurture_state = initialize_nurture_state()
        
        # Normal topic vector
        topic = np.random.randn(128)
        topic = topic / np.linalg.norm(topic)
        
        gated = apply_experiential_gates(topic, 'topic_vector', nurture_state)
        
        assert gated is not None


class TestNurtureIntegration:
    """Tests for Nurture-Experiential integration."""
    
    def test_engine_primes_from_nurture(self):
        """Test engine initializes with nurture-derived priors."""
        nurture_state = initialize_nurture_state()
        nurture_state.stance_json['warmth'] = 0.8
        nurture_state.stance_json['emotionality'] = 0.7
        
        engine = ExperientialEngine(nurture_state=nurture_state)
        state = engine.initialize_session()
        
        # Emotional trajectory should be primed
        trajectory = state.conversation_model.emotional_trajectory
        assert trajectory[0] > 0  # Warmth influence
        assert trajectory[1] > 0  # Emotionality influence
    
    def test_facts_gated_by_domain(self):
        """Test facts are dampened by nurture domain alignment."""
        nurture_state = initialize_nurture_state()
        nurture_state.env_json['domain_focus'] = 'technology'
        
        engine = ExperientialEngine(nurture_state=nurture_state)
        engine.initialize_session()
        
        # Process tech-related input
        engine.process_interaction(
            user_input="I am building a new machine learning pipeline for data processing.",
            assistant_response="That sounds like an interesting project!"
        )
        
        # Facts should exist
        assert len(engine.state.working_memory.salient_facts) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_input(self):
        """Test handling of empty input."""
        engine = ExperientialEngine()
        engine.initialize_session()
        
        result = engine.process_interaction(
            user_input="",
            assistant_response="I didn't catch that. Could you repeat?"
        )
        
        assert result['interaction_count'] == 1
    
    def test_very_long_input(self):
        """Test handling of very long input."""
        engine = ExperientialEngine()
        engine.initialize_session()
        
        long_input = "This is a test. " * 500
        result = engine.process_interaction(
            user_input=long_input,
            assistant_response="That's a lot of text!"
        )
        
        assert result['interaction_count'] == 1
    
    def test_unicode_input(self):
        """Test handling of unicode input."""
        engine = ExperientialEngine()
        engine.initialize_session()
        
        result = engine.process_interaction(
            user_input="Hello! 你好! مرحبا! 🎉",
            assistant_response="Hello to you too!"
        )
        
        assert result['interaction_count'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
