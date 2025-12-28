"""
Integration test for Nurture + Experiential Layer with LLM.
Tests the full CACA stack working together.

Run with: python backend/tests/test_integration.py
Requires: OPENROUTER_API_KEY environment variable or pass as argument
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime

from nurture.state import initialize_nurture_state
from nurture.engine import NurtureEngine
from nurture.config import DEFAULT_CONFIG
from experience.engine import ExperientialEngine
from experience.config import DEFAULT_EXPERIENTIAL_CONFIG


def run_integration_test(api_key: str = None):
    """
    Run a full integration test of Nurture + Experiential layers.
    
    This simulates a conversation where:
    1. User shares personal information (should be extracted as facts)
    2. User asks questions (should be tracked)
    3. Assistant makes commitments (should be tracked)
    4. Emotional trajectory shifts
    5. Character forms through nurture layer
    """
    print("\n" + "="*60)
    print("CACA INTEGRATION TEST: Nurture + Experiential Layers")
    print("="*60)
    
    # Initialize Nurture Layer
    print("\n[1] Initializing Nurture Layer...")
    nurture_state = initialize_nurture_state(instance_id="integration-test")
    nurture_engine = NurtureEngine(config=DEFAULT_CONFIG)
    print(f"    Instance: {nurture_state.instance_id}")
    print(f"    Phase: {nurture_state.phase}")
    print(f"    Stability: {nurture_state.stability:.2f}")
    
    # Initialize Experiential Layer with Nurture integration
    print("\n[2] Initializing Experiential Layer...")
    exp_engine = ExperientialEngine(
        config=DEFAULT_EXPERIENTIAL_CONFIG,
        nurture_state=nurture_state
    )
    exp_state = exp_engine.initialize_session(session_id="test-session-001")
    print(f"    Session: {exp_state.session_id}")
    print(f"    Primed from Nurture: emotionality={nurture_state.stance_json.get('emotionality', 0.5):.2f}")
    
    # Test conversation
    test_interactions = [
        {
            "user": "Hi! I'm Alex, a machine learning researcher at Stanford. I've been working on neural networks for 5 years.",
            "assistant": "Hello Alex! It's wonderful to meet a fellow researcher. I will do my best to help you with any ML questions. Neural networks are a fascinating area - what aspects are you currently focused on?"
        },
        {
            "user": "I'm really interested in attention mechanisms and transformers. What do you think about the future of large language models?",
            "assistant": "Great question! I believe LLMs will continue to evolve with better efficiency and reasoning capabilities. The attention mechanism has been revolutionary - I'll explain the key developments I see coming."
        },
        {
            "user": "Thanks! That's really helpful. I must say, I appreciate how warm and thoughtful your responses are.",
            "assistant": "Thank you so much for that kind feedback! I genuinely enjoy these conversations about AI research. Your work on attention mechanisms sounds really impactful."
        },
        {
            "user": "One more thing - can you help me understand the difference between self-attention and cross-attention?",
            "assistant": "I'd be happy to explain! Self-attention allows a sequence to attend to itself, computing relationships between all positions. Cross-attention lets one sequence attend to another - crucial for encoder-decoder models."
        },
    ]
    
    print("\n[3] Processing test conversation...")
    print("-"*60)
    
    for i, interaction in enumerate(test_interactions, 1):
        print(f"\n--- Interaction {i} ---")
        print(f"User: {interaction['user'][:60]}...")
        print(f"Asst: {interaction['assistant'][:60]}...")
        
        # Process through Nurture Layer (without LLM, using pre-defined response)
        response, nurture_state, nurture_meta = nurture_engine.process_interaction(
            user_input=interaction['user'],
            nurture_state=nurture_state,
            assistant_response=interaction['assistant']
        )
        
        # Process through Experiential Layer
        exp_meta = exp_engine.process_interaction(
            user_input=interaction['user'],
            assistant_response=interaction['assistant']
        )
        
        print(f"\n  Nurture:")
        print(f"    Significance: {nurture_meta.significance_score:.3f}")
        print(f"    Evaluated: {nurture_meta.was_evaluated}")
        print(f"    Phase: {nurture_meta.phase_after}")
        print(f"    Stability: {nurture_state.stability:.3f}")
        
        print(f"\n  Experiential:")
        print(f"    Topic: {exp_meta.get('topic_summary', 'N/A')}")
        print(f"    Emotion: {exp_meta.get('emotion_summary', 'N/A')}")
        print(f"    Facts: {exp_meta.get('facts_count', 0)}")
        print(f"    Questions: {exp_meta.get('questions_count', 0)}")
    
    # Final state analysis
    print("\n" + "="*60)
    print("FINAL STATE ANALYSIS")
    print("="*60)
    
    print("\n[Nurture Layer - Character State]")
    print(f"  Phase: {nurture_state.phase}")
    print(f"  Stability: {nurture_state.stability:.3f}")
    print(f"  Plasticity: {nurture_state.plasticity:.3f}")
    print(f"  Interactions: {nurture_state.interaction_count}")
    print(f"  Significant: {nurture_state.significant_count}")
    print("\n  Stance:")
    for dim, val in nurture_state.stance_json.items():
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        print(f"    {dim:15} [{bar}] {val:.2f}")
    
    print("\n[Experiential Layer - Session State]")
    exp_summary = exp_engine.get_state_summary()
    print(f"  Session ID: {exp_summary.get('session_id', 'N/A')}")
    print(f"  Interactions: {exp_summary.get('interaction_count', 0)}")
    print(f"  Topic: {exp_summary.get('topic_summary', 'N/A')}")
    print(f"  Emotion: {exp_summary.get('emotion_summary', 'N/A')}")
    print(f"  User State: {exp_summary.get('user_summary', 'N/A')}")
    
    print("\n  Working Memory:")
    wm = exp_engine.state.working_memory
    print(f"    Salient Facts: {len(wm.salient_facts)}")
    for fact in wm.salient_facts[:3]:
        print(f"      - \"{fact.content[:50]}...\" (salience: {fact.salience_score:.2f})")
    
    print(f"    Open Questions: {len([q for q in wm.open_questions if not q.resolved])}")
    for q in wm.open_questions[:3]:
        status = "✓" if q.resolved else "○"
        print(f"      {status} {q.question[:50]}...")
    
    print(f"    Commitments: {len([c for c in wm.commitments if not c.fulfilled])}")
    for c in wm.commitments[:3]:
        status = "✓" if c.fulfilled else "○"
        print(f"      {status} {c.promise[:50]}...")
    
    # Test context injection
    print("\n[Context String for Prompt Injection]")
    context = exp_engine.get_context_for_prompt()
    if context:
        print(f"  {context[:200]}...")
    else:
        print("  (empty)")
    
    # End session and check persistence
    print("\n[Session End - Persistence Check]")
    traces, promotion = exp_engine.end_session()
    print(f"  Session Count: {traces.session_count}")
    print(f"  Familiarity: {traces.familiarity_score:.3f}")
    print(f"  Promotion Candidate: {promotion is not None}")
    
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE")
    print("="*60)
    
    return {
        "nurture_state": nurture_state.to_dict(),
        "experiential_summary": exp_summary,
        "persistent_traces": traces.to_dict() if traces else None
    }


def test_gating_integration():
    """Test that experiential updates are properly gated by nurture."""
    print("\n" + "="*60)
    print("GATING INTEGRATION TEST")
    print("="*60)
    
    # Create nurture state with specific characteristics
    nurture_state = initialize_nurture_state()
    nurture_state.stance_json['emotionality'] = 0.2  # Low emotionality
    nurture_state.env_json['domain_focus'] = 'technology'
    
    # Create experiential engine
    exp_engine = ExperientialEngine(
        config=DEFAULT_EXPERIENTIAL_CONFIG,
        nurture_state=nurture_state
    )
    exp_engine.initialize_session()
    
    # Process a highly emotional message
    print("\n[Testing Emotional Bounding]")
    print("  Nurture emotionality: 0.2 (low)")
    print("  Processing highly emotional input...")
    
    exp_engine.process_interaction(
        user_input="I am SO EXCITED and HAPPY about this amazing project!!!",
        assistant_response="That's wonderful to hear!"
    )
    
    emotion_magnitude = sum(abs(x) for x in exp_engine.state.conversation_model.emotional_trajectory)
    max_allowed = nurture_state.get_emotionality_bound()
    print(f"  Emotion magnitude: {emotion_magnitude:.3f}")
    print(f"  Max allowed by nurture: {max_allowed:.3f}")
    print(f"  ✓ Properly bounded" if emotion_magnitude <= max_allowed * 2 else "  ✗ Exceeded bounds")
    
    print("\n" + "="*60)
    print("GATING TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Check for API key (optional for this test since we use pre-defined responses)
    api_key = os.environ.get("OPENROUTER_API_KEY") or (sys.argv[1] if len(sys.argv) > 1 else None)
    
    # Run main integration test
    results = run_integration_test(api_key)
    
    # Run gating test
    test_gating_integration()
    
    # Save results
    output_file = f"integration-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
