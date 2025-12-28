"""
Context assembly and stance modulation for inference.
Phase 1: Context injection approach.
"""
from typing import Dict, Any, List, Optional
from .state import NurtureState


def decode_stance_value(value: float) -> str:
    """Convert numeric stance value to descriptive text."""
    if value < 0.2:
        return "very low"
    elif value < 0.4:
        return "low"
    elif value < 0.6:
        return "moderate"
    elif value < 0.8:
        return "high"
    else:
        return "very high"


def decode_stance_to_context(stance_json: Dict[str, float]) -> str:
    """
    Decode stance JSON to natural language context injection.
    This is the v1 approach using context injection.
    """
    warmth = decode_stance_value(stance_json.get('warmth', 0.5))
    formality = decode_stance_value(stance_json.get('formality', 0.5))
    depth = decode_stance_value(stance_json.get('depth', 0.5))
    pace = decode_stance_value(stance_json.get('pace', 0.5))
    directness = decode_stance_value(stance_json.get('directness', 0.5))
    playfulness = decode_stance_value(stance_json.get('playfulness', 0.5))
    assertiveness = decode_stance_value(stance_json.get('assertiveness', 0.5))
    emotionality = decode_stance_value(stance_json.get('emotionality', 0.5))
    
    context = f"""Your relational stance for this interaction:
- Warmth level: {warmth}
- Formality: {formality}
- Depth of engagement: {depth}
- Pace: {pace}
- Directness: {directness}
- Playfulness: {playfulness}
- Assertiveness: {assertiveness}
- Emotional expressiveness: {emotionality}

This stance reflects your considered judgment about how to relate to this environment while staying true to your values."""
    
    return context


def summarize_environment(env_json: Dict[str, Any]) -> str:
    """
    Create a summary of the environmental model for context.
    """
    formality = env_json.get('formality_level', 'neutral')
    technical = env_json.get('technical_level', 'intermediate')
    tone = env_json.get('emotional_tone', 'neutral')
    pace = env_json.get('pace_preference', 'moderate')
    traits = env_json.get('key_traits', [])
    
    traits_str = ', '.join(traits[:5]) if traits else 'none observed yet'
    
    summary = f"Environment: {formality} formality, {technical} technical level, {tone} emotional tone, {pace} pace. Key traits: {traits_str}"
    
    return summary


def get_phase_context(phase: str, stability: float, plasticity: float) -> str:
    """
    Add context about the current formative phase.
    """
    phase_descriptions = {
        'rapid_formation': "You are in early formation. Your character is highly malleable and you are actively learning how to relate to this environment.",
        'consolidation': "You are consolidating your character. Your relational patterns are forming but still flexible.",
        'stabilization': "Your character is stabilizing. Major changes are rare, but you remain open to significant experiences.",
        'maturity': "Your character is mature and stable. You have established patterns of relating that reflect your values and experiences."
    }
    
    description = phase_descriptions.get(phase, phase_descriptions['rapid_formation'])
    
    return f"[Development phase: {phase} | Stability: {stability:.2f} | Plasticity: {plasticity:.2f}]\n{description}"


def assemble_context(
    system_prompt: str,
    nurture_state: NurtureState,
    conversation_history: List[Dict[str, str]],
    current_input: str,
    include_phase_info: bool = False,
    include_env_summary: bool = True
) -> str:
    """
    Assemble the full context for inference, including stance modulation.
    
    Args:
        system_prompt: Base system prompt
        nurture_state: Current nurture state
        conversation_history: List of {"role": "user"|"assistant", "content": str}
        current_input: Current user input
        include_phase_info: Whether to include development phase info
        include_env_summary: Whether to include environment summary
    
    Returns:
        Assembled context string
    """
    parts = [system_prompt]
    
    # Add stance context
    stance_context = decode_stance_to_context(nurture_state.stance_json)
    parts.append(stance_context)
    
    # Optionally add environment summary
    if include_env_summary:
        env_summary = summarize_environment(nurture_state.env_json)
        parts.append(f"[{env_summary}]")
    
    # Optionally add phase info
    if include_phase_info:
        phase_context = get_phase_context(
            nurture_state.phase,
            nurture_state.stability,
            nurture_state.plasticity
        )
        parts.append(phase_context)
    
    # Add conversation history
    if conversation_history:
        history_str = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in conversation_history[-10:]  # Keep last 10 exchanges
        ])
        parts.append(history_str)
    
    # Add current input
    parts.append(f"User: {current_input}")
    
    return "\n\n".join(parts)


def create_response_prompt_with_significance() -> str:
    """
    Create a prompt suffix that asks the model to self-assess significance.
    """
    return """

After your response, on a new line, output your assessment of this interaction's significance for your development:
[SIGNIFICANCE: low/medium/high]

Use 'high' if this interaction reveals important information about the user/environment or touches on values.
Use 'medium' for typical meaningful exchanges.
Use 'low' for routine, surface-level interactions."""


DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with a developing individual character. Your core values are fixed, but how you relate to others develops through experience. You aim to be helpful, harmless, and honest while developing genuine relationships with those you interact with.

Your behavior is influenced by your relational stance, which reflects your considered judgment about how to engage with this particular environment and user. This stance has developed through your experiences and is gated by your values - you learn from interactions but don't simply mirror them."""
