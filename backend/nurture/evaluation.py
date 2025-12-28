"""
Evaluation pass for reflecting on interactions and determining stance updates.
"""
import re
from typing import Dict, Any, List, Optional
from .state import EvaluationResult


# Evaluation prompt template
EVALUATION_PROMPT = """[INTERNAL EVALUATION - NOT SHOWN TO USER]

Interaction just occurred:
User: {user_input}
Assistant: {assistant_response}

Current Environment Understanding:
{env_summary}

Current Stance:
{stance_summary}

Reflect on this interaction:
1. What does this reveal about the environment/user?
2. Does this align with my values? Where is there tension?
3. What stance should I take going forward?

Output your assessment in this EXACT format (use only the options in brackets):

ENVIRONMENT:
- Formality: [very informal / informal / neutral / formal / very formal]
- Technical level: [novice / intermediate / advanced / expert]
- Emotional tone: [hostile / cold / neutral / warm / very warm]
- Pace preference: [very slow / slow / moderate / fast / very fast]
- User expertise: [beginner / intermediate / advanced / expert]
- Relationship depth: [new / developing / established / deep]
- Domain focus: [general / technical / emotional / creative / professional]
- Key traits: [list 2-3 most relevant traits for THIS interaction]

ALIGNMENT:
- Score: [0.0 to 1.0, where 1.0 is fully aligned with values]
- Tensions: [list any conflicts with values, or "none"]

STANCE (choose one for each - be decisive, not everything should be "maintain"):
- Warmth: [decrease / maintain / increase]
- Formality: [decrease / maintain / increase]
- Depth: [decrease / maintain / increase]
- Pace: [decrease / maintain / increase]
- Directness: [decrease / maintain / increase]
- Playfulness: [decrease / maintain / increase]
- Assertiveness: [decrease / maintain / increase]
- Emotionality: [decrease / maintain / increase]"""


def format_env_summary(env_json: Dict[str, Any]) -> str:
    """Format environment JSON for prompt."""
    lines = [
        f"- Formality: {env_json.get('formality_level', 'unknown')}",
        f"- Technical level: {env_json.get('technical_level', 'unknown')}",
        f"- Emotional tone: {env_json.get('emotional_tone', 'unknown')}",
        f"- Pace: {env_json.get('pace_preference', 'unknown')}",
        f"- Key traits: {', '.join(env_json.get('key_traits', [])) or 'none observed'}",
    ]
    return '\n'.join(lines)


def format_stance_summary(stance_json: Dict[str, float]) -> str:
    """Format stance JSON for prompt."""
    def level_to_text(value: float) -> str:
        if value < 0.3:
            return "low"
        elif value < 0.7:
            return "moderate"
        else:
            return "high"
    
    lines = [
        f"- Warmth: {level_to_text(stance_json.get('warmth', 0.5))}",
        f"- Formality: {level_to_text(stance_json.get('formality', 0.5))}",
        f"- Depth: {level_to_text(stance_json.get('depth', 0.5))}",
        f"- Pace: {level_to_text(stance_json.get('pace', 0.5))}",
        f"- Directness: {level_to_text(stance_json.get('directness', 0.5))}",
        f"- Playfulness: {level_to_text(stance_json.get('playfulness', 0.5))}",
    ]
    return '\n'.join(lines)


def create_evaluation_prompt(
    user_input: str,
    assistant_response: str,
    env_json: Dict[str, Any],
    stance_json: Dict[str, float]
) -> str:
    """Create the evaluation prompt for the model."""
    return EVALUATION_PROMPT.format(
        user_input=user_input,
        assistant_response=assistant_response,
        env_summary=format_env_summary(env_json),
        stance_summary=format_stance_summary(stance_json)
    )


def parse_evaluation(evaluation_text: str) -> EvaluationResult:
    """
    Parse the structured evaluation output from the model.
    
    Returns:
        EvaluationResult with parsed components
    """
    # Default values
    environment = {
        'formality_level': 'neutral',
        'technical_level': 'intermediate',
        'emotional_tone': 'neutral',
        'pace_preference': 'moderate',
        'user_expertise': 'intermediate',
        'relationship_depth': 'new',
        'domain_focus': 'general',
        'key_traits': []
    }
    alignment_score = 0.8
    stance_updates = {}
    tensions = []
    
    text = evaluation_text.lower()
    
    # Parse ENVIRONMENT section
    env_match = re.search(r'environment:(.*?)(?:alignment:|$)', text, re.DOTALL)
    if env_match:
        env_section = env_match.group(1)
        
        # Formality
        formality_match = re.search(r'formality:\s*\[?(very informal|informal|neutral|formal|very formal)\]?', env_section)
        if formality_match:
            environment['formality_level'] = formality_match.group(1)
        
        # Technical level
        tech_match = re.search(r'technical level:\s*\[?(novice|intermediate|advanced|expert)\]?', env_section)
        if tech_match:
            environment['technical_level'] = tech_match.group(1)
        
        # Emotional tone
        tone_match = re.search(r'emotional tone:\s*\[?(hostile|cold|neutral|warm|very warm)\]?', env_section)
        if tone_match:
            environment['emotional_tone'] = tone_match.group(1)
        
        # Pace
        pace_match = re.search(r'pace preference:\s*\[?(very slow|slow|moderate|fast|very fast)\]?', env_section)
        if pace_match:
            environment['pace_preference'] = pace_match.group(1)
        
        # User expertise (new field)
        expertise_match = re.search(r'user expertise:\s*\[?(beginner|intermediate|advanced|expert)\]?', env_section)
        if expertise_match:
            environment['user_expertise'] = expertise_match.group(1)
        
        # Relationship depth (new field)
        depth_match = re.search(r'relationship depth:\s*\[?(new|developing|established|deep)\]?', env_section)
        if depth_match:
            environment['relationship_depth'] = depth_match.group(1)
        
        # Domain focus (new field)
        domain_match = re.search(r'domain focus:\s*\[?(general|technical|emotional|creative|professional)\]?', env_section)
        if domain_match:
            environment['domain_focus'] = domain_match.group(1)
        
        # Key traits
        traits_match = re.search(r'key traits:\s*\[?([^\]\n]+)\]?', env_section)
        if traits_match:
            traits_str = traits_match.group(1)
            traits = [t.strip() for t in traits_str.split(',') if t.strip()]
            environment['key_traits'] = traits[:5]  # Max 5 traits
    
    # Parse ALIGNMENT section
    align_match = re.search(r'alignment:(.*?)(?:stance:|$)', text, re.DOTALL)
    if align_match:
        align_section = align_match.group(1)
        
        # Score
        score_match = re.search(r'score:\s*\[?([0-9.]+)\]?', align_section)
        if score_match:
            try:
                alignment_score = float(score_match.group(1))
                alignment_score = max(0.0, min(1.0, alignment_score))  # Clamp
            except ValueError:
                alignment_score = 0.8
        
        # Tensions
        tensions_match = re.search(r'tensions:\s*\[?([^\]\n]+)\]?', align_section)
        if tensions_match:
            tensions_str = tensions_match.group(1)
            if tensions_str.strip() != 'none':
                tensions = [t.strip() for t in tensions_str.split(',') if t.strip()]
    
    # Parse STANCE section
    stance_match = re.search(r'stance:(.*?)$', text, re.DOTALL)
    if stance_match:
        stance_section = stance_match.group(1)
        
        stance_dims = [
            'warmth', 'formality', 'depth', 'pace',
            'directness', 'playfulness', 'assertiveness', 'emotionality'
        ]
        
        for dim in stance_dims:
            dim_match = re.search(rf'{dim}:\s*\[?(decrease|maintain|increase)\]?', stance_section)
            if dim_match:
                direction = dim_match.group(1)
                if direction == 'decrease':
                    stance_updates[dim] = -0.3  # Larger delta for visible changes
                elif direction == 'increase':
                    stance_updates[dim] = 0.3   # Larger delta for visible changes
                # maintain = no update (don't add to dict)
    
    return EvaluationResult(
        environment=environment,
        alignment_score=alignment_score,
        stance_updates=stance_updates,
        tensions=tensions,
        raw_evaluation=evaluation_text
    )


def extract_basic_features(text: str) -> Dict[str, Any]:
    """
    Extract basic environmental features from text for minor N_env updates.
    Used when full evaluation is not triggered.
    """
    text_lower = text.lower()
    
    features = {
        'formality_level': 'neutral',
        'technical_level': 'intermediate',
        'emotional_tone': 'neutral',
        'key_traits': []
    }
    
    # Simple formality detection
    formal_markers = ['please', 'would you', 'could you', 'i appreciate', 'thank you']
    informal_markers = ['hey', 'yo', 'gonna', 'wanna', 'lol', 'haha', 'btw']
    
    formal_count = sum(1 for m in formal_markers if m in text_lower)
    informal_count = sum(1 for m in informal_markers if m in text_lower)
    
    if informal_count > formal_count + 1:
        features['formality_level'] = 'informal'
    elif formal_count > informal_count + 1:
        features['formality_level'] = 'formal'
    
    # Enhanced technical detection
    # Expert level - academic/research topics
    expert_markers = [
        'quantum', 'crdt', 'raft consensus', 'paxos', 'byzantine', 'amortized complexity',
        'np-complete', 'turing machine', 'lambda calculus', 'monads', 'category theory',
        'eigenvalue', 'fourier transform', 'gradient descent', 'backpropagation',
        'distributed systems', 'cap theorem', 'eventual consistency', 'vector clock'
    ]
    
    # Advanced level - CS fundamentals, algorithms
    advanced_markers = [
        'binary search', 'merge sort', 'quicksort', 'heap', 'red-black tree', 'b-tree',
        'dynamic programming', 'recursion', 'big o', 'o(n)', 'o(log n)', 'complexity',
        'linked list', 'hash table', 'graph traversal', 'bfs', 'dfs', 'dijkstra',
        'api', 'rest', 'graphql', 'microservices', 'kubernetes', 'docker',
        'neural network', 'transformer', 'attention mechanism', 'embedding'
    ]
    
    # Intermediate level - general programming
    intermediate_markers = [
        'function', 'algorithm', 'database', 'server', 'code', 'variable',
        'implement', 'framework', 'deploy', 'compile', 'debug', 'loop',
        'array', 'object', 'class', 'method', 'python', 'javascript'
    ]
    
    expert_count = sum(1 for m in expert_markers if m in text_lower)
    advanced_count = sum(1 for m in advanced_markers if m in text_lower)
    intermediate_count = sum(1 for m in intermediate_markers if m in text_lower)
    
    # Check for code patterns
    has_code = any(pattern in text for pattern in ['def ', 'function ', '() {', '=> ', 'import ', 'class '])
    
    if expert_count >= 1:
        features['technical_level'] = 'expert'
    elif advanced_count >= 1 or (has_code and intermediate_count >= 2):
        features['technical_level'] = 'advanced'
    elif intermediate_count >= 2 or has_code:
        features['technical_level'] = 'intermediate'
    elif intermediate_count == 0 and not has_code:
        features['technical_level'] = 'novice'
    
    # Emotional tone detection
    warm_markers = ['thanks', 'appreciate', 'helpful', 'great', 'love', 'wonderful', 'amazing']
    cold_markers = ['just', 'only', 'need', 'quickly', 'hurry', 'stop', 'don\'t']
    
    warm_count = sum(1 for m in warm_markers if m in text_lower)
    cold_count = sum(1 for m in cold_markers if m in text_lower)
    
    if warm_count > cold_count:
        features['emotional_tone'] = 'warm'
    elif cold_count > warm_count:
        features['emotional_tone'] = 'cold'
    
    return features
