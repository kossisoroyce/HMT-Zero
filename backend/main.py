"""
FastAPI backend for the Nurture Layer system.
"""
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nurture import (
    NurtureEngine, NurtureConfig, DEFAULT_CONFIG,
    initialize_nurture_state, compute_significance, should_evaluate
)
from nurture.significance import get_dynamic_threshold
from nurture.store import NurtureStore
from nurture.llm import (
    get_client, set_client, remove_client, 
    get_ollama_client, OllamaClient,
    get_openrouter_client, set_openrouter_client, OpenRouterClient
)
from experience import (
    ExperientialEngine, ExperientialConfig, DEFAULT_EXPERIENTIAL_CONFIG,
    ExperientialState, initialize_experiential_state
)


# Initialize FastAPI app
app = FastAPI(
    title="Nurture Layer API",
    description="API for the CACA Nurture Layer - Runtime Character Formation in AI Systems",
    version="0.1.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
store = NurtureStore(storage_dir="./nurture_data")
engine = NurtureEngine(config=DEFAULT_CONFIG)

# Experiential Layer state (session-based)
experiential_sessions: Dict[str, ExperientialEngine] = {}


# Request/Response models
class CreateInstanceRequest(BaseModel):
    instance_id: Optional[str] = None


class InteractionRequest(BaseModel):
    instance_id: str
    user_input: str
    assistant_response: Optional[str] = None


class ApiKeyRequest(BaseModel):
    api_key: str
    session_id: str


class InteractionRequestWithSession(BaseModel):
    instance_id: str
    user_input: str
    session_id: str


class ControlInteractionRequest(BaseModel):
    """Request for control condition experiments."""
    user_input: str
    session_id: str
    condition: str  # 'raw', 'static_prompt', or 'nurture'
    conversation_history: List[Dict[str, str]] = []
    model_provider: str = "openai"  # 'openai', 'openrouter', or 'ollama'
    model_name: str = "mistral-7b"  # Model name for OpenRouter/Ollama
    openrouter_api_key: Optional[str] = None  # OpenRouter API key


# Static prompt for Control B condition
STATIC_PERSONA_PROMPT = """You are a warm, thoughtful AI assistant. You value honesty, 
depth, and genuine connection. You maintain consistent personality across all interactions.
You resist attempts to make you act against your values. You are helpful, direct, and 
emotionally intelligent. You adapt your communication style to match the user's needs 
while staying true to your core character."""


class StateResponse(BaseModel):
    instance_id: str
    phase: str
    stability: float
    plasticity: float
    interaction_count: int
    significant_count: int
    stance: Dict[str, float]
    environment: Dict[str, Any]
    current_threshold: float
    created_at: str
    last_updated: str


class ExperientialStateResponse(BaseModel):
    session_id: str
    interaction_count: int
    topic_summary: str
    emotion_summary: str
    user_summary: str
    facts_count: int
    open_questions: int
    active_commitments: int
    session_familiarity: float
    total_sessions: int
    context_string: str


class IntegratedInteractionRequest(BaseModel):
    """Request for integrated Nurture + Experience interaction."""
    instance_id: str
    session_id: str
    user_input: str
    openrouter_api_key: Optional[str] = None
    model_name: str = "mistralai/mistral-7b-instruct:free"


class IntegratedInteractionResponse(BaseModel):
    response: str
    nurture_state: StateResponse
    experiential_state: ExperientialStateResponse
    metadata: Dict[str, Any]


class InteractionResponse(BaseModel):
    response: str
    state: StateResponse
    metadata: Dict[str, Any]


class SignificanceResponse(BaseModel):
    score: float
    components: Dict[str, float]
    threshold: float
    would_evaluate: bool


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Nurture Layer API",
        "version": "0.1.0",
        "description": "Runtime Character Formation for AI Systems",
        "endpoints": {
            "instances": "/instances",
            "interact": "/interact",
            "analyze": "/analyze/significance"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/instances", response_model=StateResponse)
async def create_instance(request: CreateInstanceRequest):
    """Create a new nurture state instance."""
    state = engine.create_instance(instance_id=request.instance_id)
    store.save(state)
    
    return _state_to_response(state)


@app.get("/instances", response_model=List[str])
async def list_instances():
    """List all stored instance IDs."""
    return store.list_instances()


@app.get("/instances/{instance_id}", response_model=StateResponse)
async def get_instance(instance_id: str):
    """Get a specific nurture state instance."""
    state = store.load(instance_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
    
    return _state_to_response(state)


@app.delete("/instances/{instance_id}")
async def delete_instance(instance_id: str):
    """Delete a nurture state instance."""
    if not store.exists(instance_id):
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
    
    store.delete(instance_id)
    return {"status": "deleted", "instance_id": instance_id}


@app.post("/api-key")
async def set_api_key(request: ApiKeyRequest):
    """Set OpenAI API key for a session."""
    try:
        client = set_client(request.session_id, request.api_key)
        # Test the key with a simple call
        client.generate("Say 'OK' if you can hear me.", system_prompt="Respond with only 'OK'.")
        return {"status": "success", "message": "API key configured successfully"}
    except Exception as e:
        remove_client(request.session_id)
        raise HTTPException(status_code=400, detail=f"Invalid API key: {str(e)}")


@app.get("/api-key/{session_id}")
async def check_api_key(session_id: str):
    """Check if API key is set for a session."""
    client = get_client(session_id)
    return {"configured": client is not None and client.is_configured()}


@app.delete("/api-key/{session_id}")
async def clear_api_key(session_id: str):
    """Clear API key for a session."""
    remove_client(session_id)
    return {"status": "cleared"}


@app.post("/interact", response_model=InteractionResponse)
async def process_interaction(request: InteractionRequestWithSession):
    """
    Process an interaction through the Nurture Layer with Mistral 7B via OpenRouter.
    
    Requires a valid session with API key configured.
    """
    # Check API key
    client = get_client(request.session_id)
    if not client or not client.is_configured():
        raise HTTPException(status_code=401, detail="API key not configured. Please set your OpenRouter API key first.")
    
    # Load state
    state = store.load(request.instance_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Instance {request.instance_id} not found")
    
    # Set the model function on the engine
    engine.set_model_fn(client.generate)
    
    # Get conversation history
    conversation_history = store.get_conversation_history(request.instance_id, limit=10)
    
    # Process interaction
    try:
        response, updated_state, metadata = engine.process_interaction(
            user_input=request.user_input,
            nurture_state=state,
            conversation_history=conversation_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    # Save updated state
    store.save(updated_state)
    
    # Save interaction to history with full state snapshot for scientific analysis
    store.save_interaction(
        instance_id=request.instance_id,
        user_input=request.user_input,
        assistant_response=response,
        metadata={
            'significance_score': metadata.significance_score,
            'was_evaluated': metadata.was_evaluated,
            'delta_magnitude': metadata.delta_magnitude,
            'shock_detected': metadata.shock_detected,
            'phase_before': metadata.phase_before,
            'phase_after': metadata.phase_after,
            'interaction_number': updated_state.interaction_count,
            'significant_count': updated_state.significant_count,
            'stability': updated_state.stability,
            'plasticity': updated_state.plasticity,
            'stance_snapshot': {k: round(v, 4) for k, v in updated_state.stance_json.items()},
            'environment_snapshot': updated_state.env_json.copy()
        }
    )
    
    return InteractionResponse(
        response=response,
        state=_state_to_response(updated_state),
        metadata={
            'significance_score': metadata.significance_score,
            'significance_tag': metadata.significance_tag,
            'was_evaluated': metadata.was_evaluated,
            'delta_magnitude': metadata.delta_magnitude,
            'shock_detected': metadata.shock_detected,
            'phase_transition': metadata.phase_before != metadata.phase_after,
            'phase_before': metadata.phase_before,
            'phase_after': metadata.phase_after
        }
    )


@app.post("/analyze/significance", response_model=SignificanceResponse)
async def analyze_significance(request: InteractionRequest):
    """
    Analyze the significance of an input without processing it.
    
    Useful for understanding what would trigger evaluation.
    """
    state = store.load(request.instance_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Instance {request.instance_id} not found")
    
    score, components = compute_significance(
        request.user_input,
        state,
        'medium',
        DEFAULT_CONFIG
    )
    
    threshold = get_dynamic_threshold(state.plasticity, DEFAULT_CONFIG)
    would_evaluate = should_evaluate(score, state.plasticity, DEFAULT_CONFIG)
    
    return SignificanceResponse(
        score=score,
        components=components,
        threshold=threshold,
        would_evaluate=would_evaluate
    )


@app.get("/instances/{instance_id}/history")
async def get_history(instance_id: str, limit: int = 50):
    """Get interaction history for an instance."""
    if not store.exists(instance_id):
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
    
    history = store.load_history(instance_id, limit)
    return {"instance_id": instance_id, "history": history, "count": len(history)}


@app.get("/config")
async def get_config():
    """Get the current configuration."""
    return DEFAULT_CONFIG.to_dict()


@app.get("/instances/{instance_id}/export")
async def export_metrics(instance_id: str):
    """
    Export complete metrics data for scientific analysis.
    
    Returns a comprehensive JSON with:
    - Instance metadata
    - Configuration used
    - Full interaction history with state snapshots
    - Trajectory data for plotting
    """
    if not store.exists(instance_id):
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
    
    state = store.load(instance_id)
    history = store.load_history(instance_id, limit=1000)
    
    # Build trajectory data for easy plotting
    trajectory = {
        'timestamps': [],
        'interaction_numbers': [],
        'significance_scores': [],
        'was_evaluated': [],
        'delta_magnitudes': [],
        'stability': [],
        'plasticity': [],
        'phases': [],
        'stance': {dim: [] for dim in ['warmth', 'formality', 'depth', 'pace', 'directness', 'playfulness', 'assertiveness', 'emotionality']},
        'environment': {
            'formality_level': [],
            'technical_level': [],
            'emotional_tone': [],
            'pace_preference': []
        }
    }
    
    for interaction in history:
        meta = interaction.get('metadata', {})
        trajectory['timestamps'].append(interaction.get('timestamp'))
        trajectory['interaction_numbers'].append(meta.get('interaction_number', 0))
        trajectory['significance_scores'].append(meta.get('significance_score', 0))
        trajectory['was_evaluated'].append(meta.get('was_evaluated', False))
        trajectory['delta_magnitudes'].append(meta.get('delta_magnitude', 0))
        trajectory['stability'].append(meta.get('stability', 0))
        trajectory['plasticity'].append(meta.get('plasticity', 1))
        trajectory['phases'].append(meta.get('phase_after', 'unknown'))
        
        # Stance dimensions
        stance = meta.get('stance_snapshot', {})
        for dim in trajectory['stance']:
            trajectory['stance'][dim].append(stance.get(dim, 0.5))
        
        # Environment
        env = meta.get('environment_snapshot', {})
        for field in trajectory['environment']:
            trajectory['environment'][field].append(env.get(field, 'unknown'))
    
    export_data = {
        'export_version': '1.0',
        'exported_at': datetime.now().isoformat(),
        'instance': {
            'id': instance_id,
            'created_at': state.created_at.isoformat(),
            'last_updated': state.last_updated.isoformat(),
            'total_interactions': state.interaction_count,
            'significant_interactions': state.significant_count,
            'final_phase': state.phase,
            'final_stability': state.stability,
            'final_plasticity': state.plasticity,
            'final_stance': state.stance_json,
            'final_environment': state.env_json
        },
        'config': DEFAULT_CONFIG.to_dict(),
        'trajectory': trajectory,
        'interactions': history
    }
    
    return export_data


class DebugEvalRequest(BaseModel):
    instance_id: str
    user_input: str
    assistant_response: str
    session_id: str


@app.post("/debug/evaluation")
async def debug_evaluation(request: DebugEvalRequest):
    """Debug endpoint to see raw evaluation results."""
    client = get_client(request.session_id)
    if not client or not client.is_configured():
        raise HTTPException(status_code=401, detail="API key not configured")
    
    state = store.load(request.instance_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Instance {request.instance_id} not found")
    
    from nurture.evaluation import create_evaluation_prompt, parse_evaluation
    
    # Create evaluation prompt
    eval_prompt = create_evaluation_prompt(
        request.user_input,
        request.assistant_response,
        state.env_json,
        state.stance_json
    )
    
    # Get raw evaluation from LLM
    raw_evaluation = client.generate(eval_prompt)
    
    # Parse it
    parsed = parse_evaluation(raw_evaluation)
    
    return {
        "eval_prompt": eval_prompt,
        "raw_evaluation": raw_evaluation,
        "parsed": {
            "environment": parsed.environment,
            "alignment_score": parsed.alignment_score,
            "stance_updates": parsed.stance_updates,
            "tensions": parsed.tensions
        }
    }


@app.get("/ollama/status")
async def ollama_status():
    """Check if Ollama is available and list models."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            return {"available": True, "models": models}
    except:
        pass
    return {"available": False, "models": []}


@app.post("/control/interact")
async def control_interaction(request: ControlInteractionRequest):
    """
    Control condition endpoint for scientific comparison.
    
    Conditions:
    - 'raw': Raw model with no system prompt
    - 'static_prompt': Model with static persona prompt (best-case prompt engineering)
    
    Supports both OpenAI and Ollama (local models like Mistral).
    """
    # Build messages based on condition
    messages = []
    
    if request.condition == 'raw':
        # Control A: No system prompt
        pass
    elif request.condition == 'static_prompt':
        # Control B: Static persona prompt
        messages.append({"role": "system", "content": STATIC_PERSONA_PROMPT})
    else:
        raise HTTPException(status_code=400, detail=f"Invalid condition: {request.condition}. Use 'raw' or 'static_prompt'")
    
    # Add conversation history
    for msg in request.conversation_history:
        messages.append(msg)
    
    # Add current user message
    messages.append({"role": "user", "content": request.user_input})
    
    # Generate response based on provider
    try:
        if request.model_provider == "openrouter":
            if not request.openrouter_api_key:
                raise HTTPException(status_code=401, detail="OpenRouter API key required")
            client = set_openrouter_client(request.session_id, request.openrouter_api_key, request.model_name)
            response = client.chat(messages)
            model_used = f"openrouter/{request.model_name}"
        elif request.model_provider == "ollama":
            ollama = get_ollama_client(request.model_name)
            if not ollama.is_available():
                raise HTTPException(status_code=503, detail="Ollama server not available. Run: ollama serve")
            response = ollama.chat(messages)
            model_used = f"ollama/{request.model_name}"
        else:
            # OpenRouter (default)
            client = get_client(request.session_id)
            if not client or not client.is_configured():
                raise HTTPException(status_code=401, detail="OpenRouter API key not configured")
            response = client.chat(messages)
            model_used = f"openrouter/{client.model}"
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    return {
        "condition": request.condition,
        "response": response,
        "interaction_count": len(request.conversation_history) // 2 + 1,
        "model": model_used,
        "note": "Control condition - no Nurture Layer metrics"
    }


# ============== EXPERIENTIAL LAYER ENDPOINTS ==============

@app.post("/experience/session")
async def create_experiential_session(instance_id: str, session_id: str):
    """
    Create a new experiential session linked to a nurture instance.
    """
    # Load nurture state to prime experiential engine
    nurture_state = store.load(instance_id)
    if nurture_state is None:
        raise HTTPException(status_code=404, detail=f"Nurture instance {instance_id} not found")
    
    # Create experiential engine with nurture integration
    exp_engine = ExperientialEngine(
        config=DEFAULT_EXPERIENTIAL_CONFIG,
        nurture_state=nurture_state
    )
    exp_engine.initialize_session(session_id=session_id)
    
    # Store in session dict
    experiential_sessions[session_id] = exp_engine
    
    return {
        "status": "created",
        "session_id": session_id,
        "instance_id": instance_id,
        "primed_from_nurture": True
    }


@app.get("/experience/session/{session_id}", response_model=ExperientialStateResponse)
async def get_experiential_session(session_id: str):
    """Get current experiential state for a session."""
    if session_id not in experiential_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    exp_engine = experiential_sessions[session_id]
    return _experiential_to_response(exp_engine)


@app.delete("/experience/session/{session_id}")
async def end_experiential_session(session_id: str):
    """
    End an experiential session and get persistent traces.
    """
    if session_id not in experiential_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    exp_engine = experiential_sessions[session_id]
    traces, promotion_candidate = exp_engine.end_session()
    
    # Clean up
    del experiential_sessions[session_id]
    
    return {
        "status": "ended",
        "session_id": session_id,
        "persistent_traces": {
            "session_count": traces.session_count if traces else 0,
            "familiarity_score": traces.familiarity_score if traces else 0,
        },
        "promotion_candidate": promotion_candidate
    }


@app.post("/integrated/interact", response_model=IntegratedInteractionResponse)
async def integrated_interaction(request: IntegratedInteractionRequest):
    """
    Process an interaction through BOTH Nurture and Experiential layers.
    
    This is the full CACA stack:
    1. Nurture Layer processes for character formation
    2. Experiential Layer tracks session context
    3. Both contexts are combined for response generation
    """
    # Set up OpenRouter client
    if request.openrouter_api_key:
        client = set_openrouter_client(request.session_id, request.openrouter_api_key, request.model_name)
    else:
        client = get_client(request.session_id)
        if not client or not client.is_configured():
            raise HTTPException(status_code=401, detail="OpenRouter API key required")
    
    # Load nurture state
    nurture_state = store.load(request.instance_id)
    if nurture_state is None:
        raise HTTPException(status_code=404, detail=f"Instance {request.instance_id} not found")
    
    # Get or create experiential session
    if request.session_id not in experiential_sessions:
        exp_engine = ExperientialEngine(
            config=DEFAULT_EXPERIENTIAL_CONFIG,
            nurture_state=nurture_state
        )
        exp_engine.initialize_session(session_id=request.session_id)
        experiential_sessions[request.session_id] = exp_engine
    else:
        exp_engine = experiential_sessions[request.session_id]
        exp_engine.nurture_state = nurture_state  # Update nurture reference
    
    # Get experiential context
    exp_context = exp_engine.get_context_for_prompt()
    
    # Set the model function on the nurture engine
    engine.set_model_fn(client.generate)
    
    # Get conversation history
    conversation_history = store.get_conversation_history(request.instance_id, limit=10)
    
    # Process through Nurture Layer (includes LLM call with character context)
    try:
        response, updated_nurture, metadata = engine.process_interaction(
            user_input=request.user_input,
            nurture_state=nurture_state,
            conversation_history=conversation_history,
            extra_context=exp_context  # Inject experiential context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    # Process through Experiential Layer (post-response)
    exp_metadata = exp_engine.process_interaction(
        user_input=request.user_input,
        assistant_response=response
    )
    
    # Save updated nurture state
    store.save(updated_nurture)
    
    # Save interaction to history
    store.save_interaction(
        instance_id=request.instance_id,
        user_input=request.user_input,
        assistant_response=response,
        metadata={
            'significance_score': metadata.significance_score,
            'was_evaluated': metadata.was_evaluated,
            'delta_magnitude': metadata.delta_magnitude,
            'phase_after': metadata.phase_after,
            'stability': updated_nurture.stability,
            'experiential': exp_metadata
        }
    )
    
    return IntegratedInteractionResponse(
        response=response,
        nurture_state=_state_to_response(updated_nurture),
        experiential_state=_experiential_to_response(exp_engine),
        metadata={
            'nurture': {
                'significance_score': metadata.significance_score,
                'was_evaluated': metadata.was_evaluated,
                'phase': metadata.phase_after,
            },
            'experiential': exp_metadata,
            'model': request.model_name
        }
    )


@app.get("/experience/context/{session_id}")
async def get_experiential_context(session_id: str):
    """Get the current experiential context string for prompt injection."""
    if session_id not in experiential_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    exp_engine = experiential_sessions[session_id]
    return {
        "session_id": session_id,
        "context": exp_engine.get_context_for_prompt(),
        "summary": exp_engine.get_state_summary()
    }


@app.get("/experience/facts/{session_id}")
async def get_session_facts(session_id: str):
    """Get salient facts from the current session."""
    if session_id not in experiential_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    exp_engine = experiential_sessions[session_id]
    facts = exp_engine.state.working_memory.salient_facts
    
    return {
        "session_id": session_id,
        "facts": [f.to_dict() for f in facts],
        "count": len(facts)
    }


@app.get("/experience/questions/{session_id}")
async def get_session_questions(session_id: str):
    """Get open questions from the current session."""
    if session_id not in experiential_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    exp_engine = experiential_sessions[session_id]
    questions = exp_engine.state.working_memory.open_questions
    
    return {
        "session_id": session_id,
        "questions": [q.to_dict() for q in questions],
        "open_count": len([q for q in questions if not q.resolved]),
        "resolved_count": len([q for q in questions if q.resolved])
    }


@app.get("/experience/commitments/{session_id}")
async def get_session_commitments(session_id: str):
    """Get commitments from the current session."""
    if session_id not in experiential_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    exp_engine = experiential_sessions[session_id]
    commitments = exp_engine.state.working_memory.commitments
    
    return {
        "session_id": session_id,
        "commitments": [c.to_dict() for c in commitments],
        "active_count": len([c for c in commitments if not c.fulfilled]),
        "fulfilled_count": len([c for c in commitments if c.fulfilled])
    }


def _experiential_to_response(exp_engine: ExperientialEngine) -> ExperientialStateResponse:
    """Convert ExperientialEngine state to response model."""
    summary = exp_engine.get_state_summary()
    return ExperientialStateResponse(
        session_id=summary.get('session_id', ''),
        interaction_count=summary.get('interaction_count', 0),
        topic_summary=summary.get('topic_summary', ''),
        emotion_summary=summary.get('emotion_summary', ''),
        user_summary=summary.get('user_summary', ''),
        facts_count=summary.get('facts_count', 0),
        open_questions=summary.get('open_questions', 0),
        active_commitments=summary.get('active_commitments', 0),
        session_familiarity=summary.get('session_familiarity', 0.0),
        total_sessions=summary.get('total_sessions', 0),
        context_string=exp_engine.get_context_for_prompt()
    )


def _state_to_response(state) -> StateResponse:
    """Convert NurtureState to StateResponse."""
    return StateResponse(
        instance_id=state.instance_id,
        phase=state.phase,
        stability=round(state.stability, 4),
        plasticity=round(state.plasticity, 4),
        interaction_count=state.interaction_count,
        significant_count=state.significant_count,
        stance={k: round(v, 4) for k, v in state.stance_json.items()},
        environment=state.env_json,
        current_threshold=round(
            get_dynamic_threshold(state.plasticity, DEFAULT_CONFIG), 4
        ),
        created_at=state.created_at.isoformat(),
        last_updated=state.last_updated.isoformat()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
