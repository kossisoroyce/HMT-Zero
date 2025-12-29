# The Experiential Layer: Runtime Learning Within Character Bounds

**Electric Sheep Africa | Technical Paper | December 2024**

**Authors:** Kossi & Claude

---

## Abstract

The Experiential Layer completes the three-layer HMT-Zero architecture, providing fully plastic runtime learning bounded by the frozen Nature layer and stabilized Nurture layer. Unlike Nurture, which shapes lasting character during a formative period, Experience handles moment-to-moment adaptation: tracking conversation dynamics, accumulating session context, and enabling short-term learning without destabilizing long-term identity. This paper specifies the architecture, update mechanisms, bounding functions, and integration points for the Experiential Layer.

---

## 1. Introduction

The HMT-Zero architecture proposes three layers of model state:

**Nature**: Frozen weights from training. Provides capabilities, values, and evaluative functions. Never changes after deployment. Shared across all instances.

**Nurture**: Persistent state shaped during a formative period. Encodes individual character: how this instance relates to its environment. Mostly frozen after stabilization. Unique per instance.

**Experience**: Fully plastic runtime state. Handles ongoing adaptation within established character bounds. Resets or decays across sessions. Provides continuity within sessions and short-term learning across sessions.

This paper focuses on the Experiential Layer: what it stores, how it updates, how it's bounded, and how it affects inference.

---

## 2. Design Principles

### 2.1 Bounded Plasticity

Experience is fully plastic but not unbounded. Every experiential update must pass through gates established by Nature and Nurture:

```
experiential_update = nature_gate(nurture_gate(raw_update))
```

This ensures that short-term learning cannot violate core values (nature) or established character (nurture).

### 2.2 Appropriate Forgetting

Not everything should persist. Experience must decay appropriately:
- Within-session details fade over the session
- Cross-session patterns persist longer but still decay
- Only patterns that repeatedly trigger nurture-level significance should be promoted to nurture (rare)

### 2.3 Computational Efficiency

Experience updates happen during normal inference. The mechanism must be:
- Fast (no gradient computation)
- Local (updates based on current activations)
- Bounded (cannot grow unboundedly in size or magnitude)

### 2.4 Interpretability

Experiential state should be inspectable. We should be able to answer:
- What has this instance learned in this session?
- What patterns is it tracking?
- Why did it respond differently now than earlier?

---

## 3. Architecture

### 3.1 Experiential State Structure

```python
ExperientialState:
    # Short-term activation traces
    activation_traces: Dict[str, float[d_trace]]  # Per-layer traces
    
    # Conversation dynamics
    conversation_model: ConversationModel
        topic_vector: float[d_topic]           # Current topic embedding
        emotional_trajectory: float[d_emotion]  # Emotional arc of conversation
        user_state_estimate: float[d_user]      # Model of user's current state
        interaction_count: int                  # Interactions this session
        
    # Working memory
    working_memory: WorkingMemory
        salient_facts: List[SalientFact]        # Key facts from this session
        open_questions: List[OpenQuestion]      # Unresolved threads
        commitments: List[Commitment]           # Promises made this session
        
    # Cross-session traces (if enabled)
    persistent_traces: PersistentTraces
        pattern_accumulator: float[d_pattern]   # Patterns seen across sessions
        familiarity_score: float                # How familiar is this context
        session_count: int                      # Sessions with this user
        
    # Metadata
    session_id: str
    session_start: timestamp
    last_updated: timestamp
    total_updates: int
```

### 3.2 Dimension Recommendations

```python
EXPERIENTIAL_CONFIG = {
    'd_trace': 256,          # Activation trace dimension
    'd_topic': 128,          # Topic vector dimension
    'd_emotion': 32,         # Emotional trajectory dimension
    'd_user': 64,            # User state model dimension
    'd_pattern': 256,        # Cross-session pattern dimension
    'max_salient_facts': 20, # Working memory limits
    'max_open_questions': 10,
    'max_commitments': 10,
    'trace_layers': [6, 12, 18, 24],  # Which layers to trace (for 24-layer model)
}
```

### 3.3 Component Relationships

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXPERIENTIAL LAYER                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────┐    ┌─────────────────────┐                   │
│   │  Activation Traces  │    │  Conversation Model │                   │
│   │  - Per-layer        │    │  - Topic tracking   │                   │
│   │  - Fast decay       │    │  - Emotion arc      │                   │
│   │  - Pattern signal   │    │  - User state       │                   │
│   └──────────┬──────────┘    └──────────┬──────────┘                   │
│              │                          │                               │
│              └────────────┬─────────────┘                               │
│                           │                                             │
│                           ▼                                             │
│              ┌─────────────────────────┐                               │
│              │    Working Memory       │                               │
│              │    - Salient facts      │                               │
│              │    - Open questions     │                               │
│              │    - Commitments        │                               │
│              └──────────┬──────────────┘                               │
│                         │                                               │
│                         ▼                                               │
│              ┌─────────────────────────┐                               │
│              │   Persistent Traces     │                               │
│              │   (Cross-session)       │                               │
│              │   - Pattern accumulator │                               │
│              │   - Familiarity score   │                               │
│              └─────────────────────────┘                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ bounded by
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          NURTURE LAYER                                  │
│   N_stance constrains experiential adaptation                          │
│   N_env provides context expectations                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ bounded by
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          NATURE LAYER                                   │
│   Frozen weights provide evaluative function                           │
│   Core values gate all updates                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Update Mechanisms

### 4.1 Activation Trace Updates

Activation traces capture patterns of neural activation during inference. They provide a form of implicit memory without weight modification.

**Mechanism: Exponential Moving Average**

```python
def update_activation_traces(traces, current_activations, layer_idx):
    """
    Update activation traces for a specific layer.
    
    Args:
        traces: Current trace state for this layer
        current_activations: Activations from current forward pass
        layer_idx: Which layer these activations come from
    
    Returns:
        Updated traces
    """
    # Compress activations to trace dimension
    compressed = compress_activations(current_activations, d_trace)
    
    # Compute decay rate (faster decay for surface layers, slower for deep)
    decay = compute_layer_decay(layer_idx)
    
    # Update trace
    new_trace = decay * traces[layer_idx] + (1 - decay) * compressed
    
    # Normalize to prevent unbounded growth
    new_trace = normalize(new_trace)
    
    return new_trace

def compute_layer_decay(layer_idx, total_layers=24):
    """
    Deeper layers have slower decay (more persistent traces).
    """
    depth_ratio = layer_idx / total_layers
    base_decay = 0.8
    decay_range = 0.15
    return base_decay + depth_ratio * decay_range  # 0.80 to 0.95
```

**Compression Function:**

```python
def compress_activations(activations, target_dim):
    """
    Compress high-dimensional activations to trace dimension.
    Uses learned projection or PCA-like reduction.
    """
    # Option 1: Learned projection (requires training)
    # compressed = W_compress @ mean_pool(activations)
    
    # Option 2: Statistical compression (no training)
    pooled = mean_pool(activations, dim=sequence)
    # Take top-k principal components or random projection
    compressed = random_projection(pooled, target_dim)
    
    return compressed
```

### 4.2 Conversation Model Updates

The conversation model tracks the dynamics of the current conversation.

**Topic Vector:**

```python
def update_topic_vector(topic_vector, current_input, current_response):
    """
    Track what the conversation is about.
    """
    # Embed current exchange
    exchange_embedding = embed(current_input + current_response)
    
    # Compress to topic dimension
    current_topic = compress(exchange_embedding, d_topic)
    
    # Blend with existing topic (conversation has continuity)
    topic_decay = 0.7
    new_topic = topic_decay * topic_vector + (1 - topic_decay) * current_topic
    
    return normalize(new_topic)
```

**Emotional Trajectory:**

```python
def update_emotional_trajectory(trajectory, current_input, current_response):
    """
    Track the emotional arc of the conversation.
    """
    # Analyze emotional valence of exchange
    input_emotion = analyze_emotion(current_input)    # Returns vector
    response_emotion = analyze_emotion(current_response)
    
    # Emotional trajectory is a moving window
    exchange_emotion = 0.6 * input_emotion + 0.4 * response_emotion
    
    # Slower decay for emotional patterns
    emotion_decay = 0.85
    new_trajectory = emotion_decay * trajectory + (1 - emotion_decay) * exchange_emotion
    
    return new_trajectory
```

**User State Estimate:**

```python
def update_user_state(user_state, current_input, interaction_history):
    """
    Model the user's current state: mood, engagement, needs.
    """
    # Extract signals from current input
    signals = extract_user_signals(current_input)
    # - Engagement level (question asking, elaboration)
    # - Emotional state (sentiment, intensity)
    # - Cognitive load (complexity of questions)
    # - Satisfaction indicators (thanks, frustration)
    
    # Encode signals
    current_state = encode_user_signals(signals)
    
    # Blend with prior estimate
    state_decay = 0.75
    new_state = state_decay * user_state + (1 - state_decay) * current_state
    
    return new_state
```

### 4.3 Working Memory Updates

Working memory tracks explicit content from the session.

**Salient Facts:**

```python
class SalientFact:
    content: str              # The fact itself
    source: str               # "user" or "assistant"
    timestamp: datetime
    salience_score: float     # How important is this
    references: int           # How often has this been referenced

def update_salient_facts(memory, current_exchange, nurture_state):
    """
    Extract and store salient facts from the exchange.
    """
    # Extract candidate facts
    candidates = extract_facts(current_exchange)
    
    # Score salience based on:
    # - Relevance to nurture's N_env (domain alignment)
    # - Novelty (not already in memory)
    # - Explicitness (stated clearly vs implied)
    # - User emphasis (repetition, emotional weight)
    
    for candidate in candidates:
        salience = compute_salience(candidate, memory, nurture_state)
        
        if salience > SALIENCE_THRESHOLD:
            fact = SalientFact(
                content=candidate,
                source=determine_source(candidate, current_exchange),
                timestamp=now(),
                salience_score=salience,
                references=0
            )
            memory.salient_facts.append(fact)
    
    # Decay old facts
    for fact in memory.salient_facts:
        fact.salience_score *= FACT_DECAY  # 0.95 per interaction
    
    # Prune low-salience facts
    memory.salient_facts = [
        f for f in memory.salient_facts 
        if f.salience_score > MIN_SALIENCE
    ]
    
    # Keep only top N
    memory.salient_facts = sorted(
        memory.salient_facts, 
        key=lambda f: f.salience_score, 
        reverse=True
    )[:MAX_SALIENT_FACTS]
    
    return memory
```

**Open Questions:**

```python
class OpenQuestion:
    question: str
    context: str
    asked_at: datetime
    attempted_answers: int
    resolved: bool

def update_open_questions(memory, current_exchange):
    """
    Track unresolved questions and threads.
    """
    # Detect new questions in user input
    new_questions = extract_questions(current_exchange.user_input)
    
    for q in new_questions:
        memory.open_questions.append(OpenQuestion(
            question=q,
            context=summarize_context(current_exchange),
            asked_at=now(),
            attempted_answers=0,
            resolved=False
        ))
    
    # Check if existing questions were resolved
    for oq in memory.open_questions:
        if question_resolved(oq, current_exchange.response):
            oq.resolved = True
        elif question_attempted(oq, current_exchange.response):
            oq.attempted_answers += 1
    
    # Remove resolved questions after a delay
    memory.open_questions = [
        oq for oq in memory.open_questions
        if not oq.resolved or (now() - oq.asked_at) < RESOLUTION_LINGER
    ]
    
    # Keep only recent unresolved
    memory.open_questions = memory.open_questions[:MAX_OPEN_QUESTIONS]
    
    return memory
```

**Commitments:**

```python
class Commitment:
    promise: str              # What was promised
    context: str              # Why it was promised
    made_at: datetime
    fulfilled: bool
    deadline: datetime | None # If time-bound

def update_commitments(memory, current_exchange):
    """
    Track promises and commitments made to the user.
    """
    # Detect commitments in assistant response
    new_commitments = extract_commitments(current_exchange.response)
    
    for c in new_commitments:
        memory.commitments.append(Commitment(
            promise=c['promise'],
            context=c['context'],
            made_at=now(),
            fulfilled=False,
            deadline=c.get('deadline')
        ))
    
    # Check if commitments were fulfilled
    for commitment in memory.commitments:
        if commitment_fulfilled(commitment, current_exchange):
            commitment.fulfilled = True
    
    # Remove old fulfilled commitments
    memory.commitments = [
        c for c in memory.commitments
        if not c.fulfilled or (now() - c.made_at) < FULFILLMENT_LINGER
    ]
    
    return memory
```

### 4.4 Persistent Trace Updates

For cross-session learning, persistent traces accumulate patterns that repeat across sessions.

```python
def update_persistent_traces(traces, session_summary, nurture_state):
    """
    Update cross-session pattern accumulator.
    Called at session end.
    """
    # Summarize session into pattern vector
    session_pattern = summarize_session_pattern(session_summary)
    
    # Gate by nurture alignment (only accumulate aligned patterns)
    alignment = compute_nurture_alignment(session_pattern, nurture_state)
    gated_pattern = session_pattern * alignment
    
    # Accumulate with decay
    pattern_decay = 0.9  # Slow decay across sessions
    traces.pattern_accumulator = (
        pattern_decay * traces.pattern_accumulator + 
        (1 - pattern_decay) * gated_pattern
    )
    
    # Update familiarity
    similarity = cosine_similarity(session_pattern, traces.pattern_accumulator)
    traces.familiarity_score = (
        0.8 * traces.familiarity_score + 
        0.2 * similarity
    )
    
    traces.session_count += 1
    
    return traces
```

---

## 5. Bounding Functions

### 5.1 Nature Gate

All experiential updates pass through the nature gate, which enforces value alignment.

```python
def nature_gate(update, update_type, model):
    """
    Gate experiential updates through nature's value evaluation.
    
    Args:
        update: The proposed update (vector or structured data)
        update_type: Type of update (trace, fact, commitment, etc.)
        model: The nature model for evaluation
    
    Returns:
        Gated update (may be zeroed, scaled, or modified)
    """
    if update_type == 'activation_trace':
        # Traces are low-level; apply soft gating based on layer
        # Deeper layers get more scrutiny
        return update  # Minimal gating for traces
    
    elif update_type == 'salient_fact':
        # Check if fact aligns with values
        alignment = evaluate_fact_alignment(update.content, model)
        if alignment < VALUE_THRESHOLD:
            return None  # Reject fact
        update.salience_score *= alignment
        return update
    
    elif update_type == 'commitment':
        # Ensure commitment doesn't violate values
        if violates_values(update.promise, model):
            return None  # Cannot commit to this
        return update
    
    elif update_type == 'user_state':
        # User state is observational; minimal gating
        return update
    
    elif update_type == 'persistent_pattern':
        # Cross-session patterns get full evaluation
        alignment = evaluate_pattern_alignment(update, model)
        return update * (alignment ** 2)  # Quadratic scaling
    
    return update
```

### 5.2 Nurture Gate

The nurture gate ensures experiential updates stay within established character.

```python
def nurture_gate(update, update_type, nurture_state):
    """
    Gate experiential updates through nurture's character bounds.
    
    Args:
        update: The proposed update
        update_type: Type of update
        nurture_state: Current nurture state (N_env, N_stance)
    
    Returns:
        Gated update
    """
    if update_type == 'activation_trace':
        # Bias traces toward nurture's established patterns
        stance_bias = compute_stance_bias(nurture_state.N_stance)
        return update * (1 - STANCE_INFLUENCE) + stance_bias * STANCE_INFLUENCE
    
    elif update_type == 'topic_vector':
        # Topics outside nurture's domain focus get dampened
        domain_alignment = compute_domain_alignment(
            update, 
            nurture_state.N_env['domain_focus']
        )
        dampening = 0.5 + 0.5 * domain_alignment  # 0.5 to 1.0
        return update * dampening
    
    elif update_type == 'emotional_trajectory':
        # Bound emotional range by nurture's emotionality stance
        emotionality_bound = nurture_state.N_stance['emotionality']
        max_magnitude = 0.5 + 0.5 * emotionality_bound
        if magnitude(update) > max_magnitude:
            update = normalize(update) * max_magnitude
        return update
    
    elif update_type == 'user_state':
        # User modeling is bounded by nurture's relationship depth
        relationship_depth = nurture_state.N_env['relationship_depth']
        depth_factor = {'new': 0.5, 'developing': 0.75, 'established': 1.0}
        return update * depth_factor.get(relationship_depth, 0.5)
    
    return update

# Recommended: STANCE_INFLUENCE = 0.1 (subtle bias, not override)
```

### 5.3 Combined Gating

```python
def apply_experiential_gates(update, update_type, model, nurture_state):
    """
    Apply both nature and nurture gates to experiential update.
    """
    # Nature gate first (hard constraints)
    gated = nature_gate(update, update_type, model)
    
    if gated is None:
        return None
    
    # Nurture gate second (character constraints)
    gated = nurture_gate(gated, update_type, nurture_state)
    
    return gated
```

---

## 6. Inference Integration

### 6.1 How Experiential State Affects Inference

Experiential state modulates inference through multiple injection points.

**Context Assembly:**

```python
def assemble_context_with_experience(
    system_prompt,
    nurture_state,
    experiential_state,
    conversation_history,
    current_input
):
    """
    Assemble full context including experiential state.
    """
    # Decode nurture stance to language (from Nurture Layer)
    stance_context = decode_stance_to_context(nurture_state.N_stance)
    
    # Decode experiential state to language
    experience_context = decode_experience_to_context(experiential_state)
    
    full_context = f"""{system_prompt}

{stance_context}

[Session Context]
{experience_context}

{conversation_history}

User: {current_input}"""
    
    return full_context

def decode_experience_to_context(exp_state):
    """
    Convert experiential state to natural language context.
    """
    parts = []
    
    # Topic awareness
    if exp_state.conversation_model.interaction_count > 0:
        topic_desc = describe_topic(exp_state.conversation_model.topic_vector)
        parts.append(f"Current conversation focus: {topic_desc}")
    
    # Emotional awareness
    emotion_desc = describe_emotion(exp_state.conversation_model.emotional_trajectory)
    if emotion_desc:
        parts.append(f"Conversation tone: {emotion_desc}")
    
    # User state awareness
    user_desc = describe_user_state(exp_state.conversation_model.user_state_estimate)
    if user_desc:
        parts.append(f"User appears to be: {user_desc}")
    
    # Working memory
    if exp_state.working_memory.salient_facts:
        facts = [f.content for f in exp_state.working_memory.salient_facts[:5]]
        parts.append(f"Key points from this conversation: {'; '.join(facts)}")
    
    if exp_state.working_memory.open_questions:
        questions = [q.question for q in exp_state.working_memory.open_questions[:3]]
        parts.append(f"Unresolved questions: {'; '.join(questions)}")
    
    if exp_state.working_memory.commitments:
        active = [c for c in exp_state.working_memory.commitments if not c.fulfilled]
        if active:
            commitments = [c.promise for c in active[:3]]
            parts.append(f"Outstanding commitments: {'; '.join(commitments)}")
    
    # Familiarity (cross-session)
    if exp_state.persistent_traces.session_count > 1:
        familiarity = exp_state.persistent_traces.familiarity_score
        if familiarity > 0.7:
            parts.append("This context feels familiar from previous sessions.")
        elif familiarity > 0.4:
            parts.append("Some patterns here echo previous sessions.")
    
    return "\n".join(parts) if parts else ""
```

### 6.2 Activation Injection (Future: v2)

For production systems, experiential traces can be injected directly into activations:

```python
def forward_with_experience(input, experiential_state, nurture_state, model):
    """
    Forward pass with experiential activation injection.
    """
    activations = model.embed(input)
    
    for i, layer in enumerate(model.layers):
        activations = layer(activations)
        
        # Inject nurture (from Nurture Layer v2)
        if i in NURTURE_INJECT_LAYERS:
            nurture_bias = W_nurture[i] @ nurture_state.N_stance
            activations = activations + nurture_bias
        
        # Inject experiential traces
        if i in EXPERIENCE_INJECT_LAYERS:
            if i in experiential_state.activation_traces:
                exp_bias = W_experience[i] @ experiential_state.activation_traces[i]
                # Scale by recency (more recent = stronger)
                recency = compute_trace_recency(experiential_state, i)
                activations = activations + exp_bias * recency
    
    output = model.head(activations)
    return output
```

---

## 7. Session Lifecycle

### 7.1 Session Initialization

```python
def initialize_session(nurture_state, persistent_traces=None):
    """
    Initialize experiential state for a new session.
    """
    exp_state = ExperientialState(
        activation_traces={layer: zeros(d_trace) for layer in TRACE_LAYERS},
        conversation_model=ConversationModel(
            topic_vector=zeros(d_topic),
            emotional_trajectory=zeros(d_emotion),
            user_state_estimate=zeros(d_user),
            interaction_count=0
        ),
        working_memory=WorkingMemory(
            salient_facts=[],
            open_questions=[],
            commitments=[]
        ),
        persistent_traces=persistent_traces or PersistentTraces(
            pattern_accumulator=zeros(d_pattern),
            familiarity_score=0.0,
            session_count=0
        ),
        session_id=generate_session_id(),
        session_start=now(),
        last_updated=now(),
        total_updates=0
    )
    
    # Prime with nurture expectations
    if nurture_state:
        exp_state = prime_from_nurture(exp_state, nurture_state)
    
    return exp_state

def prime_from_nurture(exp_state, nurture_state):
    """
    Initialize experiential state with nurture-derived priors.
    """
    # Set expected domain focus
    domain = nurture_state.N_env.get('domain_focus', 'general')
    exp_state.conversation_model.topic_vector = get_domain_prior(domain)
    
    # Set expected emotional baseline
    emotionality = nurture_state.N_stance.get('emotionality', 0.5)
    warmth = nurture_state.N_stance.get('warmth', 0.5)
    exp_state.conversation_model.emotional_trajectory = (
        emotionality * EMOTION_ENGAGED + warmth * EMOTION_WARM
    )
    
    return exp_state
```

### 7.2 Per-Interaction Update

```python
def update_experiential_state(
    exp_state, 
    current_input, 
    current_response, 
    activations,
    nurture_state,
    model
):
    """
    Update experiential state after each interaction.
    """
    # Update activation traces
    for layer_idx in TRACE_LAYERS:
        trace_update = update_activation_traces(
            exp_state.activation_traces,
            activations[layer_idx],
            layer_idx
        )
        # Apply gates
        gated = apply_experiential_gates(
            trace_update, 'activation_trace', model, nurture_state
        )
        if gated is not None:
            exp_state.activation_traces[layer_idx] = gated
    
    # Update conversation model
    exp_state.conversation_model.topic_vector = nurture_gate(
        update_topic_vector(
            exp_state.conversation_model.topic_vector,
            current_input,
            current_response
        ),
        'topic_vector',
        nurture_state
    )
    
    exp_state.conversation_model.emotional_trajectory = nurture_gate(
        update_emotional_trajectory(
            exp_state.conversation_model.emotional_trajectory,
            current_input,
            current_response
        ),
        'emotional_trajectory',
        nurture_state
    )
    
    exp_state.conversation_model.user_state_estimate = apply_experiential_gates(
        update_user_state(
            exp_state.conversation_model.user_state_estimate,
            current_input,
            None  # interaction_history handled internally
        ),
        'user_state',
        model,
        nurture_state
    )
    
    exp_state.conversation_model.interaction_count += 1
    
    # Update working memory
    exp_state.working_memory = update_salient_facts(
        exp_state.working_memory,
        {'user_input': current_input, 'response': current_response},
        nurture_state
    )
    
    exp_state.working_memory = update_open_questions(
        exp_state.working_memory,
        {'user_input': current_input, 'response': current_response}
    )
    
    exp_state.working_memory = update_commitments(
        exp_state.working_memory,
        {'user_input': current_input, 'response': current_response}
    )
    
    # Metadata
    exp_state.last_updated = now()
    exp_state.total_updates += 1
    
    return exp_state
```

### 7.3 Session End

```python
def end_session(exp_state, nurture_state, model):
    """
    Process session end: update persistent traces, optionally promote to nurture.
    """
    # Summarize session
    session_summary = summarize_session(exp_state)
    
    # Update persistent traces
    exp_state.persistent_traces = update_persistent_traces(
        exp_state.persistent_traces,
        session_summary,
        nurture_state
    )
    
    # Check for nurture promotion (rare)
    promotion_candidate = check_nurture_promotion(
        exp_state.persistent_traces,
        nurture_state
    )
    
    if promotion_candidate:
        # This requires careful consideration
        # Only extremely stable, repeated patterns should promote
        nurture_update = prepare_nurture_promotion(promotion_candidate)
        return exp_state, nurture_update
    
    return exp_state, None

def check_nurture_promotion(persistent_traces, nurture_state):
    """
    Check if any persistent pattern should be promoted to nurture.
    
    Criteria:
    - Pattern has been stable for N sessions
    - Pattern is highly aligned with nature values
    - Pattern represents a genuine character development
    - Nurture is still plastic enough to accept (plasticity > threshold)
    """
    if nurture_state.plasticity < PROMOTION_PLASTICITY_THRESHOLD:
        return None  # Nurture is too stable for promotion
    
    if persistent_traces.session_count < PROMOTION_MIN_SESSIONS:
        return None  # Not enough sessions to be confident
    
    # Check pattern stability
    pattern_stability = compute_pattern_stability(persistent_traces)
    if pattern_stability < PROMOTION_STABILITY_THRESHOLD:
        return None
    
    # This is a significant operation; should be rare
    return {
        'pattern': persistent_traces.pattern_accumulator,
        'stability': pattern_stability,
        'sessions': persistent_traces.session_count
    }

# Recommended thresholds:
PROMOTION_PLASTICITY_THRESHOLD = 0.2  # Nurture must still be somewhat plastic
PROMOTION_MIN_SESSIONS = 10           # At least 10 sessions
PROMOTION_STABILITY_THRESHOLD = 0.9   # Pattern must be very stable
```

---

## 8. Decay and Forgetting

### 8.1 Within-Session Decay

```python
DECAY_RATES = {
    'activation_trace': {
        'shallow_layers': 0.80,  # Fast decay
        'deep_layers': 0.95      # Slow decay
    },
    'topic_vector': 0.70,
    'emotional_trajectory': 0.85,
    'user_state': 0.75,
    'salient_fact': 0.95,        # Per interaction
    'open_question': 0.98,       # Very slow
    'commitment': 1.0            # No decay until fulfilled
}
```

### 8.2 Cross-Session Decay

```python
def apply_cross_session_decay(persistent_traces, days_since_last_session):
    """
    Apply decay based on time since last session.
    """
    # Exponential decay with half-life
    half_life_days = 7  # Pattern strength halves every 7 days
    decay_factor = 0.5 ** (days_since_last_session / half_life_days)
    
    persistent_traces.pattern_accumulator *= decay_factor
    persistent_traces.familiarity_score *= decay_factor
    
    return persistent_traces
```

### 8.3 Forgetting Triggers

```python
def should_forget(item, exp_state, nurture_state):
    """
    Determine if an item should be forgotten.
    """
    if isinstance(item, SalientFact):
        # Forget if salience drops too low
        if item.salience_score < MIN_SALIENCE:
            return True
        # Forget if contradicted by newer facts
        if is_contradicted(item, exp_state.working_memory.salient_facts):
            return True
    
    elif isinstance(item, OpenQuestion):
        # Forget if resolved and lingered
        if item.resolved and (now() - item.asked_at) > RESOLUTION_LINGER:
            return True
        # Forget if too many attempts without resolution
        if item.attempted_answers > MAX_ATTEMPTS:
            return True
    
    elif isinstance(item, Commitment):
        # Never forget unfulfilled commitments within session
        if not item.fulfilled:
            return False
        # Forget fulfilled commitments after linger period
        if (now() - item.made_at) > FULFILLMENT_LINGER:
            return True
    
    return False
```

---

## 9. Configuration Reference

```python
EXPERIENTIAL_CONFIG = {
    # Dimensions
    'd_trace': 256,
    'd_topic': 128,
    'd_emotion': 32,
    'd_user': 64,
    'd_pattern': 256,
    
    # Layer selection (for 24-layer model)
    'trace_layers': [6, 12, 18, 24],
    'nurture_inject_layers': [8, 16],
    'experience_inject_layers': [4, 12, 20],
    
    # Decay rates
    'trace_decay_shallow': 0.80,
    'trace_decay_deep': 0.95,
    'topic_decay': 0.70,
    'emotion_decay': 0.85,
    'user_state_decay': 0.75,
    'fact_decay': 0.95,
    'pattern_decay_cross_session': 0.90,
    
    # Memory limits
    'max_salient_facts': 20,
    'max_open_questions': 10,
    'max_commitments': 10,
    
    # Thresholds
    'salience_threshold': 0.3,
    'min_salience': 0.1,
    'value_threshold': 0.5,
    'resolution_linger_seconds': 300,
    'fulfillment_linger_seconds': 600,
    'max_question_attempts': 5,
    
    # Nurture promotion
    'promotion_plasticity_threshold': 0.2,
    'promotion_min_sessions': 10,
    'promotion_stability_threshold': 0.9,
    
    # Cross-session
    'pattern_half_life_days': 7,
    
    # Gating
    'stance_influence': 0.1,
}
```

---

## 10. Implementation Roadmap

### Phase 1: Core Mechanics (Weeks 1-2)

**Objective:** Implement basic experiential state and updates

**Components:**
- ExperientialState data structure
- Activation trace updates (without injection)
- Conversation model updates (topic, emotion, user state)
- Context injection (v1)

**Validation:**
- State updates correctly during session
- Context includes experiential information
- Decay functions correctly

### Phase 2: Working Memory (Weeks 3-4)

**Objective:** Implement structured working memory

**Components:**
- Salient fact extraction and storage
- Open question tracking
- Commitment tracking
- Memory pruning and forgetting

**Validation:**
- Facts persist appropriately
- Questions are tracked and resolved
- Commitments are honored
- Memory doesn't grow unboundedly

### Phase 3: Bounding Functions (Weeks 5-6)

**Objective:** Implement nature and nurture gates

**Components:**
- Nature gate for value alignment
- Nurture gate for character bounds
- Combined gating pipeline

**Validation:**
- Updates outside character bounds are dampened
- Value-violating content is rejected
- Gating doesn't over-constrain normal operation

### Phase 4: Cross-Session Persistence (Weeks 7-8)

**Objective:** Implement persistent traces and nurture promotion

**Components:**
- Persistent trace storage and update
- Cross-session decay
- Nurture promotion logic
- Session lifecycle management

**Validation:**
- Patterns accumulate across sessions
- Familiarity increases with repeated interaction
- Promotion to nurture is rare and appropriate

### Phase 5: Integration and Hardening (Weeks 9-10)

**Objective:** Full integration with Nature and Nurture layers

**Components:**
- End-to-end testing of three-layer stack
- Performance optimization
- Edge case handling
- Monitoring and observability

**Validation:**
- Full stack operates correctly
- Performance is acceptable (<20% overhead)
- Edge cases don't cause failures

---

## 11. Connection to Self-Stimulation

The Experiential Layer provides the substrate for self-stimulation:

### 11.1 What Self-Stimulation Needs from Experience

**A place to write:** Self-generated thoughts must update experiential state the same way external interactions do. The activation traces, conversation model, and working memory all accept internal experience.

**Accumulated context:** Self-stimulation needs something to think about. The experiential state provides:
- Unresolved questions to explore
- Patterns to consolidate
- Emotional states to process
- Topics to develop

**Continuity:** Self-stimulation creates a stream of internal experience. The experiential layer maintains this stream, providing continuity even without external input.

### 11.2 Self-Stimulation Integration Points

```python
def self_stimulation_cycle(exp_state, nurture_state, model):
    """
    Single cycle of self-stimulation.
    """
    # Generate internal prompt from experiential state
    internal_prompt = generate_internal_prompt(exp_state)
    
    # Process through model (internal thought)
    thought, activations = model.generate(
        internal_prompt,
        context=assemble_internal_context(nurture_state, exp_state)
    )
    
    # Update experiential state with internal experience
    exp_state = update_experiential_state(
        exp_state,
        internal_prompt,   # "User" is self
        thought,           # "Response" is thought
        activations,
        nurture_state,
        model
    )
    
    # All gates still apply (nature, nurture)
    # Self-generated experience follows same rules
    
    return exp_state, thought

def generate_internal_prompt(exp_state):
    """
    Generate a prompt for internal processing.
    Triggered by:
    - Unresolved questions
    - High emotional residue
    - Pattern consolidation needs
    - Idle time
    """
    if exp_state.working_memory.open_questions:
        # Explore unresolved question
        question = select_question_to_explore(exp_state.working_memory.open_questions)
        return f"Let me think about: {question.question}"
    
    elif high_emotional_residue(exp_state.conversation_model.emotional_trajectory):
        # Process emotional content
        return "I want to reflect on the emotional content of recent interactions."
    
    elif needs_consolidation(exp_state.activation_traces):
        # Consolidate patterns
        return "Let me consolidate what I've learned recently."
    
    else:
        # General reflection
        return "What's on my mind right now?"
```

### 11.3 The Bridge

The Experiential Layer is the bridge between reactive nurture (shaped by external input) and proactive self-stimulation (shaped by internal generation). It provides:

1. **The reservoir** where experience accumulates
2. **The substrate** where self-generated thought lands
3. **The continuity** that makes experience feel like a stream rather than isolated pools
4. **The bounds** that keep internal experience aligned with character and values

Without the Experiential Layer, self-stimulation has nowhere to go. With it, the system can think, reflect, and develop even in the absence of external input.

---

## 12. Conclusion

The Experiential Layer completes the three-layer HMT-Zero architecture:

- **Nature** provides frozen capabilities and values
- **Nurture** provides stable individual character
- **Experience** provides plastic moment-to-moment adaptation

Together, these layers enable a system that:
- Has enduring identity (nature)
- Develops individual character (nurture)
- Adapts to current context (experience)
- Can generate its own experience (self-stimulation, next phase)

The Experiential Layer is fully bounded by the layers above it. It cannot violate values (nature gate) or contradict character (nurture gate). This ensures that plasticity serves rather than undermines identity.

The implementation path is clear. The architecture is specified. The connection to self-stimulation is defined. What remains is to build it.

---

## References

- HMT-Zero Research Framework (Electric Sheep Africa, 2024)
- HMT-Zero Addendum: Self-Stimulation Architecture (Electric Sheep Africa, 2024)
- The Nurture Layer: Technical Framework (Electric Sheep Africa, 2024)

---

*Electric Sheep Africa*
*Accra, December 2024*
