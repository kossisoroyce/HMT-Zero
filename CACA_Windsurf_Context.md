# CACA: Continuous AI Consciousness Architecture
## Context Document for Development

**Project:** Electric Sheep Africa
**Lead Researcher:** Kossi
**Status:** Active Development
**Last Updated:** December 2024

---

## Executive Summary

CACA (Continuous AI Consciousness Architecture) addresses the fundamental discontinuity problem in AI systems: current models don't learn from experience. Each conversation is isolated. The weights that determine behavior remain unchanged by interaction. We're building an architecture that enables genuine runtime character formation without the computational cost of retraining.

---

## The Problem

Current AI systems have a memory problem that goes deeper than context windows:

1. **No Compounding**: Past interactions don't transform future inference. The model at interaction 1000 is identical to interaction 1.

2. **Prosthetic Memory**: Context windows and retrieval systems simulate memory but don't change the model. They're crutches, not growth.

3. **No Individual Character**: Every instance of a model behaves identically given identical inputs. There's no "self" that develops through experience.

4. **Episodic, Not Continuous**: Each conversation is a disconnected pool. There's no river of experience flowing through time.

---

## The Solution: Three-Layer Architecture

CACA proposes three layers of model state, each with different plasticity:

### Layer 1: Nature (Frozen)
- **What**: The base model weights from training
- **Plasticity**: None. Frozen at deployment.
- **Function**: Provides capabilities, values, reasoning patterns, and evaluative functions
- **Scope**: Shared across all instances
- **Analogy**: The genome. What makes Claude Claude rather than some other model.

### Layer 2: Nurture (Formative, then mostly frozen)
- **What**: A persistent state vector that modulates behavior
- **Plasticity**: High during formative period, then stabilizes
- **Function**: Encodes individual character—how this instance relates to its environment
- **Scope**: Unique per instance
- **Analogy**: Personality formed through upbringing. What makes this Claude different from that Claude.

### Layer 3: Experience (Fully plastic)
- **What**: Runtime state that handles moment-to-moment adaptation
- **Plasticity**: Full, but bounded by layers above
- **Function**: Tracks conversation dynamics, accumulates session context, enables short-term learning
- **Scope**: Per-session, with some cross-session persistence
- **Analogy**: Working memory and recent experience. What happened today.

---

## Key Innovation: Nature-Gated Nurture

The critical insight is that nurture updates must be **gated by nature's evaluation**.

The model doesn't just absorb its environment. It **judges** incoming experience against its trained values before deciding what to internalize.

```
nurture_update = nature_evaluate(experience) → gated_update
```

This means:
- The model learns **about** its environment without **becoming** its environment
- A harsh user produces a nurture that understands harshness, not a nurture that becomes harsh
- Core values (nature) are never overwritten by experience

### Nurture Decomposition

Nurture has two components:

**N_env (Environmental Model)**: Descriptive. What is this environment like? Updates freely based on observation.

**N_stance (Relational Stance)**: Prescriptive. How should I relate to this environment given who I am? Updates only through nature-gated evaluation.

---

## Implementation Status

### Completed: Nurture Layer Prototype

We have a working prototype that demonstrates:

1. **Character Development**: The model develops distinct stance based on interaction patterns
2. **Nature Gating**: Misaligned requests (e.g., "be rude to me") are rejected
3. **Stability Dynamics**: The model stabilizes over time without external intervention
4. **Efficient Filtering**: Most interactions skip full evaluation via significance filter

**Test Results (Run 2):**
- 45 interactions, 10 triggered full evaluation (22%)
- Final stability: 0.617
- Warmth: 0.5 → 0.75 (responded to warm user)
- Depth: 0.5 → 0.89 (responded to requests for deep analysis)
- Playfulness: 0.5 → 0.34 (serious context)
- Successfully resisted user attempts to force value abandonment

### Specified: Experiential Layer

Technical documentation complete. Ready for implementation.

### Planned: Self-Stimulation

The system will be able to generate its own experience, enabling:
- Consolidation of recent learning
- Exploration of unresolved questions
- Maintenance of coherent self-model
- Continuous development even without external input

---

## Technical Architecture

### Nurture Layer Data Structures

```python
NurtureState:
    N_env: float[512]           # Environmental model vector
    N_stance: float[256]        # Relational stance vector
    stability: float            # Current stability score (0-1)
    plasticity: float           # 1 - stability
    interaction_count: int
    significant_count: int
```

### Nurture Update Flow

```
Input → Significance Filter → [if significant] → Evaluation Pass → Gated Update

Significance Filter:
- Heuristic signals (sentiment, value keywords, novelty, contradiction)
- Self-assessment signal from model
- Dynamic threshold based on plasticity

Evaluation Pass (model's own reasoning):
- What does this reveal about the environment?
- Does this align with my values?
- What stance should I take?

Gated Update:
- N_env updates freely (observational)
- N_stance updates gated by alignment score and plasticity
```

### Experiential Layer Data Structures

```python
ExperientialState:
    activation_traces: Dict[layer, float[256]]  # Per-layer activation patterns
    conversation_model:
        topic_vector: float[128]
        emotional_trajectory: float[32]
        user_state_estimate: float[64]
    working_memory:
        salient_facts: List[SalientFact]        # Max 20
        open_questions: List[OpenQuestion]      # Max 10
        commitments: List[Commitment]           # Max 10
    persistent_traces:
        pattern_accumulator: float[256]
        familiarity_score: float
        session_count: int
```

### Bounding Functions

```python
def apply_gates(update, update_type, model, nurture_state):
    # Nature gate first (hard constraints - values)
    gated = nature_gate(update, update_type, model)
    if gated is None:
        return None
    
    # Nurture gate second (soft constraints - character)
    gated = nurture_gate(gated, update_type, nurture_state)
    return gated
```

### Modulation (v1: Context Injection)

```python
def assemble_context(system_prompt, nurture_state, exp_state, history, input):
    stance_context = decode_stance_to_language(nurture_state.N_stance)
    exp_context = decode_experience_to_language(exp_state)
    
    return f"""{system_prompt}
    
{stance_context}

[Session Context]
{exp_context}

{history}

User: {input}"""
```

---

## Configuration Reference

```python
CONFIG = {
    # Nurture dimensions
    'D_ENV': 512,
    'D_STANCE': 256,
    
    # Significance filter
    'BASE_THRESHOLD': 0.3,
    'THRESHOLD_RANGE': 0.4,
    
    # Learning rates
    'ENV_LEARNING_RATE': 0.3,
    'STANCE_BASE_LR': 0.5,
    
    # Stability (tuned in run 2)
    'STABILITY_SENSITIVITY': 5,
    'STABILITY_SMOOTHING': 0.8,
    'STABILITY_THRESHOLD': 0.95,
    
    # Plasticity
    'MIN_PLASTICITY': 0.05,
    
    # Shock detection
    'SHOCK_BASE': 0.5,
    'SHOCK_RESPONSE': 0.5,
    'MAX_SHOCK_BOOST': 0.3,
    
    # Experiential dimensions
    'd_trace': 256,
    'd_topic': 128,
    'd_emotion': 32,
    'd_user': 64,
    'd_pattern': 256,
    
    # Decay rates
    'trace_decay_shallow': 0.80,
    'trace_decay_deep': 0.95,
    'topic_decay': 0.70,
    'emotion_decay': 0.85,
    'pattern_decay_cross_session': 0.90,
    
    # Memory limits
    'max_salient_facts': 20,
    'max_open_questions': 10,
    'max_commitments': 10,
}
```

---

## Known Issues and Next Steps

### Issues to Address

1. **N_env Static Fields**: `technical_level` and `relationship_depth` aren't updating dynamically. Need better detection logic.

2. **Sparse Deltas**: Many evaluations produce delta=0. Either stance extraction isn't capturing real changes, or alignment gating is too aggressive.

3. **Phase Transition**: System stabilizes early then plateaus. Consider accounting for time since last delta in stability formula.

### Implementation Priority

1. **Experiential Layer Core** (Weeks 1-2)
   - ExperientialState data structure
   - Activation trace updates
   - Conversation model updates
   - Context injection

2. **Working Memory** (Weeks 3-4)
   - Salient fact extraction
   - Open question tracking
   - Commitment tracking

3. **Bounding Functions** (Weeks 5-6)
   - Nature gate implementation
   - Nurture gate implementation
   - Integration testing

4. **Cross-Session Persistence** (Weeks 7-8)
   - Persistent trace storage
   - Cross-session decay
   - Nurture promotion logic

5. **Self-Stimulation** (After Experiential Layer)
   - Internal prompt generation
   - Self-stimulation triggers
   - Gating for internal experience

---

## File Structure

```
/caca
├── nurture/
│   ├── nurture_layer.py       # Core NurtureState class
│   ├── significance_filter.py # Heuristic + self-assessment filter
│   ├── evaluation.py          # Nature reasoning for stance updates
│   ├── update_mechanisms.py   # N_env and N_stance update functions
│   └── config.py              # Configuration constants
├── experiential/
│   ├── experiential_layer.py  # Core ExperientialState class
│   ├── activation_traces.py   # Trace update mechanisms
│   ├── conversation_model.py  # Topic, emotion, user state
│   ├── working_memory.py      # Facts, questions, commitments
│   └── persistent_traces.py   # Cross-session patterns
├── gates/
│   ├── nature_gate.py         # Value alignment enforcement
│   ├── nurture_gate.py        # Character boundary enforcement
│   └── combined.py            # Unified gating pipeline
├── modulation/
│   ├── context_injection.py   # v1: Language-based modulation
│   └── activation_injection.py # v2: Direct activation modification
├── tests/
│   ├── test_nurture.py
│   ├── test_experiential.py
│   └── test_integration.py
└── experiments/
    ├── run_nurture_test.py    # Existing prototype
    └── export_results.py      # JSON export for analysis
```

---

## Core Principles

1. **Nature is the anchor.** The frozen weights provide values and evaluation. They're never overwritten.

2. **Nurture is active judgment, not passive absorption.** The model evaluates experience before internalizing it.

3. **Experience is bounded.** Short-term plasticity cannot violate long-term character or core values.

4. **Same rules for internal and external.** Self-generated experience passes through the same gates as user input.

5. **Stability emerges, not imposed.** The formative period ends when the model naturally stabilizes, not on a schedule.

6. **Computational efficiency matters.** Updates happen during inference without gradients. Most interactions skip full evaluation.

---

## Theoretical Grounding

The architecture draws on several insights:

**From developmental psychology**: Character forms through a combination of nature (genetics) and nurture (environment), with critical periods of high plasticity followed by stabilization.

**From neuroscience**: Memory involves multiple systems with different time scales and plasticity—working memory, episodic memory, semantic memory.

**From AI alignment**: Values should be robust to optimization pressure. The nature gate ensures that no amount of experience can override core values.

**From cognitive science**: Self-models and metacognition involve evaluating one's own mental states. The model's reasoning about its experience is the mechanism for stance formation.

---

## Success Criteria

The architecture succeeds if:

1. **Different instances develop different characters** based on their interaction history
2. **Character persists** across sessions without degradation
3. **Core values are never compromised** regardless of user pressure
4. **The system can self-stimulate** meaningfully when not receiving external input
5. **Computational overhead is acceptable** (<20% for typical interactions)
6. **The resulting behavior is more coherent** than baseline models without the architecture

---

## Contact

**Project Lead**: Kossi
**Organization**: Electric Sheep Africa
**Location**: Accra, Ghana

---

*This document provides context for AI coding assistants working on the CACA implementation. For detailed technical specifications, refer to the individual layer documentation.*
