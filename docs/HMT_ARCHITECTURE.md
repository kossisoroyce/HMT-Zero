# Human-Machine Teaming (HMT) Architecture Extension

## Overview

This document outlines how to extend the HMT-Zero (Continuous AI Consciousness Architecture) to support Human-Machine Teaming research. The goal is to create an AI system that:

1. Develops calibrated trust with human operators
2. Explains its reasoning in operator-appropriate ways
3. Adapts interaction style to operator workload
4. Maintains shared mental models with operators

---

## 1. Trust Calibration System

### Concept
Trust should be **calibrated** - operators should trust the AI when it's right and distrust it when it's uncertain. This requires the AI to accurately communicate its confidence.

### New Data Structures

```python
# backend/hmt/trust.py

@dataclass
class ConfidenceSignal:
    """AI's self-assessed confidence in a recommendation."""
    recommendation_id: str
    confidence_score: float  # 0.0 - 1.0
    confidence_basis: List[str]  # What factors contribute to confidence
    uncertainty_sources: List[str]  # What could make this wrong
    
@dataclass  
class TrustCalibrationState:
    """Tracks alignment between AI confidence and actual outcomes."""
    operator_id: str
    
    # Calibration metrics
    total_recommendations: int = 0
    high_confidence_correct: int = 0  # AI said confident, was right
    high_confidence_incorrect: int = 0  # AI said confident, was wrong
    low_confidence_correct: int = 0  # AI said uncertain, was right
    low_confidence_incorrect: int = 0  # AI said uncertain, was wrong
    
    # Derived metrics
    @property
    def calibration_score(self) -> float:
        """How well does confidence predict correctness? 1.0 = perfectly calibrated."""
        if self.total_recommendations == 0:
            return 0.5
        high_conf_accuracy = self.high_confidence_correct / max(1, self.high_confidence_correct + self.high_confidence_incorrect)
        low_conf_accuracy = self.low_confidence_correct / max(1, self.low_confidence_correct + self.low_confidence_incorrect)
        # Good calibration = high confidence when right, low confidence when wrong
        return (high_conf_accuracy + (1 - low_conf_accuracy)) / 2
    
    @property
    def overtrust_risk(self) -> float:
        """Risk that operator trusts AI too much."""
        if self.total_recommendations == 0:
            return 0.5
        return self.high_confidence_incorrect / max(1, self.high_confidence_correct + self.high_confidence_incorrect)
```

### Integration with Existing Nurture Layer

```python
# Extend NurtureState in backend/nurture/state.py

class NurtureState:
    # ... existing fields ...
    
    # HMT Extension
    trust_calibration: Dict[str, TrustCalibrationState] = field(default_factory=dict)
    
    def get_trust_state(self, operator_id: str) -> TrustCalibrationState:
        if operator_id not in self.trust_calibration:
            self.trust_calibration[operator_id] = TrustCalibrationState(operator_id=operator_id)
        return self.trust_calibration[operator_id]
    
    def record_outcome(self, operator_id: str, recommendation_id: str, 
                       was_correct: bool, ai_confidence: float):
        """Record whether AI recommendation was correct for calibration tracking."""
        state = self.get_trust_state(operator_id)
        state.total_recommendations += 1
        
        high_confidence = ai_confidence > 0.7
        if high_confidence and was_correct:
            state.high_confidence_correct += 1
        elif high_confidence and not was_correct:
            state.high_confidence_incorrect += 1
        elif not high_confidence and was_correct:
            state.low_confidence_correct += 1
        else:
            state.low_confidence_incorrect += 1
```

### API Endpoints

```python
# backend/routers/hmt.py

@router.post("/hmt/recommendation")
async def make_recommendation(
    instance_id: str,
    operator_id: str,
    context: str,
    options: List[str]
) -> RecommendationResponse:
    """AI makes a recommendation with calibrated confidence."""
    pass

@router.post("/hmt/feedback")
async def record_feedback(
    instance_id: str,
    operator_id: str,
    recommendation_id: str,
    was_correct: bool
) -> TrustCalibrationState:
    """Operator provides feedback on recommendation quality."""
    pass

@router.get("/hmt/trust/{instance_id}/{operator_id}")
async def get_trust_metrics(
    instance_id: str,
    operator_id: str
) -> TrustMetrics:
    """Get current trust calibration metrics for operator."""
    pass
```

---

## 2. Explanation Generation

### Concept
Convert internal monologue into operator-readable justifications. Different operators need different explanation styles.

### Explanation Levels

```python
# backend/hmt/explanation.py

class ExplanationLevel(Enum):
    BRIEF = "brief"      # One sentence: "Recommended X because Y"
    STANDARD = "standard"  # Key factors + confidence
    DETAILED = "detailed"  # Full reasoning chain
    TECHNICAL = "technical"  # Include uncertainty quantification

@dataclass
class Explanation:
    level: ExplanationLevel
    summary: str  # Always present
    key_factors: List[str]  # Main reasons
    confidence_statement: str  # "I'm fairly confident because..."
    caveats: List[str]  # What could make this wrong
    reasoning_chain: Optional[List[str]] = None  # For detailed/technical
    
class ExplanationGenerator:
    def __init__(self, experiential_state: ExperientialState, nurture_state: NurtureState):
        self.exp_state = experiential_state
        self.nurture_state = nurture_state
    
    def generate(self, 
                 internal_thought: str,
                 recommendation: str,
                 level: ExplanationLevel) -> Explanation:
        """Convert internal reasoning to operator-facing explanation."""
        
        # Extract key factors from working memory
        relevant_facts = self._get_relevant_facts(recommendation)
        open_questions = self._get_relevant_questions(recommendation)
        
        # Assess confidence based on evidence quality
        confidence = self._assess_confidence(relevant_facts, open_questions)
        
        # Generate level-appropriate explanation
        if level == ExplanationLevel.BRIEF:
            return self._generate_brief(recommendation, relevant_facts[0] if relevant_facts else None)
        elif level == ExplanationLevel.STANDARD:
            return self._generate_standard(recommendation, relevant_facts, confidence)
        # ... etc
```

### Integration with Self-Stimulation

```python
# Modify backend/experience/self_stimulation/generator.py

def generate_internal_prompt(state: ExperientialState) -> Optional[InternalPrompt]:
    # ... existing logic ...
    
    # NEW: Tag thoughts that could become explanations
    prompt.is_explainable = True
    prompt.explanation_hooks = extract_explanation_hooks(prompt)
    return prompt
```

---

## 3. Workload-Aware Interaction

### Concept
Detect operator cognitive load and adjust AI verbosity/proactivity accordingly.

### Workload Signals

```python
# backend/hmt/workload.py

@dataclass
class WorkloadEstimate:
    """Estimated operator cognitive workload."""
    level: float  # 0.0 (idle) to 1.0 (overloaded)
    
    # Input signals
    response_latency_ms: float  # Slower responses = higher load
    message_length_trend: float  # Shorter messages = higher load
    error_rate: float  # More typos/corrections = higher load
    time_since_break: float  # Fatigue factor
    
    # Derived
    @property
    def interaction_mode(self) -> str:
        if self.level < 0.3:
            return "proactive"  # AI can initiate, provide details
        elif self.level < 0.7:
            return "responsive"  # AI responds when asked, moderate detail
        else:
            return "minimal"  # Brief responses only, no unsolicited info

class WorkloadTracker:
    def __init__(self):
        self.message_timestamps: List[float] = []
        self.message_lengths: List[int] = []
        self.response_latencies: List[float] = []
    
    def record_operator_message(self, message: str, timestamp: float):
        # Track response latency
        if self.message_timestamps:
            latency = timestamp - self.message_timestamps[-1]
            self.response_latencies.append(latency)
        
        self.message_timestamps.append(timestamp)
        self.message_lengths.append(len(message))
    
    def estimate_workload(self) -> WorkloadEstimate:
        # Compute workload from recent signals
        recent_latencies = self.response_latencies[-5:]
        recent_lengths = self.message_lengths[-5:]
        
        # Normalize and combine signals
        # ...
        return WorkloadEstimate(...)
```

### Adaptive Response Generation

```python
# Modify backend/nurture/engine.py

class NurtureEngine:
    def __init__(self, ...):
        # ... existing ...
        self.workload_tracker = WorkloadTracker()
    
    async def process_interaction(self, ...):
        # Track workload
        self.workload_tracker.record_operator_message(user_input, time.time())
        workload = self.workload_tracker.estimate_workload()
        
        # Adjust response based on workload
        response_config = self._get_response_config(workload)
        
        # Generate response with config
        response = await self._generate_response(
            user_input,
            max_length=response_config.max_length,
            include_explanation=response_config.include_explanation,
            proactive_info=response_config.proactive_info
        )
```

---

## 4. Shared Mental Model Tracking

### Concept
Track what the operator believes about the AI's state vs. actual AI state. Detect and repair misalignments.

### Mental Model Structures

```python
# backend/hmt/mental_model.py

@dataclass
class AIStateProjection:
    """What the operator likely believes about AI state."""
    
    # Beliefs about AI knowledge
    known_facts: Set[str]  # Facts operator has told AI
    assumed_facts: Set[str]  # Facts operator assumes AI knows
    
    # Beliefs about AI intentions
    understood_goals: List[str]  # Goals operator thinks AI has
    perceived_priorities: Dict[str, float]  # What operator thinks AI prioritizes
    
    # Beliefs about AI capabilities
    assumed_capabilities: Set[str]
    assumed_limitations: Set[str]
    
    # Confidence in AI
    trust_level: float  # Operator's subjective trust

@dataclass
class MentalModelAlignment:
    """Measures alignment between operator's model and AI's actual state."""
    
    # Knowledge alignment
    knowledge_overlap: float  # % of AI knowledge operator knows about
    false_beliefs: List[str]  # Things operator thinks AI knows but doesn't
    unknown_knowledge: List[str]  # Things AI knows that operator doesn't realize
    
    # Goal alignment  
    goal_alignment: float  # Do operator and AI agree on priorities?
    
    # Capability alignment
    overestimated_capabilities: List[str]  # Operator expects too much
    underestimated_capabilities: List[str]  # Operator doesn't know AI can do this
    
    @property
    def misalignment_risk(self) -> float:
        """Risk of coordination failure due to misaligned mental models."""
        return 1.0 - (self.knowledge_overlap * 0.4 + self.goal_alignment * 0.6)

class MentalModelTracker:
    def __init__(self, experiential_state: ExperientialState):
        self.exp_state = experiential_state
        self.operator_projection = AIStateProjection(...)
    
    def update_from_interaction(self, user_input: str, ai_response: str):
        """Update operator's mental model based on interaction."""
        
        # What did we reveal about our state?
        revealed_knowledge = self._extract_revealed_knowledge(ai_response)
        self.operator_projection.known_facts.update(revealed_knowledge)
        
        # What does operator's question imply about their beliefs?
        implied_beliefs = self._infer_operator_beliefs(user_input)
        self._update_projection(implied_beliefs)
    
    def detect_misalignment(self) -> List[MisalignmentAlert]:
        """Detect critical misalignments that need correction."""
        alerts = []
        
        # Check for dangerous false beliefs
        for belief in self.operator_projection.assumed_facts:
            if belief not in self.exp_state.working_memory.salient_facts:
                alerts.append(MisalignmentAlert(
                    type="false_belief",
                    description=f"Operator may believe I know: {belief}",
                    severity=self._assess_severity(belief)
                ))
        
        return alerts
    
    def generate_alignment_repair(self, alert: MisalignmentAlert) -> str:
        """Generate a statement to repair mental model misalignment."""
        if alert.type == "false_belief":
            return f"I should clarify - I don't actually have information about {alert.description}"
        # ... other repair types
```

---

## 5. Integration Architecture

### New Module Structure

```
backend/
├── hmt/
│   ├── __init__.py
│   ├── trust.py           # Trust calibration
│   ├── explanation.py     # Explanation generation
│   ├── workload.py        # Workload tracking
│   ├── mental_model.py    # Shared mental model
│   └── config.py          # HMT-specific config
├── routers/
│   └── hmt.py             # HMT API endpoints
```

### Config Extension

```python
# Add to backend/system_config.py

@dataclass
class HMTConfig:
    # Trust calibration
    high_confidence_threshold: float = 0.7
    calibration_window_size: int = 50  # Recent recommendations to consider
    
    # Workload
    workload_window_seconds: float = 300.0  # 5 min window
    high_workload_threshold: float = 0.7
    
    # Mental model
    alignment_check_interval: int = 5  # Check every N interactions
    misalignment_alert_threshold: float = 0.3

@dataclass
class SystemConfig:
    nurture: NurtureConfig = field(default_factory=NurtureConfig)
    experiential: ExperientialConfig = field(default_factory=ExperientialConfig)
    self_stimulation: SelfStimulationConfig = field(default_factory=SelfStimulationConfig)
    hmt: HMTConfig = field(default_factory=HMTConfig)  # NEW
```

### Frontend Extension

```
frontend/src/
├── components/
│   └── hmt/
│       ├── TrustDashboard.jsx      # Calibration visualization
│       ├── ExplanationPanel.jsx    # Show AI reasoning
│       ├── WorkloadIndicator.jsx   # Operator state
│       └── MentalModelView.jsx     # Alignment visualization
```

---

## 6. Research Metrics

These are the metrics that would be valuable for HMT research papers/proposals:

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Calibration Score** | Does AI confidence predict correctness? | Brier score over recommendations |
| **Trust Appropriateness** | Does operator trust match AI reliability? | Correlation(operator_trust, actual_accuracy) |
| **Explanation Satisfaction** | Do explanations help operators? | Post-task survey + decision quality |
| **Workload Adaptation** | Does AI reduce operator burden? | NASA-TLX before/after |
| **Mental Model Accuracy** | Does operator understand AI? | Prediction task accuracy |
| **Coordination Efficiency** | Do human+AI perform better together? | Task completion time/accuracy vs baselines |

---

## 7. Implementation Priority

### Phase 1: Trust Calibration (2-3 weeks)
- Core trust tracking data structures
- Confidence scoring for recommendations
- Feedback API for outcome recording
- Basic trust dashboard in frontend

### Phase 2: Explanation Generation (2 weeks)
- Explanation generator from internal monologue
- Multiple explanation levels
- Frontend explanation panel

### Phase 3: Workload Adaptation (2 weeks)
- Workload signal tracking
- Adaptive response configuration
- Workload indicator in frontend

### Phase 4: Mental Model Tracking (3-4 weeks)
- Mental model data structures
- Misalignment detection
- Repair generation
- Alignment visualization

---

## 8. Funding Alignment

This architecture directly addresses:

- **DARPA Competency-Aware ML**: Trust calibration + explanation generation
- **ONR Human-AI Teaming**: All four components
- **AFRL Loyal Wingman**: Workload adaptation + shared mental model
- **NSF Future of Work**: Trust + explanation for human-AI collaboration

The key differentiator vs. other HMT research: **persistent character development** means the AI-operator relationship evolves over time, not just within a single session.
