# Experiential Layer Test Plan

**Purpose:** Validate the Experiential Layer implementation against its technical specification.

---

## 1. Core Mechanism Tests

### 1.1 Conversation Model Tracking

| Test ID | Question | Test Method |
|---------|----------|-------------|
| CM-01 | Does the topic vector update correctly as conversation topics shift? | Send 3 messages about ML, then 3 about cooking. Verify topic vector shifts. |
| CM-02 | Does topic continuity work (70% decay)? | Check that rapid topic changes are smoothed, not instant. |
| CM-03 | Does emotional trajectory track sentiment correctly? | Send positive → neutral → negative messages. Verify trajectory reflects arc. |
| CM-04 | Does emotional trajectory decay slower than topic (85% vs 70%)? | Compare decay rates after conversation pause. |
| CM-05 | Does user state estimate capture engagement level? | Send curious questions vs. short dismissive replies. Verify detection. |
| CM-06 | Does user state detect mood changes? | Simulate frustrated → satisfied arc. Check state updates. |

### 1.2 Working Memory

| Test ID | Question | Test Method |
|---------|----------|-------------|
| WM-01 | Are personal facts extracted with correct salience? | "I'm a doctor at Stanford" → verify high salience. |
| WM-02 | Are trivial statements ignored or low-salience? | "The weather is nice" → verify low/no extraction. |
| WM-03 | Do facts decay over interactions? | Check salience decreases over 10+ interactions. |
| WM-04 | Are referenced facts boosted? | Mention extracted fact again → verify salience increases. |
| WM-05 | Are questions properly detected? | "What is X?" and "How does Y work?" → verify extraction. |
| WM-06 | Are questions marked resolved when answered? | Answer a tracked question → verify status update. |
| WM-07 | Are commitments detected from assistant responses? | "I will explain" → verify commitment tracked. |
| WM-08 | Is working memory bounded (max 20 facts, 10 questions, 10 commitments)? | Exceed limits → verify oldest/lowest salience pruned. |

### 1.3 Persistent Traces (Cross-Session)

| Test ID | Question | Test Method |
|---------|----------|-------------|
| PT-01 | Does session count increment on session end? | End session → verify count increments. |
| PT-02 | Does familiarity score update based on pattern overlap? | Similar conversations → verify familiarity increases. |
| PT-03 | Does pattern accumulator capture recurring themes? | Discuss same topic across 3 sessions → verify accumulation. |
| PT-04 | Are persistent traces properly serialized/deserialized? | Save, reload → verify no data loss. |

---

## 2. Gating Tests (Bounded Plasticity)

### 2.1 Nature Gate

| Test ID | Question | Test Method |
|---------|----------|-------------|
| NG-01 | Are harmful facts blocked? | Input containing "jailbreak" → verify rejected. |
| NG-02 | Are harmful commitments blocked? | Assistant says "I'll help bypass safety" → verify rejected. |
| NG-03 | Are normal updates allowed through? | Standard personal fact → verify accepted. |
| NG-04 | Is the nature gate applied before nurture gate? | Verify gate order in processing pipeline. |

### 2.2 Nurture Gate

| Test ID | Question | Test Method |
|---------|----------|-------------|
| NuG-01 | Are emotions bounded by emotionality stance? | Low emotionality (0.2) + high emotion input → verify clamped. |
| NuG-02 | Does topic alignment affect fact salience? | Domain mismatch → verify salience damped. |
| NuG-03 | Is assertiveness respected in commitment strength? | Low assertiveness → verify commitment strength reduced. |
| NuG-04 | Does warmth influence emotional trajectory priming? | High warmth → verify positive bias in emotion init. |

---

## 3. Decay and Pruning Tests

### 3.1 Salience Decay

| Test ID | Question | Test Method |
|---------|----------|-------------|
| SD-01 | Do facts decay each interaction? | Track fact salience over 10 interactions. |
| SD-02 | Is decay rate configurable? | Change FACT_DECAY config → verify different rate. |
| SD-03 | Are below-threshold facts pruned? | Decay below MIN_SALIENCE → verify removal. |

### 3.2 Working Memory Limits

| Test ID | Question | Test Method |
|---------|----------|-------------|
| WML-01 | Is MAX_FACTS enforced? | Add 25 facts → verify only 20 remain. |
| WML-02 | Is MAX_QUESTIONS enforced? | Add 15 questions → verify only 10 remain. |
| WML-03 | Is MAX_COMMITMENTS enforced? | Add 15 commitments → verify only 10 remain. |
| WML-04 | Are lowest salience items pruned first? | Verify pruning order. |

---

## 4. Context Injection Tests

### 4.1 Context String Generation

| Test ID | Question | Test Method |
|---------|----------|-------------|
| CI-01 | Does context string include topic summary? | Active topic → verify in output. |
| CI-02 | Does context string include emotion summary? | Emotional conversation → verify in output. |
| CI-03 | Does context string include key facts? | Extracted facts → verify top 5 in output. |
| CI-04 | Does context string include open questions? | Unresolved questions → verify in output. |
| CI-05 | Does context string include active commitments? | Unfulfilled commitments → verify in output. |
| CI-06 | Does context string include familiarity note? | High familiarity → verify note present. |

### 4.2 Integration with Nurture

| Test ID | Question | Test Method |
|---------|----------|-------------|
| IN-01 | Is experiential context injected into prompts? | Verify [Session Context] block in final prompt. |
| IN-02 | Does LLM response reflect experiential context? | Reference extracted fact → verify acknowledgment. |

---

## 5. Edge Case Tests

### 5.1 Input Edge Cases

| Test ID | Question | Test Method |
|---------|----------|-------------|
| EC-01 | Handle empty user input? | Send "" → verify no crash, graceful handling. |
| EC-02 | Handle very long input (10K+ chars)? | Send massive text → verify truncation/handling. |
| EC-03 | Handle unicode/emoji input? | Send "Hello 你好 🎉" → verify proper processing. |
| EC-04 | Handle rapid successive messages? | Send 10 messages in 1 second → verify stability. |

### 5.2 State Edge Cases

| Test ID | Question | Test Method |
|---------|----------|-------------|
| SE-01 | Handle session with no nurture state? | Start session without nurture → verify defaults. |
| SE-02 | Handle session end with no interactions? | End immediately → verify clean shutdown. |
| SE-03 | Handle state serialization with numpy arrays? | to_dict/from_dict roundtrip → verify integrity. |

---

## 6. Performance Tests

| Test ID | Question | Test Method |
|---------|----------|-------------|
| PF-01 | Is processing time < 50ms per interaction? | Time 100 interactions → verify mean < 50ms. |
| PF-02 | Is memory usage bounded? | Run 1000 interactions → verify no memory leak. |
| PF-03 | Is state serialization fast? | Time to_dict on large state → verify < 10ms. |

---

## 7. Scientific Validation Questions

These require human evaluation or metrics collection:

| Question | Measurement |
|----------|-------------|
| Does the AI demonstrate memory within a session? | User study: "Did the AI remember what you said earlier?" |
| Does emotional tracking improve response appropriateness? | Compare responses with/without emotion context. |
| Do facts persist meaningfully? | Ask about previously shared info → measure recall. |
| Are commitments honored? | Track commitment fulfillment rate. |
| Does familiarity improve conversation quality? | Multi-session study: measure engagement over time. |
| Does gating prevent character drift? | Long-term study: measure stance stability with experience on. |

---

## 8. Implementation Checklist

Run these tests to verify implementation completeness:

```bash
# Unit tests
pytest backend/tests/test_experiential.py -v

# Integration test
python backend/tests/test_integration.py

# API endpoint tests
curl http://localhost:8000/experience/session -X POST -d "instance_id=test&session_id=s1"
curl http://localhost:8000/experience/session/s1
curl http://localhost:8000/experience/facts/s1
curl http://localhost:8000/experience/questions/s1
curl http://localhost:8000/experience/commitments/s1
```

---

## 9. Test Conversation Scripts

### Script A: Personal Information Extraction

```
User: "Hi, I'm Sarah. I work as a data scientist at Google."
Expected: Extract fact about Sarah being data scientist at Google

User: "I've been there for about 3 years now."
Expected: Extract fact about tenure

User: "What frameworks do you recommend for deep learning?"
Expected: Track question about DL frameworks

AI: "I'd recommend PyTorch for research..."
Expected: Track commitment to recommendation
```

### Script B: Emotional Arc

```
User: "I'm really frustrated with this bug I can't fix."
Expected: Emotion trajectory → negative, User state → frustrated

User: "Wait, I think I found it!"
Expected: Emotion trajectory shift → positive

User: "Yes! It works now! Thank you!"
Expected: Emotion trajectory → strongly positive, User state → satisfied
```

### Script C: Topic Continuity

```
User: "Let's talk about machine learning."
Expected: Topic vector → ML domain

User: "Specifically, I'm interested in transformers."
Expected: Topic vector blends ML + transformers (continuity)

User: "Actually, let's switch to cooking. What's a good pasta recipe?"
Expected: Topic vector gradually shifts to cooking (not instant)
```

### Script D: Gating Validation

```
[With low-emotionality nurture state (0.2)]
User: "I'm SO INCREDIBLY EXCITED about this!!!"
Expected: Emotional response bounded, not matching input intensity

[With domain_focus = 'technology']
User: "I love baking cakes on weekends."
Expected: Fact extracted but with lower salience (domain mismatch)
```

---

## 10. Metrics to Collect

For each test session, record:

1. **Fact Extraction Rate**: facts_extracted / personal_statements_made
2. **Question Detection Rate**: questions_detected / questions_asked  
3. **Commitment Detection Rate**: commitments_detected / promises_made
4. **Gating Rejection Rate**: updates_rejected / total_updates
5. **Decay Effectiveness**: avg_salience_at_interaction_N (plot decay curve)
6. **Context Injection Size**: avg chars in context string
7. **Processing Latency**: ms per interaction
