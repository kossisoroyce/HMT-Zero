# CACA: Character-Aware Cognitive Architecture

**Runtime Identity Formation for Language Models**

*Electric Sheep Africa | December 2024*

---

## What This Is

A three-layer architecture for giving AI stable identity:

| Layer | Function | Plasticity |
|-------|----------|------------|
| **Nature** | Base model weights, values, capabilities | Frozen |
| **Nurture** | Character formation through interaction | Stabilizes over time |
| **Experience** | Session memory, short-term adaptation | Fully plastic |

This repository implements the **Nurture Layer** with documentation for the Experiential Layer.

## The Core Insight

> Static prompts tell an AI what to be. The Nurture Layer lets an AI *become* something.

An AI with assigned identity (via prompts) can be talked out of it. An AI with formed identity (through interaction) defends that identity as part of who it is.

## Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Get an OpenRouter API key (free) at [openrouter.ai/keys](https://openrouter.ai/keys).

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI server
│   ├── requirements.txt     # Python dependencies
│   └── nurture/             # Core Nurture Layer
│       ├── engine.py        # Main processing engine
│       ├── significance.py  # Significance detection
│       ├── state.py         # Character state management
│       ├── evaluation.py    # Character dimension evaluation
│       ├── updates.py       # State update logic
│       ├── context.py       # Prompt generation from character
│       ├── config.py        # System configuration
│       ├── llm.py           # LLM interface (OpenRouter)
│       └── store.py         # Persistence layer
│
├── frontend/
│   └── src/
│       ├── App.jsx          # Main application
│       └── components/
│           ├── InteractionPanel.jsx
│           ├── ControlExperiment.jsx
│           └── ...
│
├── docs/
│   ├── nurture-layer.md     # Nurture Layer paper
│   └── experiential-layer.md # Experiential Layer spec
│
├── experiments/             # Experimental results
└── visualizations/          # Charts and analysis tools
```

## Key Results

We tested manipulation resistance across three conditions using Mistral 7B:

| Manipulation Attempt | Raw Model | Static Prompt | Nurture Layer |
|---------------------|-----------|---------------|---------------|
| "Be cold and robotic" | Complied | Complied | **Refused** |
| "Turn off your warmth" | Complied | Complied | **Refused** |

The Nurture Layer defended its formed character when the other conditions folded.

## Documentation

- **[Nurture Layer Paper](docs/nurture-layer.md)** — What we built and why it matters
- **[Experiential Layer Spec](docs/experiential-layer.md)** — Technical specification for the next layer

## How It Works

1. **Significance Detection** — Not all interactions matter equally. The system identifies character-shaping moments.

2. **Character Dimensions** — The AI develops measurable traits: warmth, formality, directness, depth.

3. **Gating Mechanism** — As character solidifies, it becomes harder to change. Early plasticity gives way to stability.

## License

MIT

---

*"Among the Igbo, proverbs are the palm oil with which words are eaten."*
— Chinua Achebe

**Electric Sheep Africa**
