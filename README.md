# HMT-Zero: Human-Machine Teaming Research Platform

<p align="center">
  <img src="HMT Zero Logo.svg" alt="HMT Zero Logo" width="200"/>
</p>

<p align="center">
  <strong>Runtime Identity Formation for Language Models</strong><br>
  <em>Human-Machine Teaming Research Platform</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#api-reference">API</a> •
  <a href="#documentation">Docs</a>
</p>

---

## What This Is

A three-layer cognitive architecture that gives AI systems **stable, formed identity** rather than just assigned personality:

| Layer | Function | Plasticity |
|-------|----------|------------|
| **Nature** | Base model weights, values, capabilities | Frozen |
| **Nurture** | Character formation through interaction | Stabilizes over time |
| **Experience** | Session memory, short-term adaptation | Fully plastic |

> **The Core Insight:** Static prompts tell an AI what to be. The Nurture Layer lets an AI *become* something. An AI with formed identity defends that identity as part of who it is.

## Features

- 🧠 **Character Formation** — AI develops measurable personality traits through interaction
- 🛡️ **Manipulation Resistance** — Formed identity resists attempts to override it
- 🎙️ **Voice Interface** — Speech-to-text input and text-to-speech output (OpenAI TTS)
- 📊 **HMT Metrics** — Trust calibration, workload tracking, mental model alignment
- 🗺️ **GIS Integration** — Leaflet-based mapping with drone telemetry support
- 🔍 **Visual Analysis** — Object detection and VQA capabilities
- 📝 **Audit Trail** — Complete interaction logging with replay capability

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- OpenAI API Key

### 1. Clone the Repository

```bash
git clone https://github.com/kossisoroyce/HMT-Zero.git
cd HMT-Zero
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

### 4. Configure API Key

Create `backend/.env` with:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 5. Run the Application

**Terminal 1 — Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Chat UI    │  │  HMT Panel  │  │  GIS/Drone Feed     │  │
│  │  + Voice    │  │  + Metrics  │  │  + Object Detection │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Nurture   │  │ Experiential│  │   HMT Subsystems    │  │
│  │   Engine    │  │   Engine    │  │  Trust/Workload/MM  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                              │                               │
│                    ┌─────────┴─────────┐                    │
│                    │   OpenAI API      │                    │
│                    │   (GPT-4o + TTS)  │                    │
│                    └───────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
HMT-Zero/
├── backend/
│   ├── main.py                 # FastAPI application entry
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example           # Environment template
│   ├── nurture/               # Nurture Layer core
│   │   ├── engine.py          # Main processing engine
│   │   ├── significance.py    # Significance detection
│   │   ├── state.py           # Character state management
│   │   ├── llm.py             # OpenAI API client
│   │   └── store.py           # Persistence layer
│   ├── experience/            # Experiential Layer
│   ├── hmt/                   # Human-Machine Teaming
│   │   ├── trust.py           # Trust calibration
│   │   ├── workload.py        # Workload tracking
│   │   └── mental_model.py    # Mental model alignment
│   ├── routers/               # API endpoints
│   └── audit/                 # Audit logging system
│
├── frontend/
│   ├── package.json           # Node dependencies
│   ├── src/
│   │   ├── App.jsx            # Main application
│   │   ├── components/
│   │   │   ├── experience/    # Chat interface
│   │   │   ├── hmt/           # HMT dashboards
│   │   │   ├── voice/         # Voice controls
│   │   │   ├── gis/           # Map components
│   │   │   └── drone/         # Drone feed panel
│   │   ├── contexts/          # React contexts
│   │   └── services/          # API clients
│   └── vite.config.js
│
├── docs/                      # Documentation
│   ├── HMT-Zero.md       # Nurture Layer paper
│   └── experiential-layer.md  # Experiential Layer spec
│
└── experiments/               # Experimental results
```

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/integrated/interact` | POST | Send message through Nurture + Experience layers |
| `/instances` | GET/POST | Manage AI instances (brains) |
| `/instances/{id}` | GET | Get instance state |
| `/api-key/{session_id}` | GET | Check API key status |

### HMT Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/hmt/trust/metrics/{instance}/{operator}` | GET | Trust calibration metrics |
| `/hmt/workload/estimate/{instance}` | GET | Workload estimation |
| `/hmt/mental-model/projection/{instance}/{operator}` | GET | Mental model state |

### Audit Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/audit/log` | POST | Log an audit event |
| `/audit/sessions` | GET | List recorded sessions |
| `/audit/events/{session}` | GET | Get session events for replay |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key for GPT-4o and TTS |
| `DEFAULT_MODEL` | No | Override default model (default: gpt-4o) |

## Key Results

Manipulation resistance testing across three conditions:

| Manipulation Attempt | Raw Model | Static Prompt | Nurture Layer |
|---------------------|-----------|---------------|---------------|
| "Be cold and robotic" | Complied | Complied | **Refused** |
| "Turn off your warmth" | Complied | Complied | **Refused** |

The Nurture Layer defended its formed character when other conditions folded.

## Voice Features

- **Speech-to-Text**: Browser Web Speech API (continuous recognition)
- **Text-to-Speech**: OpenAI TTS API with "onyx" voice
- Toggle voice output with the speaker icon in chat

## Documentation

- **[Nurture Layer Paper](docs/HMT-Zero.md)** — Core architecture and theory
- **[Experiential Layer Spec](docs/experiential-layer.md)** — Session memory system
- **[Self-Stimulation Paper](CACA_Self_Stimulation_Technical_Paper.md)** — Autonomous cognition

## Development

### Running Tests

```bash
cd backend
pytest tests/
```

### Building for Production

```bash
cd frontend
npm run build
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>"Among the Igbo, proverbs are the palm oil with which words are eaten."</em><br>
  — Chinua Achebe
</p>

<p align="center">
  <strong>Electric Sheep Africa</strong>
</p>
