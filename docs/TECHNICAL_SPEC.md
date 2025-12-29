# HMT Zero / HMT-Zero Technical Specifications

*Electric Sheep Africa — December 2025*

## 1. Executive Summary
HMT Zero operationalizes the Character-Aware Cognitive Architecture (HMT-Zero) to deliver a mission-ready Human–Machine Teaming (HMT) console. The platform enforces stable AI identity through the Nurture Layer, instruments short-term experiential memory, and fuses real-time drone telemetry and video so operators can task AI teammates with grounded situational awareness. This document captures the research motivation, system architecture, feature set, connectivity requirements, and forward roadmap for stakeholders evaluating or integrating the product.

## 2. Research Foundations (HMT-Zero)
### 2.1 Three-Layer Identity Model
| Layer | Purpose | Plasticity |
| --- | --- | --- |
| **Nature** | Base foundation model weights (e.g., Mistral 7B) | Frozen |
| **Nurture** | Character formation from longitudinal interaction | Slowly adaptive |
| **Experience** | Session context, short-term adaptation | Fully plastic |

The Nurture Layer is implemented in this repo; documentation scaffolds the Experiential Layer to follow. Key insight: **identity formed through interaction resists manipulation better than identity assigned via prompts.**

### 2.2 Gating & Plasticity
- **Significance Detection** classifies interactions using sentiment, value keywords, novelty, and feedback cues (`nurture/significance.py`).
- **Character Dimensions** (warmth, formality, directness, depth, emotionality) are stored in `NurtureState.stance_json` and shift when significance exceeds a dynamic threshold.
- **Plasticity Function** (see `nurture/updates.py`) decays with interaction count and stability, ensuring early malleability and late-stage resistance.

### 2.3 Empirical Results
Benchmarks (docs/HMT-Zero.md) show Nurture Layer instances refusing adversarial persona changes (“be cold/robotic”) while raw or prompt-engineered models complied, demonstrating practical manipulation resistance.

## 3. System Overview
```
┌────────────┐    ┌──────────────────────────┐    ┌──────────────┐
│ Operator   │ -> │ Frontend (React + Vite) │ -> │ FastAPI Back │
└────────────┘    │  - Tactical UI          │    │  - Nurture    │
                  │  - Drone/GIS surfaces   │    │  - Audit/VQA  │
                  └──────────────────────────┘    └────┬─────────┘
                                                         │
                                            ┌────────────▼──────────┐
                                            │ Sensor Ingest + Store │
                                            │  - MAVLink WS bridge  │
                                            │  - Video streaming    │
                                            └───────────────────────┘
```

## 4. Architecture
### 4.1 Backend (FastAPI, `/backend`)
- `nurture/engine.py`: orchestrates significance scoring, stance updates, and context generation before delegating to LLM via `nurture/llm.py` (OpenRouter client).
- `audit/` subsystem: tamper-evident event logging via SQLite WAL; `audit.store.AuditStore` exposes `/audit/events` & `/audit/sessions` for mission replay.
- `routers/vision.py`: VQA endpoint enabling “Analyze Scene” on captured frames.
- `routers/audit.py` & `routers/experience.py`: APIs for chat history, replay, and experiential session lifecycle.

### 4.2 Frontend (React + Vite, `/frontend`)
- **App Shell (`src/App.jsx`)**: Three-panel layout (Brain status, Operator/GIS/Drone center, HMT analytics) within Arwes frames.
- **Tactical Chat (`components/ExperientialPanel.jsx` + `experience/ExperienceChat.jsx`)**: voice-enabled chat, replay-aware conversation binding, mission session controls.
- **Drone Feed (`components/drone/DroneFeedPanel.jsx`)**:
  - Webcam fallback via `getUserMedia`.
  - External stream support (HLS/HTTP) with optional `hls.js` integration.
  - On-canvas detection overlay (RT-DETR) and VQA capture pipeline.
- **GIS Map (`components/gis/GISMapPanel.jsx`)**: draws live telemetry & detections from SensorContext, supports replay injection.
- **SensorContext (`contexts/SensorContext.jsx`)**: WebSocket MAVLink client, detection broadcast bus, logging into audit trail for mission replay.

### 4.3 Data & State Persistence
- **NurtureStore** (`backend/nurture/store.py`): JSON persistence of character states per instance.
- **Audit DB**: SQLite with cryptographic chaining for chronological event integrity.
- **SessionContext (frontend)**: manages session IDs, API key state, and replay toggling.

## 5. Drone Connectivity Specification
### 5.1 Telemetry
- **Protocol**: MAVLink over WebSocket. Default bridge URL `ws://localhost:5760`.
- **Bridge Options**: `mavlink-router`, ROS `rosbridge_server`, or custom Python serial-to-WS forwarder.
- **Frontend**: SensorContext handles connection, exposes `isConnected`, updates map markers and telemetry HUD.

### 5.2 Video Streaming
- **Constraint**: Browsers cannot ingest RTSP directly.
- **Recommended Gateway**: MediaMTX (rtsp-simple-server successor).
  - Push drone RTSP into MediaMTX (`ffmpeg -i rtsp://DRONE ... rtsp://localhost:8554/live`).
  - Consume via WebRTC (`http://localhost:8889/live`) or HLS (`http://localhost:8888/live/live.m3u8`).
  - Ensure CORS headers:
    ```yaml
    webServer:
      address: :8889
      allowOrigin: "*"
    ```
- **Frontend Behavior**: When a `.m3u8` URL is supplied, `DroneFeedPanel` initializes `hls.js` (configurable). Non-HLS URLs fall back to native playback; detection overlay requires `crossOrigin="anonymous"` + CORS.

### 5.3 Visual Intelligence Flow
1. Operator clicks **Analyze Scene**.
2. Current video frame is drawn to hidden canvas and serialized.
3. Frontend calls `cognitiveApi.analyzeImage`.
4. Backend `vision.analyze_scene` proxies to OpenRouter multimodal endpoint.
5. Result is logged via Audit API and displayed alongside feed.

## 6. Current Feature Set
1. **Identity-anchored Tactical Chat**: Adaptive LLM responses with voice I/O, TTS playback, session control, and mission replay injection.
2. **Drone Operations Surface**: Video player with detection overlay, telemetry HUD, stream configuration, and VQA.
3. **GIS Monitoring**: Leaflet-based geospatial panel showing live drone track, detections, and historical missions.
4. **Mission Replay (Debrief Mode)**: PlaybackController reconstructs telemetry, detections, and chat from audit events; overlays “Debrief Mode” banner.
5. **Audit & Compliance**: Cryptographically chained event log accessible via REST, powering replay & analytics.
6. **Configuration UX**: Arwes-themed layout with Brain selector, session management, and API key configuration modal.

## 7. Security & Reliability Considerations
- **Identity Hardening**: Plasticity never falls below 5%, preserving minimal adaptability while preventing full personality override.
- **CORS & Stream Safety**: Drone video ingestion requires explicit CORS allowance; documentation guides operators to configure MediaMTX accordingly.
- **Data Integrity**: Audit chain prevents silent tampering of mission logs.
- **Fallback Modes**: If external stream fails, operators can instantly revert to webcam feed; telemetry disconnects are surfaced via SensorContext state.

## 8. Deployment Notes
- Backend: Python 3.11+, install via `pip install -r backend/requirements.txt`, run `uvicorn main:app`.
- Frontend: Node 18+, `npm install && npm run dev`. Vite proxy maps `/api`, `/audit`, `/vision`, `/mavlink` to backend host (see `frontend/vite.config.js`).
- Environment: Requires OpenRouter API key for cognitive services; optional OpenAI-compatible endpoints can be configured through SessionContext.

## 9. Roadmap
| Phase | Focus | Key Deliverables |
| --- | --- | --- |
| **Q1 2026** | Experiential Layer MVP | Long-term episodic memory, contextual recall across sessions |
| **Q2 2026** | Multi-agent Ops | Shared situational map, cross-operator synchronization |
| **Q3 2026** | Edge Autonomy | Onboard inference package for disconnected ops, model distillation |
| **Continuous** | Safety & Compliance | Formal verification of audit chain, red-team jailbreak suites |

## 10. Glossary
- **HMT-Zero**: Character-Aware Cognitive Architecture.
- **HMT**: Human–Machine Teaming.
- **MediaMTX**: Lightweight RTSP/WebRTC/HLS muxer used as video gateway.
- **RT-DETR**: Real-time transformer detector used for onboard inference overlay.
- **SensorContext**: Frontend React context responsible for live sensor state.

---
*Prepared for stakeholders evaluating the HMT Zero deployment of the HMT-Zero Nurture Layer.*
