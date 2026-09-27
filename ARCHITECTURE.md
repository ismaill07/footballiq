# FootballIQ — System Architecture

## 1. Architectural Goal

FootballIQ is an end-to-end football video intelligence system.

The architecture should transform unstructured football video into:

**Detection → Tracking → Structured Events → Analytics → AI Interpretation → User Interface**

The architecture must remain modular so that individual AI components can be improved without rewriting the entire system.

---

# 2. High-Level Architecture

```text
                    ┌─────────────────────┐
                    │      USER           │
                    │                     │
                    │ Upload Match Video  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   WEB FRONTEND      │
                    │      React          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FASTAPI         │
                    │      BACKEND        │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       VIDEO PROCESSING         │
              │                                │
              │ OpenCV / FFmpeg                │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       COMPUTER VISION          │
              │                                │
              │ Player Detection               │
              │ Ball Detection                 │
              │ Referee Detection              │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │        OBJECT TRACKING         │
              │                                │
              │ ByteTrack / BoT-SORT           │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       TEAM IDENTIFICATION      │
              │                                │
              │ Color / Clustering / ML        │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       PITCH MAPPING            │
              │                                │
              │ Homography / Perspective       │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │      EVENT EXTRACTION          │
              │                                │
              │ Pass / Shot / Possession       │
              │ Movement / Zones               │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       ANALYTICS ENGINE         │
              │                                │
              │ Player Statistics              │
              │ Team Statistics                │
              │ Tactical Metrics               │
              └────────────────┬───────────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                     ▼                   ▼
          ┌──────────────────┐  ┌──────────────────┐
          │ DATABASE         │  │ LLM ANALYST      │
          │                  │  │                  │
          │ Match Data       │  │ Evidence-based   │
          │ Player Data      │  │ explanations     │
          │ Events           │  │                  │
          └────────┬─────────┘  └────────┬─────────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
                    ┌─────────────────────┐
                    │   WEB DASHBOARD     │
                    │                     │
                    │ Charts              │
                    │ Heatmaps            │
                    │ Player Analysis     │
                    │ AI Analyst           │
                    └─────────────────────┘
```

---

# 3. Major Components

## 3.1 Frontend

Technology:

React

Responsibilities:

* Video upload
* Match selection
* Processing status
* Match dashboard
* Player dashboard
* Tactical dashboard
* AI analyst interface
* Visualizations

The frontend should not perform heavy AI computation.

---

# 4. Backend

Technology:

FastAPI

Responsibilities:

* Authentication if eventually required
* Video upload
* Job creation
* Processing status
* Results API
* Match API
* Player API
* Analytics API
* AI analyst API

Example API structure:

```text
POST /api/matches
GET  /api/matches/{id}
POST /api/matches/{id}/process
GET  /api/matches/{id}/status

GET /api/matches/{id}/players
GET /api/matches/{id}/analytics
GET /api/matches/{id}/events

POST /api/matches/{id}/ask
```

---

# 5. Video Processing Layer

Responsibilities:

* Read video
* Extract frames
* Determine FPS
* Resize frames when necessary
* Handle video metadata
* Save processed outputs
* Generate preview clips

Potential tools:

* OpenCV
* FFmpeg

Do not process every frame unnecessarily if sampling can provide sufficient performance.

The frame-sampling strategy should be configurable.

---

# 6. Detection Layer

Responsibilities:

Detect:

* Players
* Ball
* Referee
* Goalkeeper where possible

Potential model:

YOLO-family object detector or another suitable current detector.

The exact model must be selected after evaluating:

* Accuracy
* Speed
* Hardware requirements
* Licensing
* Availability
* Ease of deployment

Detection output should have a consistent schema.

Example:

```json
{
  "frame": 1024,
  "detections": [
    {
      "class": "player",
      "confidence": 0.91,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

---

# 7. Tracking Layer

Input:

Detection results.

Output:

Persistent object IDs.

Example:

```json
{
  "frame": 1024,
  "track_id": 17,
  "class": "player",
  "bbox": [x1, y1, x2, y2]
}
```

Potential tracking algorithms:

* ByteTrack
* BoT-SORT

Tracking performance should be evaluated using appropriate metrics such as:

* IDF1
* MOTA
* ID switches
* Track fragmentation

---

# 8. Team Classification

The system should associate tracked players with teams.

Possible pipeline:

```text
Player Bounding Box
        ↓
Crop Jersey Region
        ↓
Feature Extraction
        ↓
Color / Embedding
        ↓
Clustering / Classification
        ↓
Team A / Team B / Referee
```

Initial implementation should favor a simple robust solution.

Advanced classification can be introduced later.

---

# 9. Pitch Mapping

Player image coordinates are not directly useful for physical football analytics because of perspective distortion.

Therefore:

```text
Image Coordinates
        ↓
Pitch Landmarks
        ↓
Homography Matrix
        ↓
Pitch Coordinates
```

This enables approximate spatial calculations.

Potential outputs:

```text
x = 42.1m
y = 27.8m
```

These values should be clearly described as estimates when exact calibration is unavailable.

---

# 10. Event Extraction

The event engine converts trajectories and detections into structured football events.

Potential events:

* Player movement
* Possession change
* Pass
* Shot
* Ball recovery
* Entry into zones
* High-intensity movement
* Team transition

Example:

```json
{
  "timestamp": 125.4,
  "event": "possession_change",
  "team": "Team A",
  "player_id": 7
}
```

Event detection should initially focus on events that can be reliably inferred.

---

# 11. Analytics Engine

The analytics engine should NOT depend on the LLM.

It should use deterministic calculations wherever possible.

Examples:

### Distance

```text
Distance = Σ distance(position[t], position[t-1])
```

### Speed

```text
Speed = Distance / Time
```

### Heatmap

Use player pitch coordinates aggregated over time.

### Possession

Estimate using ball tracking and player proximity or another documented methodology.

The methodology must be documented.

---

# 12. Data Layer

Potential database:

PostgreSQL

Entities:

```text
Match
Player
Team
Frame
Detection
Track
Event
Metric
Analysis
```

Example relationship:

```text
Match
 ├── Teams
 ├── Players
 ├── Events
 ├── Tracks
 └── Metrics
```

For the MVP, a simpler storage system such as JSON/Parquet may be sufficient.

Do not introduce PostgreSQL merely for the sake of using a database.

---

# 13. LLM Layer

The LLM should be downstream of the analytics engine.

```text
Analytics
     ↓
Structured Evidence
     ↓
Prompt Builder
     ↓
LLM
     ↓
Response Validation
     ↓
User
```

The LLM should receive:

* Match context
* Team statistics
* Player statistics
* Detected events
* Relevant tactical metrics
* User question

It should NOT receive fabricated statistics.

---

# 14. LLM Grounding

Example:

```json
{
  "team_a": {
    "possession": 42,
    "progressive_passes": 18,
    "central_receipts": 12
  },
  "team_b": {
    "possession": 58,
    "compactness": 0.72
  }
}
```

The LLM can interpret these values.

It should not invent:

```text
"Team A made 43 progressive passes"
```

if the analytics engine did not provide that value.

---

# 15. Optional RAG Layer

RAG should be added only when there is a clear use case.

Potential use:

```text
Football Knowledge Base
        ↓
Embeddings
        ↓
Vector Database
        ↓
Relevant Context
        ↓
LLM
```

Potential knowledge:

* Football tactical concepts
* Definitions
* Analytics methodology
* Team/player context
* Historical information

RAG is NOT required for basic match analysis.

---

# 16. Job Processing

Video analysis can be computationally expensive.

The architecture should eventually support asynchronous processing:

```text
Upload
 ↓
Create Job
 ↓
Queue
 ↓
AI Processing
 ↓
Store Results
 ↓
Notify Frontend
```

For the MVP, synchronous processing may be acceptable for short videos.

A queue should be introduced when required.

---

# 17. Project Folder Structure

Recommended structure:

```text
footballiq/
│
├── README.md
├── PROJECT_SPEC.md
├── ARCHITECTURE.md
├── DEVELOPMENT_PLAN.md
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
│
├── backend/
│   ├── main.py
│   ├── api/
│   ├── services/
│   ├── schemas/
│   └── config/
│
├── cv/
│   ├── detection/
│   ├── tracking/
│   ├── team_classification/
│   ├── pitch_mapping/
│   └── visualization/
│
├── analytics/
│   ├── player_metrics/
│   ├── team_metrics/
│   ├── events/
│   └── tactical/
│
├── ai/
│   ├── llm/
│   ├── prompts/
│   ├── rag/
│   └── validation/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── models/
│
├── frontend/
│
├── tests/
│
├── scripts/
│
└── docs/
    ├── architecture/
    ├── evaluation/
    └── screenshots/
```

---

# 18. Configuration

All configurable values should be externalized.

Examples:

```text
MODEL_PATH
VIDEO_SAMPLE_RATE
CONFIDENCE_THRESHOLD
TRACKER_TYPE
LLM_MODEL
DATABASE_URL
MAX_VIDEO_SIZE
```

Never hard-code secrets.

---

# 19. Error Handling

The system should gracefully handle:

* Invalid videos
* Missing model files
* Poor-quality footage
* No detected players
* Ball detection failure
* Tracking failures
* LLM API failures
* GPU unavailable
* Large videos
* Unsupported formats

The system should report meaningful errors.

---

# 20. Evaluation Architecture

Each AI component should have its own evaluation.

```text
Detection
→ Precision / Recall / mAP

Tracking
→ IDF1 / MOTA / ID switches

Classification
→ Accuracy / Precision / Recall / F1

Event Detection
→ Precision / Recall / F1

Regression Metrics
→ MAE / RMSE

System
→ Latency / FPS / Memory
```

Do not report metrics without actually measuring them.

---

# 21. Deployment

Potential deployment architecture:

```text
Frontend
   ↓
Backend API
   ↓
AI Processing Service
   ↓
Storage / Database
```

Docker should be used to make the environment reproducible.

GPU deployment should be considered if the selected models require it.

---

# 22. Architectural Rule

Every component must have a clear responsibility.

Avoid:

```text
one giant Python file
```

Prefer:

```text
small modular components
+
clear interfaces
+
testable functions
```

The architecture should be simple enough for a student to understand and explain.
