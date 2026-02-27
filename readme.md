# AI-MINDS

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-Microservices-009688?logo=fastapi&logoColor=white)](#)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316)](#)
[![RLM](https://img.shields.io/badge/Agentic-REPL%20Tracing-111827)](#)
[![Local First](https://img.shields.io/badge/Privacy-Local--First-22C55E)](#)

AI-MINDS is a local-first, multi-agent intelligence platform for analyzing files, building persistent memory, and answering complex queries through a REPL-driven reasoning loop.

## Table Of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Project Structure](#project-structure)
4. [Features](#features)
5. [Tech Stack](#tech-stack)
6. [Quick Start](#quick-start)
7. [API Domains](#api-domains)
8. [RLM Tracing and Logs](#rlm-tracing-and-logs)
9. [Screenshots and Demos](#screenshots-and-demos)
10. [Testing](#testing)
11. [Roadmap Ideas](#roadmap-ideas)
12. [License](#license)

## Overview

Most assistants answer one prompt and forget the rest. AI-MINDS is designed for iterative workflows:
- Ingest and parse real files (CSV, JSON, PDF, DOCX, XLSX, TXT)
- Run agentic analysis with traceable steps
- Persist useful context in memory
- Build and visualize a knowledge graph
- Serve everything through API + UI

## Core Architecture

High-level flow (aligned with your diagrams):

1. User query enters root orchestration.
2. Root agent delegates to specialized sub-agents in sandboxed execution.
3. Retrieval + tool agents parse files and run analysis.
4. Memory + knowledge graph enrich future decisions.
5. Testing agent validates behavior (LangWatch scenarios).
6. Final answer is returned with execution trace.

<details>
<summary><strong>Architecture Deep Dive</strong></summary>

- Root orchestration receives query and loads root prompt/context.
- Sub-agent 1 handles advisory reasoning tasks.
- Sub-agent 2 handles retrieval over parsed sources and memory.
- Sub-agent 3 handles tool execution and filesystem operations.
- RLM REPL executes code blocks and tool calls iteratively.
- Memory layer stores episodic and working memory state.
- Knowledge graph layer stores extracted semantic relations.

</details>

## Project Structure

```text
AI-MINDS/
  backend/
    api/                    # API gateway + routers + schemas + services
    rlm/                    # Recursive LLM engine, agents, REPL, tracing
    data/                   # Memory + graph artifacts
    logs/                   # Backend RLM traces
    requirements.txt
  frontend/
    ui/tabs/                # Gradio tabs (budget/files/memory/pipeline/etc.)
    logs/                   # Frontend RLM traces
    data/                   # Frontend-local memory data
    app.py                  # Main frontend entrypoint
  readme.md
```

## Features

- Multi-service API gateway (`/api/v1/*`)
- Budget analysis and affordability checks
- File listing + parsing + content-level extraction
- RLM REPL execution with filesystem tool-calling
- Prompt chunking pipeline (worker sub-agents + aggregator)
- Persistent memory retrieval and storage
- Knowledge graph generation and search
- Built-in backend test triggers from UI
- JSONL tracing for every RLM turn

## Tech Stack

- Python
- FastAPI + Uvicorn
- Gradio
- Ollama-compatible LLM endpoints
- Daytona sandbox integration
- Mem0-style memory store (local)
- Knowledge graph tooling (including HTML visualization)

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd AI-MINDS
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Configure environment

Create `backend/.env` (or update existing) with at least:

```env
RLM_API_URL=http://localhost:11434/v1
RLM_API_KEY=ollama
RLM_ROOT_MODEL=phi4-mini:latest
RLM_WORKER_MODEL=qwen2.5:3b
RLM_SUB_MODEL=qwen2.5:3b

RLM_CHUNK_SIZE=24000
RLM_CHUNK_OVERLAP=300
RLM_MAX_WORKERS=2
RLM_MAX_ITERATIONS=15
RLM_LOG_DIR=logs
```

Optional for Daytona:

```env
DAYTONA_API_KEY=
DAYTONA_API_URL=
RLM_ALLOWED_ROOTS=/workspace,/tmp/rlm
```

### 3. Run backend API gateway

```bash
cd backend
python api_server.py
```

Gateway:
- `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

### 4. Run frontend

```bash
cd frontend
python app.py
```

UI:
- `http://127.0.0.1:7860`

## API Domains

Main routers under `/api/v1`:
- `/budget`
- `/files`
- `/memory`
- `/pipeline`
- `/knowledge-graph`
- `/repl`
- `/testing`

Health endpoint:
- `/health`

## RLM Tracing and Logs

RLM sessions are recorded as JSONL traces, including:
- turn/iteration metadata
- model messages
- code blocks executed in REPL
- tool calls and outputs
- partial/final responses

Typical log locations:
- `backend/logs/rlm_trace_*.jsonl`
- `frontend/logs/rlm_trace_*.jsonl`

Example trace file:
- `frontend/logs/rlm_trace_20260215_043705_096693.jsonl`

## Screenshots and Demos

Place your images in `docs/assets/` using the names below.

### Architecture

![Architecture Diagram](docs/assets/architecture-main.png)
![Architecture Diagram (Detailed)](docs/assets/architecture-detailed.png)

### RLM Processing Logs

![RLM Log - Discovery and Detection](docs/assets/rlm-log-step-1.png)
![RLM Log - Field Filling Progress](docs/assets/rlm-log-step-2.png)
![RLM Log - Completion Summary](docs/assets/rlm-log-step-3.png)

### App Snapshots

![Filled Form Output](docs/assets/app-filled-form.png)
![Extracted JSON Output](docs/assets/app-extracted-data.png)
![Budget CSV Example](docs/assets/app-budget-csv.png)
![Budget Q&A Result](docs/assets/app-budget-qa.png)

<details>
<summary><strong>Asset Upload Checklist</strong></summary>

- `docs/assets/architecture-main.png`
- `docs/assets/architecture-detailed.png`
- `docs/assets/rlm-log-step-1.png`
- `docs/assets/rlm-log-step-2.png`
- `docs/assets/rlm-log-step-3.png`
- `docs/assets/app-filled-form.png`
- `docs/assets/app-extracted-data.png`
- `docs/assets/app-budget-csv.png`
- `docs/assets/app-budget-qa.png`

</details>

## Testing

Run backend test suites from root:

```bash
cd backend
python -m pytest rlm/test_budget_advisor.py -v
python -m pytest rlm/test_rlm_repl.py -v
```

Or run from the frontend `Testing` tab.

## Roadmap Ideas

- Rich trace viewer in UI (timeline + tool-call replay)
- Better source-citation grounding in final answers
- Multi-user memory isolation and export/import
- Streaming agent steps and live cost telemetry

## License

Add your license here (MIT, Apache-2.0, etc.).
