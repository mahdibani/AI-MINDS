# AI-MINDS Microservices Backend

## Services

- `budget` service: affordability checks and budget summaries.
- `files` service: directory listing, parsing, and RLM-powered analysis.
- `memory` service: add/search/list/clear memory entries.
- `pipeline` service: large prompt chunking + worker/aggregator orchestration.
- `knowledge-graph` service: graph build, visualization export, and graph search.
- `repl` service: sandboxed RLM code execution and directory task execution.
- `testing` service: run backend pytest suites.

## Run Gateway

From `backend/`:

```bash
python api_server.py
```

Gateway URL: `http://127.0.0.1:8000`
Swagger: `http://127.0.0.1:8000/docs`

## Run Individual Services

```bash
uvicorn api.apps.budget_app:app --host 127.0.0.1 --port 8101
uvicorn api.apps.files_app:app --host 127.0.0.1 --port 8102
uvicorn api.apps.memory_app:app --host 127.0.0.1 --port 8103
uvicorn api.apps.pipeline_app:app --host 127.0.0.1 --port 8104
uvicorn api.apps.knowledge_graph_app:app --host 127.0.0.1 --port 8105
uvicorn api.apps.repl_app:app --host 127.0.0.1 --port 8106
uvicorn api.apps.testing_app:app --host 127.0.0.1 --port 8107
```
