from __future__ import annotations

import os
from typing import Any

import requests


class ApiClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("AIMINDS_API_URL", "http://127.0.0.1:8000")).rstrip("/")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}{path}", json=payload, timeout=300)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def budget_check(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/budget/check", payload)

    def budget_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/budget/summary", payload)

    def files_list(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/files/list", payload)

    def files_parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/files/parse", payload)

    def files_analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/files/analyze", payload)

    def memory_add(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/memory/add", payload)

    def memory_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/memory/search", payload)

    def memory_list(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/memory/list", payload)

    def memory_clear(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/memory/clear", payload)

    def pipeline_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/pipeline/run", payload)

    def knowledge_graph_build(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/knowledge-graph/build", payload)

    def knowledge_graph_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/knowledge-graph/search", payload)

    def repl_code(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/repl/code", payload)

    def repl_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/repl/task", payload)

    def testing_budget(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/testing/budget", payload)

    def testing_rlm(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/testing/rlm", payload)
