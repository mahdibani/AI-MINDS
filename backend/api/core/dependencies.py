from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Tuple

from budget_advisor import BudgetAdvisor
from rlm.fs_tools import FSTools
from rlm.memory import AgentMemory
from rlm.agents import PromptPipeline


@dataclass
class ServiceContainer:
    budget_lock: Lock = field(default_factory=Lock)
    budget_advisors: Dict[Tuple[str, float, str], BudgetAdvisor] = field(default_factory=dict)
    memory_clients: Dict[str, AgentMemory] = field(default_factory=dict)
    pipeline_clients: Dict[Tuple[int, int], PromptPipeline] = field(default_factory=dict)
    knowledge_graphs: Dict[str, object] = field(default_factory=dict)
    knowledge_lock: Lock = field(default_factory=Lock)

    def get_budget_advisor(self, budget_file: str, safety_buffer: float, user_id: str) -> BudgetAdvisor:
        key = (budget_file, safety_buffer, user_id)
        with self.budget_lock:
            if key not in self.budget_advisors:
                self.budget_advisors[key] = BudgetAdvisor(
                    budget_file=budget_file,
                    user_id=user_id,
                    safety_buffer=safety_buffer,
                )
            return self.budget_advisors[key]

    def get_memory(self, user_id: str) -> AgentMemory:
        if user_id not in self.memory_clients:
            self.memory_clients[user_id] = AgentMemory(user_id=user_id)
        return self.memory_clients[user_id]

    def get_pipeline(self, chunk_size: int, max_workers: int) -> PromptPipeline:
        key = (chunk_size, max_workers)
        if key not in self.pipeline_clients:
            self.pipeline_clients[key] = PromptPipeline(chunk_size=chunk_size, max_workers=max_workers)
        return self.pipeline_clients[key]

    def put_knowledge_graph(self, graph_id: str, graph: object) -> None:
        with self.knowledge_lock:
            self.knowledge_graphs[graph_id] = graph

    def get_knowledge_graph(self, graph_id: str) -> object | None:
        with self.knowledge_lock:
            return self.knowledge_graphs.get(graph_id)

    @staticmethod
    def get_fs_tools(allowed_roots: list[str]) -> FSTools:
        return FSTools(allowed_roots=allowed_roots)


container = ServiceContainer()
