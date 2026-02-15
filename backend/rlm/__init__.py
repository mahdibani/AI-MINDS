"""
RLM – Recursive Language Model package.

Quick-start:
    from rlm.rlm_repl import RLM_REPL          # core RLM with REPL
    from rlm.agents import PromptPipeline        # sub-agent pipeline
    from rlm.agents import process_prompt        # one-liner convenience wrapper
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union


class RLM(ABC):
    """Abstract base class defining the RLM interface."""

    @abstractmethod
    def completion(
        self,
        context: Union[List[str], str, List[Dict[str, str]]],
        query: str,
    ) -> str:
        """
        Generate a completion for the given query and context.

        Args:
            context: The context to process (can be very long).
            query:   The query/question to answer.

        Returns:
            The final answer as a string.
        """

    @abstractmethod
    def cost_summary(self) -> Dict[str, Any]:
        """
        Return a cost breakdown dict with keys:
            total_cost, root_llm_cost, sub_llm_cost,
            root_llm_tokens, sub_llm_tokens,
            root_llm_calls, sub_llm_calls
        """

    @abstractmethod
    def reset(self):
        """Reset the RLM state for a new task."""


__all__ = ["RLM"]