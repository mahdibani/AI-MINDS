"""
Detailed JSONL tracing for RLM sessions.

Each turn is appended to logs/rlm_trace_<session_id>.jsonl.
The tracer is a module-level singleton so every component shares it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class RLMDetailedTracer:
    """Append-only JSONL tracer for an RLM session."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir    = log_dir
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.log_file   = os.path.join(log_dir, f"rlm_trace_{self.session_id}.jsonl")
        self.turn_count = 0
        os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def log_turn(
        self,
        iteration: int,
        messages: List[Dict[str, str]],
        response: str,
        code_blocks: List[str],
        execution_results: List[str],
        final_answer: Optional[str] = None,
        repl_state: Optional[Dict[str, Any]] = None,
        cost_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one RLM turn to the JSONL log file."""
        record = {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "session_id":       self.session_id,
            "turn":             self.turn_count,
            "iteration":        iteration,
            "messages":         messages,
            "response":         response,
            "code_blocks":      code_blocks,
            "execution_results": execution_results,
            "final_answer":     final_answer,
            "repl_state":       repl_state or {},
            "cost_info":        cost_info or {},
        }
        self._write(record)
        self.turn_count += 1

    def log_error(self, error: str, context: str = "") -> None:
        """Append an error record."""
        record = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "turn":       self.turn_count,
            "error":      error,
            "context":    context,
        }
        self._write(record)

    def log_pipeline_event(self, event: str, data: Dict[str, Any]) -> None:
        """Log a pipeline-level event (chunking, aggregation, etc.)."""
        record = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "event":      event,
            "data":       data,
        }
        self._write(record)

    def get_log_path(self) -> str:
        return self.log_file

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, record: Dict[str, Any]) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass  # never crash the main loop due to logging failure


# Module-level singleton – import and use directly:
#   from rlm.utils.tracing import tracer
tracer = RLMDetailedTracer()