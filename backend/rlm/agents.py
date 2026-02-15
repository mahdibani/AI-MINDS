"""
Sub-agent pipeline for RLM.

Architecture
------------
                        ┌──────────────────────────────────────┐
  huge_prompt (str)     │           PromptPipeline             │
  + task_instruction ──►│                                       │
                        │  1. ChunkingAgent   → chunks[]        │
                        │  2. WorkerAgent×N   → partial[]       │
                        │  3. AggregatorAgent → final_output    │
                        └──────────────────────────────────────┘

Usage
-----
    from rlm.agents import PromptPipeline

    pipeline = PromptPipeline()          # reads config from .env automatically
    result = pipeline.run(
        prompt=my_huge_text,
        task="Extract every action item and return them as a numbered list."
    )
    print(result.final_output)
    print(result.cost_summary())
"""

from __future__ import annotations

import os
import math
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rlm.utils.llm import get_llm_client


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    index: int
    text: str
    char_start: int
    char_end: int


@dataclass
class WorkerResult:
    chunk_index: int
    partial_output: str
    tokens_used: int
    cost: float
    elapsed: float


@dataclass
class PipelineResult:
    final_output: str
    chunks_processed: int
    worker_results: List[WorkerResult]
    aggregator_tokens: int
    aggregator_cost: float
    total_tokens: int
    total_cost: float
    elapsed_total: float

    def cost_summary(self) -> str:
        return (
            f"Chunks processed : {self.chunks_processed}\n"
            f"Total tokens     : {self.total_tokens:,}\n"
            f"Total cost       : ${self.total_cost:.6f}\n"
            f"Elapsed          : {self.elapsed_total:.1f}s"
        )


# ---------------------------------------------------------------------------
# Individual agents
# ---------------------------------------------------------------------------

class ChunkingAgent:
    """
    Splits a large prompt into overlapping chunks so no context is lost
    at chunk boundaries.

    Strategy:
      - Target chunk size read from env: RLM_CHUNK_SIZE (chars, default 80_000)
      - Overlap read from env: RLM_CHUNK_OVERLAP (chars, default 500)
      - Tries to split on paragraph boundaries (double-newline) when possible.
    """

    def __init__(
        self,
        chunk_size: int = 0,
        overlap: int = 0,
    ):
        self.chunk_size  = chunk_size  or int(os.getenv("RLM_CHUNK_SIZE",  "80000"))
        self.overlap     = overlap     or int(os.getenv("RLM_CHUNK_OVERLAP", "500"))

    def split(self, text: str) -> List[ChunkResult]:
        """Split *text* into chunks, returning ChunkResult list."""
        if len(text) <= self.chunk_size:
            return [ChunkResult(0, text, 0, len(text))]

        chunks: List[ChunkResult] = []
        start = 0
        idx   = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break on a paragraph boundary within the last 20 % of the window
            if end < len(text):
                search_from = start + int(self.chunk_size * 0.8)
                para_break  = text.rfind("\n\n", search_from, end)
                if para_break != -1:
                    end = para_break + 2   # include the newlines

            chunks.append(ChunkResult(idx, text[start:end], start, end))

            # Advance with overlap so boundaries don't lose context
            start = end - self.overlap if end - self.overlap > start else end
            idx  += 1

        return chunks


class WorkerAgent:
    """
    Processes a single chunk against the user's task instruction.

    Each WorkerAgent call is a single LLM completion:
        system: worker persona
        user  : task + chunk
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
        You are a focused analysis agent. You will receive a TASK and a CHUNK
        of a larger document. Your job is to perform the task on this chunk only.
        Be precise and concise. If the chunk does not contain information
        relevant to the task, reply with exactly: NO_RELEVANT_CONTENT
    """)

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("RLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model   = model   or os.getenv("RLM_WORKER_MODEL") or os.getenv("RLM_SUB_MODEL", "gpt-5-mini")
        self._llm    = get_llm_client(self.api_key, self.model)

    def process(self, task: str, chunk: ChunkResult) -> WorkerResult:
        t0 = time.time()
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"TASK:\n{task}\n\n"
                    f"CHUNK {chunk.index + 1} "
                    f"(chars {chunk.char_start}–{chunk.char_end}):\n"
                    f"{chunk.text}"
                ),
            },
        ]
        response, cost_info = self._llm.completion_with_cost(messages)
        return WorkerResult(
            chunk_index    = chunk.index,
            partial_output = response.strip(),
            tokens_used    = cost_info["tokens"],
            cost           = cost_info["cost"],
            elapsed        = time.time() - t0,
        )


class AggregatorAgent:
    """
    Takes all worker partial outputs and synthesises the final answer.

    Receives:
      - original task instruction
      - all non-empty worker results (NO_RELEVANT_CONTENT filtered out)
      - optional: the first 2 000 chars of the original prompt as context hint
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
        You are a synthesis agent. You receive a TASK and a set of PARTIAL
        RESULTS produced by worker agents that each analysed one chunk of a
        large document. Your job is to merge, de-duplicate, and synthesise
        these partial results into a single, coherent, final answer to the
        task. Be thorough but do not repeat yourself.
    """)

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("RLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model   = model   or os.getenv("RLM_ROOT_MODEL", "gpt-5")
        self._llm    = get_llm_client(self.api_key, self.model)

    def aggregate(
        self,
        task: str,
        worker_results: List[WorkerResult],
        prompt_hint: str = "",
    ) -> tuple[str, Dict[str, Any]]:
        """
        Aggregate worker results into a final answer.

        Returns (final_output_str, cost_info_dict).
        """
        # Filter out empty / no-relevant chunks
        relevant = [
            wr for wr in worker_results
            if wr.partial_output and wr.partial_output != "NO_RELEVANT_CONTENT"
        ]

        if not relevant:
            return "No relevant content found across all chunks.", {"cost": 0, "tokens": 0}

        partials_text = "\n\n".join(
            f"--- Partial result from chunk {wr.chunk_index + 1} ---\n{wr.partial_output}"
            for wr in relevant
        )

        hint_section = (
            f"\nORIGINAL PROMPT (first 2000 chars for context):\n{prompt_hint[:2000]}\n"
            if prompt_hint else ""
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"TASK:\n{task}\n"
                    f"{hint_section}\n"
                    f"PARTIAL RESULTS ({len(relevant)} of {len(worker_results)} chunks had relevant content):\n\n"
                    f"{partials_text}"
                ),
            },
        ]

        response, cost_info = self._llm.completion_with_cost(messages)
        return response.strip(), cost_info


# ---------------------------------------------------------------------------
# PromptPipeline – orchestrates the full flow
# ---------------------------------------------------------------------------

class PromptPipeline:
    """
    End-to-end pipeline: huge prompt → sub-agents → final output.

    Configuration (via .env / environment variables):
        RLM_API_URL          LLM server base URL      (default: http://localhost:8080/v1)
        RLM_ROOT_MODEL       Aggregator model name     (default: gpt-5)
        RLM_WORKER_MODEL     Worker model name         (default: gpt-5-mini)
        RLM_CHUNK_SIZE       Chars per chunk           (default: 80000)
        RLM_CHUNK_OVERLAP    Overlap between chunks    (default: 500)
        RLM_MAX_WORKERS      Parallel worker threads   (default: 4)
        RLM_API_KEY          API key (or OPENAI_API_KEY)

    Example:
        pipeline = PromptPipeline()
        result = pipeline.run(
            prompt=huge_text,
            task="Summarise the key findings."
        )
        print(result.final_output)
    """

    def __init__(
        self,
        api_key:       Optional[str] = None,
        root_model:    Optional[str] = None,
        worker_model:  Optional[str] = None,
        chunk_size:    int = 0,
        overlap:       int = 0,
        max_workers:   int = 0,
    ):
        self.api_key      = api_key     or os.getenv("RLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.root_model   = root_model  or os.getenv("RLM_ROOT_MODEL",   "gpt-5")
        self.worker_model = worker_model or os.getenv("RLM_WORKER_MODEL", "gpt-5-mini")
        self.max_workers  = max_workers or int(os.getenv("RLM_MAX_WORKERS", "4"))

        self.chunker    = ChunkingAgent(chunk_size=chunk_size, overlap=overlap)
        self.aggregator = AggregatorAgent(api_key=self.api_key, model=self.root_model)

    def _make_worker(self) -> WorkerAgent:
        return WorkerAgent(api_key=self.api_key, model=self.worker_model)

    def run(
        self,
        prompt: str,
        task: str,
        parallel: bool = True,
    ) -> PipelineResult:
        """
        Run the full pipeline.

        Args:
            prompt:   The large text to process (stored as a variable, never
                      passed whole to any single LLM call).
            task:     Instruction for what to do with the prompt (e.g.
                      "Extract all action items as a numbered list.").
            parallel: Whether to run workers in parallel threads (default True).

        Returns:
            PipelineResult with final_output and detailed cost/timing stats.
        """
        t0 = time.time()
        print(f"[pipeline] Prompt length: {len(prompt):,} chars")

        # ── 1. Chunking ────────────────────────────────────────────────────
        chunks = self.chunker.split(prompt)
        print(f"[pipeline] Split into {len(chunks)} chunk(s) "
              f"(target size: {self.chunker.chunk_size:,} chars, "
              f"overlap: {self.chunker.overlap:,} chars)")

        # ── 2. Worker agents ───────────────────────────────────────────────
        worker_results: List[WorkerResult] = []

        if parallel and len(chunks) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(self._make_worker().process, task, chunk): chunk
                    for chunk in chunks
                }
                for future in as_completed(futures):
                    wr = future.result()
                    worker_results.append(wr)
                    status = (
                        "✓" if wr.partial_output != "NO_RELEVANT_CONTENT" else "–"
                    )
                    print(
                        f"[worker {wr.chunk_index + 1}/{len(chunks)}] {status}  "
                        f"{wr.tokens_used} tokens  ${wr.cost:.5f}  {wr.elapsed:.1f}s"
                    )
            # Sort back into chunk order
            worker_results.sort(key=lambda r: r.chunk_index)
        else:
            worker = self._make_worker()
            for chunk in chunks:
                wr = worker.process(task, chunk)
                worker_results.append(wr)
                status = "✓" if wr.partial_output != "NO_RELEVANT_CONTENT" else "–"
                print(
                    f"[worker {wr.chunk_index + 1}/{len(chunks)}] {status}  "
                    f"{wr.tokens_used} tokens  ${wr.cost:.5f}  {wr.elapsed:.1f}s"
                )

        # ── 3. Aggregator ──────────────────────────────────────────────────
        print("[pipeline] Aggregating results...")
        final_output, agg_cost = self.aggregator.aggregate(
            task           = task,
            worker_results = worker_results,
            prompt_hint    = prompt,
        )

        # ── 4. Build result ────────────────────────────────────────────────
        worker_tokens = sum(wr.tokens_used for wr in worker_results)
        worker_cost   = sum(wr.cost        for wr in worker_results)
        total_tokens  = worker_tokens + agg_cost.get("tokens", 0)
        total_cost    = worker_cost   + agg_cost.get("cost",   0.0)

        result = PipelineResult(
            final_output       = final_output,
            chunks_processed   = len(chunks),
            worker_results     = worker_results,
            aggregator_tokens  = agg_cost.get("tokens", 0),
            aggregator_cost    = agg_cost.get("cost",   0.0),
            total_tokens       = total_tokens,
            total_cost         = total_cost,
            elapsed_total      = time.time() - t0,
        )
        print(f"[pipeline] Done.\n{result.cost_summary()}")
        return result


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def process_prompt(
    prompt: str,
    task: str,
    *,
    api_key:      Optional[str] = None,
    root_model:   Optional[str] = None,
    worker_model: Optional[str] = None,
    chunk_size:   int = 0,
    overlap:      int = 0,
    max_workers:  int = 0,
    parallel:     bool = True,
) -> str:
    """
    One-liner convenience wrapper.

    Returns the final output string.

    Example:
        from rlm.agents import process_prompt

        output = process_prompt(
            prompt=my_huge_text,
            task="List every date mentioned in the text."
        )
        print(output)
    """
    pipeline = PromptPipeline(
        api_key      = api_key,
        root_model   = root_model,
        worker_model = worker_model,
        chunk_size   = chunk_size,
        overlap      = overlap,
        max_workers  = max_workers,
    )
    result = pipeline.run(prompt=prompt, task=task, parallel=parallel)
    return result.final_output