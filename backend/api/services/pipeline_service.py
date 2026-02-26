from __future__ import annotations

from api.core.dependencies import container


def run_pipeline(prompt: str, task: str, chunk_size: int, max_workers: int, parallel: bool) -> dict:
    pipeline = container.get_pipeline(chunk_size=chunk_size, max_workers=max_workers)
    result = pipeline.run(prompt=prompt, task=task, parallel=parallel)
    return {
        "final_output": result.final_output,
        "chunks_processed": result.chunks_processed,
        "total_tokens": result.total_tokens,
        "total_cost": result.total_cost,
        "elapsed_total": result.elapsed_total,
    }

