from __future__ import annotations

from pathlib import Path

from rlm.rlm_repl import RLM_REPL


def run_repl_code(code: str, context: str, max_iterations: int) -> str:
    rlm = RLM_REPL(
        allowed_roots=[str(Path.home() / "Downloads"), "/tmp/rlm_scratch"],
        max_iterations=max_iterations,
    )
    query = f"""
Execute this code:

```python
{code}
```

FINAL(result of execution)
"""
    return rlm.completion(context=context or "REPL execution", query=query) or ""


def run_repl_file_task(directory: str, task: str, max_iterations: int) -> str:
    directory = directory.replace("\\", "/")
    rlm = RLM_REPL(
        allowed_roots=[directory, "/tmp/rlm_scratch"],
        max_iterations=max_iterations,
        extra_locals={"work_dir": directory},
    )
    query = f"""
Task: {task}

Working directory is stored in 'work_dir'.
Use fs_list(work_dir), fs_parse(file_path), and llm_query(prompt) as needed.
Complete the task and emit FINAL(your_answer).
"""
    return rlm.completion(context="File analysis task", query=query) or ""

