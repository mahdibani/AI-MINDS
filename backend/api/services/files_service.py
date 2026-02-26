from __future__ import annotations

from pathlib import Path

from api.core.dependencies import container
from rlm.rlm_repl import RLM_REPL


def list_files(directory: str) -> dict:
    directory = directory.replace("\\", "/")
    fs = container.get_fs_tools([directory, "/tmp/rlm_scratch", str(Path.home())])
    return fs.list(directory)


def parse_file(filepath: str, max_chars: int) -> dict:
    filepath = filepath.replace("\\", "/")
    root = str(Path(filepath).parent)
    fs = container.get_fs_tools([root, "/tmp/rlm_scratch", str(Path.home())])
    return fs.parse(filepath, max_chars=max_chars)


def analyze_files(directory: str, query: str, max_iterations: int) -> str:
    directory = directory.replace("\\", "/")
    rlm = RLM_REPL(
        allowed_roots=[directory, "/tmp/rlm_scratch"],
        max_iterations=max_iterations,
        extra_locals={"target_dir": directory},
    )
    analysis_query = f"""
Analyze files in 'target_dir' and answer this request:
{query}

Use fs_list(target_dir) to inspect files and fs_parse(file_path) to read parseable files.
Return a concise grounded answer and end with FINAL(your_answer).
"""
    return rlm.completion(context=f"File analysis in {directory}", query=analysis_query) or ""

