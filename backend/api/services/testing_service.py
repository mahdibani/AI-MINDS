from __future__ import annotations

import subprocess
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]


def _run_pytest(target_file: Path, verbose: bool) -> dict:
    cmd = ["pytest", str(target_file), "-v"]
    if verbose:
        cmd.append("-s")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BACKEND_DIR))
    return {
        "command": " ".join(cmd),
        "return_code": result.returncode,
        "output": f"{result.stdout}\n{result.stderr}",
    }


def run_budget_tests(verbose: bool) -> dict:
    candidates = [
        BACKEND_DIR / "rlm" / "test_budget_advisor.py",
        BACKEND_DIR / "test_budget_advisor.py",
    ]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        return {"command": "", "return_code": 1, "output": "Budget test file not found"}
    return _run_pytest(target, verbose)


def run_rlm_tests(verbose: bool) -> dict:
    candidates = [
        BACKEND_DIR / "rlm" / "test_rlm_repl.py",
        BACKEND_DIR / "test_rlm_repl.py",
    ]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        return {"command": "", "return_code": 1, "output": "RLM test file not found"}
    return _run_pytest(target, verbose)

