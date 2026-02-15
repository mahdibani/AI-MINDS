"""
REPL environment for RLM – enhanced with:
  - fs_parse() injected alongside the existing fs_* helpers
  - Windows path normalisation (delegates to FSTools._resolve)
  - Cleaner FINAL_VAR handling
"""

import sys
import io
import threading
import json
import tempfile
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List

from rlm.fs_tools import FSTools


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float

    def __str__(self):
        return (f"REPLResult(stdout={self.stdout[:200]}, "
                f"stderr={self.stderr[:200]}, "
                f"execution_time={self.execution_time:.2f}s)")


class REPLEnv:
    """
    REPL environment providing:
      - llm_query()                   – recursive sub-LLM calls
      - fs_list / fs_read / fs_write  – sandboxed directory ops
      - fs_parse()                    – multi-format text extraction (NEW)
      - fs_exists / fs_info           – metadata helpers
      - FINAL_VAR()                   – terminate with a variable value
    """

    def __init__(
        self,
        llm_query_fn: Callable[[str], str],
        context_json: Optional[Dict[str, Any] | List[Any]] = None,
        context_str: Optional[str] = None,
        allowed_roots: Optional[List[str]] = None,
    ):
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")
        self._fs = FSTools(allowed_roots=allowed_roots)

        # Safe built-ins
        self.globals = {
            '__builtins__': {
                'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
                'type': type, 'isinstance': isinstance, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
                'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
                'chr': chr, 'ord': ord, 'hex': hex, 'bin': bin, 'oct': oct,
                'repr': repr, 'ascii': ascii, 'format': format,
                '__import__': __import__,
                'open': open,
                'range': range, 'reversed': reversed, 'slice': slice,
                'iter': iter, 'next': next, 'pow': pow, 'divmod': divmod,
                'any': any, 'all': all, 'hasattr': hasattr, 'getattr': getattr,
                'setattr': setattr, 'delattr': delattr, 'dir': dir, 'vars': vars,
                'complex': complex, 'bytes': bytes, 'bytearray': bytearray,
                'memoryview': memoryview, 'hash': hash, 'id': id, 'callable': callable,
                'issubclass': issubclass, 'super': super, 'property': property,
                'staticmethod': staticmethod, 'classmethod': classmethod,
                'object': object,
                'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
                'KeyError': KeyError, 'IndexError': IndexError, 'AttributeError': AttributeError,
                'FileNotFoundError': FileNotFoundError, 'OSError': OSError, 'IOError': IOError,
                'RuntimeError': RuntimeError, 'NameError': NameError, 'ImportError': ImportError,
                'StopIteration': StopIteration, 'AssertionError': AssertionError,
                'NotImplementedError': NotImplementedError,
            }
        }
        self.locals = {}
        self._lock = threading.Lock()

        # ── Inject llm_query ──────────────────────────────────────────────
        self.globals['llm_query'] = llm_query_fn

        # ── Inject filesystem helpers ─────────────────────────────────────
        fs = self._fs

        def _fs_list(path: str = ".") -> dict:
            """List a directory.  Each entry has: name, type, size, modified, parseable."""
            result = fs.list(path)
            if result["error"]:
                print(f"[fs_list] Error: {result['error']}")
            return result

        def _fs_read(path: str) -> str:
            """Read a plain-text file. For PDF/DOCX/XLSX use fs_parse()."""
            result = fs.read(path)
            if result["error"]:
                raise RuntimeError(f"fs_read('{path}'): {result['error']}")
            return result["content"]

        def _fs_parse(path: str, max_chars: int = 50_000) -> dict:
            """
            Extract text from any supported format:
            .pdf, .docx, .doc, .xlsx, .csv, .json, .txt, .md, …

            Returns dict with: path, text, format, size, truncated, error.

            Example:
                result = fs_parse('/downloads/report.pdf')
                if not result['error']:
                    summary = llm_query('Summarise: ' + result['text'])
            """
            result = fs.parse(path, max_chars=max_chars)
            if result["error"]:
                print(f"[fs_parse] Error: {result['error']}")
            elif result["truncated"]:
                print(f"[fs_parse] Text truncated to {max_chars} chars from {result['size']} bytes")
            return result

        def _fs_write(path: str, content: str, overwrite: bool = True) -> dict:
            """Write text to a file. Creates parent dirs automatically."""
            result = fs.write(path, content, overwrite=overwrite)
            if result["error"]:
                print(f"[fs_write] Error: {result['error']}")
            else:
                print(f"[fs_write] Wrote {result['written']} bytes to {result['path']}")
            return result

        def _fs_exists(path: str) -> bool:
            """Check whether a path exists."""
            result = fs.exists(path)
            return result["exists"]

        def _fs_info(path: str) -> dict:
            """Return metadata: size, modified timestamp, permissions."""
            result = fs.info(path)
            if result["error"]:
                print(f"[fs_info] Error: {result['error']}")
            return result

        self.globals['fs_list']   = _fs_list
        self.globals['fs_read']   = _fs_read
        self.globals['fs_parse']  = _fs_parse
        self.globals['fs_write']  = _fs_write
        self.globals['fs_exists'] = _fs_exists
        self.globals['fs_info']   = _fs_info

        # ── Inject FINAL_VAR ──────────────────────────────────────────────
        def final_var(variable_name: str) -> str:
            variable_name = variable_name.strip().strip('"').strip("'").strip()
            if variable_name in self.locals:
                return str(self.locals[variable_name])
            return f"Error: Variable '{variable_name}' not found in REPL environment"

        self.globals['FINAL_VAR'] = final_var

        # ── Load context ──────────────────────────────────────────────────
        self._load_context(context_json, context_str)

    # ------------------------------------------------------------------

    def _load_context(self, context_json=None, context_str=None):
        if context_json is not None:
            self.locals['context'] = context_json
        if context_str is not None:
            self.locals['context'] = context_str

    def __del__(self):
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # I/O capture
    # ------------------------------------------------------------------

    @contextmanager
    def _capture_output(self):
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_working_directory(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def code_execution(self, code: str) -> REPLResult:
        start_time = time.time()
        stdout_content = stderr_content = ""

        with self._capture_output() as (stdout_buf, stderr_buf):
            with self._temp_working_directory():
                try:
                    lines = code.split('\n')
                    import_lines = [l for l in lines if l.strip().startswith(('import ', 'from ')) and not l.strip().startswith('#')]
                    other_lines  = [l for l in lines if not (l.strip().startswith(('import ', 'from ')) and not l.strip().startswith('#'))]

                    if import_lines:
                        exec('\n'.join(import_lines), self.globals, self.globals)

                    if other_lines:
                        combined = {**self.globals, **self.locals}
                        non_comment = [l for l in other_lines if l.strip() and not l.strip().startswith('#')]

                        if non_comment:
                            last = non_comment[-1]
                            is_expr = (
                                not last.strip().startswith((
                                    'import ', 'from ', 'def ', 'class ',
                                    'if ', 'for ', 'while ', 'try:', 'with ',
                                    'return ', 'yield ', 'break', 'continue', 'pass'
                                )) and
                                '=' not in last.split('#')[0] and
                                not last.strip().endswith(':') and
                                not last.strip().startswith('print(')
                            )
                            if is_expr:
                                try:
                                    stmts_code = '\n'.join(other_lines[:-1])
                                    if stmts_code.strip():
                                        exec(stmts_code, combined, combined)
                                    result = eval(last, combined, combined)
                                    if result is not None:
                                        print(repr(result))
                                except Exception:
                                    exec('\n'.join(other_lines), combined, combined)
                            else:
                                exec('\n'.join(other_lines), combined, combined)
                        else:
                            exec('\n'.join(other_lines), combined, combined)

                        for k, v in combined.items():
                            if k not in self.globals:
                                self.locals[k] = v

                    stdout_content = stdout_buf.getvalue()
                    stderr_content = stderr_buf.getvalue()

                except Exception as e:
                    import traceback
                    stderr_content = stderr_buf.getvalue() + str(e) + "\n" + traceback.format_exc()
                    stdout_content = stdout_buf.getvalue()

        self.locals['_stdout'] = stdout_content
        self.locals['_stderr'] = stderr_content

        return REPLResult(
            stdout=stdout_content,
            stderr=stderr_content,
            locals=self.locals.copy(),
            execution_time=time.time() - start_time,
        )