"""
REPL environment for RLM with support for recursive LLM calls
and sandboxed filesystem access.
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
    """Result from REPL code execution."""
    stdout: str
    stderr: str
    locals: dict
    execution_time: float

    def __str__(self):
        return f"REPLResult(stdout={self.stdout}, stderr={self.stderr}, execution_time={self.execution_time})"


class REPLEnv:
    """
    REPL environment that executes Python code and provides access to:
      - recursive LLM calls via llm_query()
      - sandboxed filesystem access via fs_list(), fs_read(), fs_write(),
        fs_exists(), fs_info()

    Context is stored as an in-memory variable, not passed to the model directly.
    """

    def __init__(
        self,
        llm_query_fn: Callable[[str], str],
        context_json: Optional[Dict[str, Any] | List[Any]] = None,
        context_str: Optional[str] = None,
        allowed_roots: Optional[List[str]] = None,
    ):
        """
        Initialize REPL environment.

        Args:
            llm_query_fn:  Function to call for recursive LLM queries.
            context_json:  Context as JSON-serializable structure.
            context_str:   Context as plain string.
            allowed_roots: Whitelist of root directories for filesystem access.
                           Defaults to ['/workspace', '/tmp/rlm', '~/workspace'].
        """
        # Store original working directory
        self.original_cwd = os.getcwd()

        # Create temporary directory for file operations
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")

        # ------------------------------------------------------------------
        # Filesystem tool layer (sandboxed)
        # ------------------------------------------------------------------
        self._fs = FSTools(allowed_roots=allowed_roots)

        # ------------------------------------------------------------------
        # Build safe globals
        # ------------------------------------------------------------------
        self.globals = {
            '__builtins__': {
                # Safe built-ins
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
                # Exception classes
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

        # ------------------------------------------------------------------
        # Inject llm_query
        # ------------------------------------------------------------------
        self.globals['llm_query'] = llm_query_fn

        # ------------------------------------------------------------------
        # Inject sandboxed filesystem helpers
        # ------------------------------------------------------------------
        fs = self._fs  # capture for closures

        def _fs_list(path: str = ".") -> dict:
            """List a directory inside the sandboxed workspace.

            Returns a dict with keys: path, entries (list of {name,type,size,modified}), error.
            Example:
                result = fs_list('/workspace/src')
                for e in result['entries']:
                    print(e['name'], e['type'])
            """
            result = fs.list(path)
            if result["error"]:
                print(f"[fs_list] Error: {result['error']}")
            return result

        def _fs_read(path: str) -> str:
            """Read a text file from the sandboxed workspace.

            Returns the file content as a string, or raises RuntimeError on failure.
            Example:
                code = fs_read('/workspace/src/main.py')
                print(code[:500])
            """
            result = fs.read(path)
            if result["error"]:
                raise RuntimeError(f"fs_read('{path}'): {result['error']}")
            return result["content"]

        def _fs_write(path: str, content: str, overwrite: bool = True) -> dict:
            """Write text to a file in the sandboxed workspace.

            Creates parent directories automatically. Returns a result dict
            with keys: path, written (bytes), error.
            Example:
                fs_write('/workspace/output/result.txt', 'Hello world')
            """
            result = fs.write(path, content, overwrite=overwrite)
            if result["error"]:
                print(f"[fs_write] Error: {result['error']}")
            else:
                print(f"[fs_write] Wrote {result['written']} bytes to {result['path']}")
            return result

        def _fs_exists(path: str) -> bool:
            """Check whether a path exists inside the sandboxed workspace.

            Returns True/False (errors treated as False).
            Example:
                if fs_exists('/workspace/config.json'):
                    cfg = fs_read('/workspace/config.json')
            """
            result = fs.exists(path)
            return result["exists"]

        def _fs_info(path: str) -> dict:
            """Return metadata for a path: size, modified timestamp, permissions.

            Returns a dict with keys: path, size, modified, is_file, is_dir,
            permissions, error.
            Example:
                meta = fs_info('/workspace/data.csv')
                print(f"Size: {meta['size']} bytes")
            """
            result = fs.info(path)
            if result["error"]:
                print(f"[fs_info] Error: {result['error']}")
            return result

        self.globals['fs_list']   = _fs_list
        self.globals['fs_read']   = _fs_read
        self.globals['fs_write']  = _fs_write
        self.globals['fs_exists'] = _fs_exists
        self.globals['fs_info']   = _fs_info

        # ------------------------------------------------------------------
        # Inject FINAL_VAR helper
        # ------------------------------------------------------------------
        def final_var(variable_name: str) -> str:
            """Return the value of a variable from REPL as final answer."""
            variable_name = variable_name.strip().strip('"').strip("'").strip('\n').strip('\r')
            try:
                if variable_name in self.locals:
                    return str(self.locals[variable_name])
                else:
                    return f"Error: Variable '{variable_name}' not found in REPL environment"
            except Exception as e:
                return f"Error retrieving variable '{variable_name}': {str(e)}"

        self.globals['FINAL_VAR'] = final_var

        # ------------------------------------------------------------------
        # Load context
        # ------------------------------------------------------------------
        self._load_context(context_json, context_str)

    # ----------------------------------------------------------------------
    # Context loading
    # ----------------------------------------------------------------------

    def _load_context(self, context_json=None, context_str=None):
        """Load context directly into REPL locals."""
        if context_json is not None:
            self.locals['context'] = context_json
        if context_str is not None:
            self.locals['context'] = context_str

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------

    def __del__(self):
        """Clean up temporary directory."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    # ----------------------------------------------------------------------
    # I/O capture
    # ----------------------------------------------------------------------

    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr."""
        with self._lock:
            old_stdout = sys.stdout
            old_stderr = sys.stderr

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            try:
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                yield stdout_buffer, stderr_buffer
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr   # was a bug: previously restored stderr_buffer

    @contextmanager
    def _temp_working_directory(self):
        """Context manager to temporarily change working directory."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    # ----------------------------------------------------------------------
    # Code execution
    # ----------------------------------------------------------------------

    def code_execution(self, code: str) -> REPLResult:
        """
        Execute Python code in the REPL environment.

        Args:
            code: Python code to execute.

        Returns:
            REPLResult with stdout, stderr, locals, and execution time.
        """
        start_time = time.time()

        with self._capture_output() as (stdout_buffer, stderr_buffer):
            with self._temp_working_directory():
                try:
                    lines = code.split('\n')
                    import_lines = []
                    other_lines = []

                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith(('import ', 'from ')) and not stripped.startswith('#'):
                            import_lines.append(line)
                        else:
                            other_lines.append(line)

                    # Execute imports into globals so they persist
                    if import_lines:
                        import_code = '\n'.join(import_lines)
                        exec(import_code, self.globals, self.globals)

                    if other_lines:
                        other_code = '\n'.join(other_lines)
                        combined_namespace = {**self.globals, **self.locals}

                        non_comment_lines = [
                            line for line in other_lines
                            if line.strip() and not line.strip().startswith('#')
                        ]

                        if non_comment_lines:
                            last_line = non_comment_lines[-1]

                            is_expression = (
                                not last_line.strip().startswith((
                                    'import ', 'from ', 'def ', 'class ',
                                    'if ', 'for ', 'while ', 'try:', 'with ',
                                    'return ', 'yield ', 'break', 'continue', 'pass'
                                )) and
                                '=' not in last_line.split('#')[0] and
                                not last_line.strip().endswith(':') and
                                not last_line.strip().startswith('print(')
                            )

                            if is_expression:
                                try:
                                    if len(non_comment_lines) > 1:
                                        last_line_start = -1
                                        for i, line in enumerate(other_lines):
                                            if line.strip() == last_line.strip():
                                                last_line_start = i
                                                break
                                        if last_line_start > 0:
                                            statements_code = '\n'.join(other_lines[:last_line_start])
                                            exec(statements_code, combined_namespace, combined_namespace)

                                    result = eval(last_line, combined_namespace, combined_namespace)
                                    if result is not None:
                                        print(repr(result))
                                except Exception:
                                    exec(other_code, combined_namespace, combined_namespace)
                            else:
                                exec(other_code, combined_namespace, combined_namespace)
                        else:
                            exec(other_code, combined_namespace, combined_namespace)

                        # Persist new variables back to self.locals
                        for key, value in combined_namespace.items():
                            if key not in self.globals:
                                self.locals[key] = value

                    stdout_content = stdout_buffer.getvalue()
                    stderr_content = stderr_buffer.getvalue()

                except Exception as e:
                    stderr_content = stderr_buffer.getvalue() + str(e)
                    stdout_content = stdout_buffer.getvalue()
                    print(f"REPL execution error: {e}")
                    import traceback
                    traceback.print_exc()

        end_time = time.time()

        self.locals['_stdout'] = stdout_content
        self.locals['_stderr'] = stderr_content

        return REPLResult(stdout_content, stderr_content, self.locals.copy(), end_time - start_time)