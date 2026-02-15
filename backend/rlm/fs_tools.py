"""
Sandboxed filesystem tools for RLM.

All operations are restricted to a whitelist of allowed root directories.
Paths are canonicalized and validated before any operation to prevent
directory traversal attacks (e.g. ../../etc/passwd).

Available REPL functions injected into the environment:
    fs_list(path)           -> list directory contents
    fs_read(path)           -> read a file's text content
    fs_write(path, content) -> write text to a file
    fs_exists(path)         -> check whether a path exists
    fs_info(path)           -> return stat metadata for a path

Available tool-call definitions (for LLM tool-calling layer):
    FS_TOOL_DEFINITIONS     -> list[dict] in OpenAI tool format
"""

import os
import stat
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Default allowed roots – point at Daytona workspace root and a scratch dir.
# Callers can override by passing allowed_roots to FSTools().
# ---------------------------------------------------------------------------
_DEFAULT_ALLOWED_ROOTS: List[str] = [
    "/workspace",   # typical Daytona project root
    "/tmp/rlm",     # scratch space for generated files
    os.path.expanduser("~/workspace"),  # fallback if no /workspace
]

# Reasonable size cap so the LLM doesn't accidentally try to read a 2 GB binary
_MAX_READ_BYTES: int = 4 * 1024 * 1024   # 4 MB
_MAX_LIST_ENTRIES: int = 500              # cap directory listings


# ---------------------------------------------------------------------------
# Core sandboxed implementation
# ---------------------------------------------------------------------------

class FSAccessDenied(PermissionError):
    """Raised when a path falls outside all allowed roots."""


class FSTools:
    """
    Filesystem tool set with path-jail sandboxing.

    All public methods return plain Python values (str, list, dict)
    so they are easy to use from the REPL and easy to serialize for
    tool-call responses.
    """

    def __init__(self, allowed_roots: Optional[List[str]] = None):
        if allowed_roots is None:
            allowed_roots = _DEFAULT_ALLOWED_ROOTS

        # Resolve and deduplicate roots; create scratch dir if needed
        self._roots: List[Path] = []
        for r in allowed_roots:
            expanded = Path(r).expanduser()
            try:
                expanded.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass  # non-writable roots are fine for reads
            self._roots.append(expanded.resolve())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, path: Union[str, Path]) -> Path:
        """
        Resolve *path* to an absolute canonical Path and assert it sits
        inside at least one of the allowed roots.

        Raises FSAccessDenied if the resolved path escapes all roots.
        """
        p = Path(path).expanduser()
        if not p.is_absolute():
            # Treat relative paths as relative to the first allowed root
            p = self._roots[0] / p
        resolved = p.resolve()

        for root in self._roots:
            try:
                resolved.relative_to(root)
                return resolved          # passes jail check
            except ValueError:
                continue

        raise FSAccessDenied(
            f"Path '{resolved}' is outside all allowed roots: "
            + ", ".join(str(r) for r in self._roots)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list(self, path: str = ".") -> Dict[str, Any]:
        """
        List a directory.

        Returns a dict with:
            path    : str   – resolved path
            entries : list  – each entry has {name, type, size, modified}
            error   : str | None
        """
        try:
            resolved = self._resolve(path)
        except FSAccessDenied as e:
            return {"path": path, "entries": [], "error": str(e)}

        if not resolved.exists():
            return {"path": str(resolved), "entries": [], "error": "Path does not exist"}
        if not resolved.is_dir():
            return {"path": str(resolved), "entries": [], "error": "Path is not a directory"}

        entries = []
        try:
            children = sorted(resolved.iterdir())[:_MAX_LIST_ENTRIES]
            for child in children:
                try:
                    s = child.stat()
                    entries.append({
                        "name": child.name,
                        "type": "dir" if child.is_dir() else "file",
                        "size": s.st_size if child.is_file() else None,
                        "modified": s.st_mtime,
                    })
                except OSError:
                    entries.append({"name": child.name, "type": "unknown", "size": None, "modified": None})
        except PermissionError as e:
            return {"path": str(resolved), "entries": [], "error": f"Permission denied: {e}"}

        return {"path": str(resolved), "entries": entries, "error": None}

    def read(self, path: str) -> Dict[str, Any]:
        """
        Read a text file.

        Returns a dict with:
            path    : str
            content : str | None
            size    : int
            error   : str | None
        """
        try:
            resolved = self._resolve(path)
        except FSAccessDenied as e:
            return {"path": path, "content": None, "size": 0, "error": str(e)}

        if not resolved.exists():
            return {"path": str(resolved), "content": None, "size": 0, "error": "File not found"}
        if not resolved.is_file():
            return {"path": str(resolved), "content": None, "size": 0, "error": "Path is not a file"}

        file_size = resolved.stat().st_size
        if file_size > _MAX_READ_BYTES:
            return {
                "path": str(resolved),
                "content": None,
                "size": file_size,
                "error": f"File too large to read ({file_size} bytes > {_MAX_READ_BYTES} byte limit). "
                         "Use fs_read with a byte-range slice or process in chunks.",
            }

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except PermissionError as e:
            return {"path": str(resolved), "content": None, "size": file_size, "error": str(e)}

        return {"path": str(resolved), "content": content, "size": file_size, "error": None}

    def write(self, path: str, content: str, overwrite: bool = True) -> Dict[str, Any]:
        """
        Write text to a file, creating parent directories as needed.

        Returns a dict with:
            path    : str
            written : int   – bytes written
            error   : str | None
        """
        try:
            resolved = self._resolve(path)
        except FSAccessDenied as e:
            return {"path": path, "written": 0, "error": str(e)}

        if resolved.exists() and not overwrite:
            return {"path": str(resolved), "written": 0, "error": "File exists and overwrite=False"}

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return {"path": str(resolved), "written": len(content.encode("utf-8")), "error": None}
        except PermissionError as e:
            return {"path": str(resolved), "written": 0, "error": str(e)}
        except OSError as e:
            return {"path": str(resolved), "written": 0, "error": str(e)}

    def exists(self, path: str) -> Dict[str, Any]:
        """
        Check whether a path exists inside the sandbox.

        Returns a dict with:
            path   : str
            exists : bool
            type   : "file" | "dir" | "other" | None
            error  : str | None
        """
        try:
            resolved = self._resolve(path)
        except FSAccessDenied as e:
            return {"path": path, "exists": False, "type": None, "error": str(e)}

        if not resolved.exists():
            return {"path": str(resolved), "exists": False, "type": None, "error": None}

        kind = "dir" if resolved.is_dir() else ("file" if resolved.is_file() else "other")
        return {"path": str(resolved), "exists": True, "type": kind, "error": None}

    def info(self, path: str) -> Dict[str, Any]:
        """
        Return stat metadata for a path.

        Returns a dict with:
            path        : str
            size        : int | None
            modified    : float | None  (unix timestamp)
            is_file     : bool
            is_dir      : bool
            permissions : str  (octal string, e.g. "0o644")
            error       : str | None
        """
        try:
            resolved = self._resolve(path)
        except FSAccessDenied as e:
            return {"path": path, "size": None, "modified": None,
                    "is_file": False, "is_dir": False, "permissions": None, "error": str(e)}

        if not resolved.exists():
            return {"path": str(resolved), "size": None, "modified": None,
                    "is_file": False, "is_dir": False, "permissions": None,
                    "error": "Path does not exist"}
        try:
            s = resolved.stat()
            return {
                "path": str(resolved),
                "size": s.st_size,
                "modified": s.st_mtime,
                "is_file": resolved.is_file(),
                "is_dir": resolved.is_dir(),
                "permissions": oct(stat.S_IMODE(s.st_mode)),
                "error": None,
            }
        except OSError as e:
            return {"path": str(resolved), "size": None, "modified": None,
                    "is_file": False, "is_dir": False, "permissions": None, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool definitions for the LLM tool-calling layer
# ---------------------------------------------------------------------------

FS_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fs_list",
            "description": (
                "List the contents of a directory inside the sandboxed workspace. "
                "Returns names, types (file/dir), sizes, and modification timestamps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Relative paths are resolved "
                                       "against the workspace root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_read",
            "description": (
                "Read the text content of a file inside the sandboxed workspace. "
                "Files larger than 4 MB cannot be read directly; chunk them instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_write",
            "description": (
                "Write text content to a file inside the sandboxed workspace. "
                "Parent directories are created automatically. "
                "Set overwrite=false to protect existing files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Destination file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write.",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Whether to overwrite an existing file. Default true.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_exists",
            "description": "Check whether a path exists inside the sandboxed workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to check.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_info",
            "description": (
                "Return metadata for a path (size, modification time, permissions). "
                "Useful before deciding whether to read or overwrite a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to inspect.",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Helper: dispatch a tool call by name
# ---------------------------------------------------------------------------

def dispatch_tool_call(
    fs: FSTools,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> str:
    """
    Execute a single filesystem tool call and return a JSON string result.
    Safe to call from the tool-calling layer in rlm_repl.py.
    """
    handlers = {
        "fs_list":   lambda a: fs.list(a["path"]),
        "fs_read":   lambda a: fs.read(a["path"]),
        "fs_write":  lambda a: fs.write(a["path"], a["content"], a.get("overwrite", True)),
        "fs_exists": lambda a: fs.exists(a["path"]),
        "fs_info":   lambda a: fs.info(a["path"]),
    }
    handler = handlers.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = handler(tool_args)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})