"""
Sandboxed filesystem tools for RLM – enhanced with:
  - Windows path normalisation (C:\\Users\\... → works on both platforms)
  - Multi-format file parsing: .pdf, .docx, .doc, .xlsx, .csv, .json,
    .md, .txt (and any plain-text extension)
  - fs_parse(path) → extracted plain-text from any supported format

All operations are restricted to a whitelist of allowed root directories.
"""

import os
import re
import stat
import json
import platform
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Union

# ── Optional parser imports (graceful degradation) ───────────────────────────
try:
    import pdfplumber
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

try:
    from docx import Document as DocxDocument
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False

try:
    import openpyxl
    _HAS_XLSX = True
except ImportError:
    _HAS_XLSX = False

try:
    import csv as _csv
    _HAS_CSV = True
except ImportError:
    _HAS_CSV = False


# ── Constants ─────────────────────────────────────────────────────────────────
_DEFAULT_ALLOWED_ROOTS: List[str] = [
    "/workspace",
    "/tmp/rlm",
    os.path.expanduser("~/workspace"),
    os.path.expanduser("~/Downloads"),
    str(Path.home()),
]

_MAX_READ_BYTES: int = 4 * 1024 * 1024   # 4 MB
_MAX_LIST_ENTRIES: int = 500

# Extensions we can convert to plain text
_PARSEABLE_EXTENSIONS = {
    ".pdf", ".docx", ".doc",
    ".xlsx", ".xls",
    ".csv", ".tsv",
    ".json", ".jsonl",
    ".md", ".txt", ".log",
    ".py", ".js", ".ts", ".html", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".rst", ".tex",
}


# ── Path normalisation ────────────────────────────────────────────────────────

def _normalise_path(raw: str) -> Path:
    """
    Accept Windows-style (C:\\Users\\...) or Unix-style (/c/Users/...) paths
    and return a proper platform Path.

    On Linux (inside WSL or Docker) Windows absolute paths are translated:
        C:\\Users\\bob\\Downloads  →  /mnt/c/Users/bob/Downloads  (WSL)
        or simply expanded as given if the host is Windows.
    """
    s = raw.strip()

    # Detect Windows absolute path:  C:\...  or  C:/...
    win_abs = re.match(r'^([A-Za-z]):[/\\](.*)', s)
    if win_abs:
        drive, rest = win_abs.group(1).lower(), win_abs.group(2)
        rest = rest.replace("\\", "/")
        if platform.system() == "Windows":
            # Native Windows – reconstruct proper path
            return Path(f"{drive.upper()}:/{rest}")
        else:
            # Linux/macOS – try WSL mount point first, then home-relative fallback
            wsl = Path(f"/mnt/{drive}/{rest}")
            if wsl.exists():
                return wsl
            # Fallback: treat as relative to home
            return Path.home() / rest

    # Unix-style /c/Users/...  (Git Bash / MSYS2 convention)
    unix_drive = re.match(r'^/([a-z])/(.*)', s)
    if unix_drive and platform.system() != "Windows":
        drive, rest = unix_drive.group(1), unix_drive.group(2)
        wsl = Path(f"/mnt/{drive}/{rest}")
        if wsl.exists():
            return wsl

    return Path(s).expanduser()


# ── Exceptions ────────────────────────────────────────────────────────────────

class FSAccessDenied(PermissionError):
    """Raised when a path falls outside all allowed roots."""


# ── Core implementation ───────────────────────────────────────────────────────

class FSTools:
    """
    Filesystem tool set with path-jail sandboxing.

    New method:  fs_parse(path) → str
        Extracts plain-text from PDF, DOCX, XLSX, CSV, JSON, and plain-text
        files. Returns the extracted text, or raises RuntimeError on failure.
    """

    def __init__(self, allowed_roots: Optional[List[str]] = None):
        if allowed_roots is None:
            allowed_roots = _DEFAULT_ALLOWED_ROOTS

        self._roots: List[Path] = []
        for r in allowed_roots:
            expanded = _normalise_path(r)
            try:
                expanded.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            try:
                self._roots.append(expanded.resolve())
            except OSError:
                self._roots.append(expanded)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, path: Union[str, Path]) -> Path:
        """Resolve *path* to an absolute canonical Path inside the jail."""
        p = _normalise_path(str(path))
        if not p.is_absolute():
            p = self._roots[0] / p
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p

        for root in self._roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue

        # Also allow if any root is a prefix string match (handles unresolvable mounts)
        resolved_str = str(resolved)
        for root in self._roots:
            if resolved_str.startswith(str(root)):
                return resolved

        raise FSAccessDenied(
            f"Path '{resolved}' is outside all allowed roots: "
            + ", ".join(str(r) for r in self._roots)
        )

    # ------------------------------------------------------------------
    # Public API – directory / metadata
    # ------------------------------------------------------------------

    def list(self, path: str = ".") -> Dict[str, Any]:
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
                        "name":     child.name,
                        "type":     "dir" if child.is_dir() else "file",
                        "size":     s.st_size if child.is_file() else None,
                        "modified": s.st_mtime,
                        "parseable": child.suffix.lower() in _PARSEABLE_EXTENSIONS,
                    })
                except OSError:
                    entries.append({"name": child.name, "type": "unknown",
                                    "size": None, "modified": None, "parseable": False})
        except PermissionError as e:
            return {"path": str(resolved), "entries": [], "error": f"Permission denied: {e}"}

        return {"path": str(resolved), "entries": entries, "error": None}

    def read(self, path: str) -> Dict[str, Any]:
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
                "path": str(resolved), "content": None, "size": file_size,
                "error": f"File too large ({file_size} bytes). Use fs_parse() for structured extraction.",
            }

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except PermissionError as e:
            return {"path": str(resolved), "content": None, "size": file_size, "error": str(e)}

        return {"path": str(resolved), "content": content, "size": file_size, "error": None}

    def write(self, path: str, content: str, overwrite: bool = True) -> Dict[str, Any]:
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
        except (PermissionError, OSError) as e:
            return {"path": str(resolved), "written": 0, "error": str(e)}

    def exists(self, path: str) -> Dict[str, Any]:
        try:
            resolved = self._resolve(path)
        except FSAccessDenied as e:
            return {"path": path, "exists": False, "type": None, "error": str(e)}

        if not resolved.exists():
            return {"path": str(resolved), "exists": False, "type": None, "error": None}

        kind = "dir" if resolved.is_dir() else ("file" if resolved.is_file() else "other")
        return {"path": str(resolved), "exists": True, "type": kind, "error": None}

    def info(self, path: str) -> Dict[str, Any]:
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
                "path": str(resolved), "size": s.st_size, "modified": s.st_mtime,
                "is_file": resolved.is_file(), "is_dir": resolved.is_dir(),
                "permissions": oct(stat.S_IMODE(s.st_mode)), "error": None,
            }
        except OSError as e:
            return {"path": str(resolved), "size": None, "modified": None,
                    "is_file": False, "is_dir": False, "permissions": None, "error": str(e)}

    # ------------------------------------------------------------------
    # NEW:  fs_parse – multi-format text extraction
    # ------------------------------------------------------------------

    def parse(self, path: str, max_chars: int = 50_000) -> Dict[str, Any]:
        """
        Extract plain text from a file regardless of format.

        Supported formats:
            .pdf   → pdfplumber (install: pip install pdfplumber)
            .docx  → python-docx  (install: pip install python-docx)
            .xlsx / .xls → openpyxl (install: pip install openpyxl)
            .csv / .tsv  → csv stdlib
            .json / .jsonl → json stdlib
            everything else → raw UTF-8 read

        Returns dict:
            path      : str
            text      : str | None   (extracted plain text, up to max_chars)
            format    : str          (detected format)
            size      : int          (file size in bytes)
            truncated : bool
            error     : str | None
        """
        try:
            resolved = self._resolve(path)
        except FSAccessDenied as e:
            return {"path": path, "text": None, "format": "unknown", "size": 0,
                    "truncated": False, "error": str(e)}

        if not resolved.exists():
            return {"path": str(resolved), "text": None, "format": "unknown", "size": 0,
                    "truncated": False, "error": "File not found"}
        if not resolved.is_file():
            return {"path": str(resolved), "text": None, "format": "unknown", "size": 0,
                    "truncated": False, "error": "Not a file"}

        ext = resolved.suffix.lower()
        size = resolved.stat().st_size

        try:
            text = self._extract_text(resolved, ext)
        except Exception as e:
            return {"path": str(resolved), "text": None, "format": ext or "binary", "size": size,
                    "truncated": False, "error": f"Parse error: {e}"}

        truncated = len(text) > max_chars
        return {
            "path": str(resolved),
            "text": text[:max_chars],
            "format": ext or "text",
            "size": size,
            "truncated": truncated,
            "error": None,
        }

    def _extract_text(self, path: Path, ext: str) -> str:
        """Dispatch to the right parser based on file extension."""

        # ── PDF ──────────────────────────────────────────────────────────────
        if ext == ".pdf":
            if not _HAS_PDF:
                raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")
            import pdfplumber
            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
            return "\n\n".join(pages)

        # ── DOCX ─────────────────────────────────────────────────────────────
        if ext in (".docx", ".doc"):
            if ext == ".docx":
                if not _HAS_DOCX:
                    raise ImportError("python-docx not installed. Run: pip install python-docx")
                from docx import Document
                doc = Document(str(path))
                return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            else:
                # .doc (legacy binary) – fall back to antiword or textract if available
                try:
                    import subprocess
                    result = subprocess.run(
                        ["antiword", str(path)], capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0:
                        return result.stdout
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
                # Last resort: read raw bytes and decode printable ASCII
                raw = path.read_bytes()
                return "".join(chr(b) for b in raw if 32 <= b < 127 or b in (9, 10, 13))

        # ── XLSX / XLS ───────────────────────────────────────────────────────
        if ext in (".xlsx", ".xls"):
            if not _HAS_XLSX:
                raise ImportError("openpyxl not installed. Run: pip install openpyxl")
            import openpyxl
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
            lines = []
            for sheet in wb.worksheets:
                lines.append(f"=== Sheet: {sheet.title} ===")
                for row in sheet.iter_rows(values_only=True):
                    row_str = "\t".join(str(c) if c is not None else "" for c in row)
                    if row_str.strip():
                        lines.append(row_str)
            return "\n".join(lines)

        # ── CSV / TSV ────────────────────────────────────────────────────────
        if ext in (".csv", ".tsv"):
            import csv
            delimiter = "\t" if ext == ".tsv" else ","
            lines = []
            with open(str(path), newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter=delimiter)
                for row in reader:
                    lines.append("\t".join(row))
            return "\n".join(lines)

        # ── JSON / JSONL ─────────────────────────────────────────────────────
        if ext in (".json", ".jsonl"):
            raw = path.read_text(encoding="utf-8", errors="replace")
            try:
                if ext == ".jsonl":
                    objs = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    return json.dumps(objs, indent=2)
                else:
                    return json.dumps(json.loads(raw), indent=2)
            except json.JSONDecodeError:
                return raw  # return as-is if not valid JSON

        # ── Plain text (default) ─────────────────────────────────────────────
        return path.read_text(encoding="utf-8", errors="replace")


# ── Tool definitions (OpenAI function-calling format) ─────────────────────────

FS_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fs_list",
            "description": (
                "List the contents of a directory. Each entry includes name, type, "
                "size, modified timestamp, and whether it's parseable by fs_parse."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_read",
            "description": "Read raw text content of a file (plain text / source code). Use fs_parse for PDF/DOCX/XLSX.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_parse",
            "description": (
                "Extract plain text from any supported file format: "
                ".pdf, .docx, .doc, .xlsx, .csv, .json, .txt, .md, and more. "
                "Always prefer this over fs_read for non-plain-text files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to parse."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default 50000).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_write",
            "description": "Write text content to a file inside the sandboxed workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean", "description": "Default true."},
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
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_info",
            "description": "Return metadata for a path: size, modified timestamp, permissions.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def dispatch_tool_call(
    fs: FSTools,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> str:
    handlers = {
        "fs_list":   lambda a: fs.list(a.get("path", ".")),
        "fs_read":   lambda a: fs.read(a["path"]),
        "fs_parse":  lambda a: fs.parse(a["path"], a.get("max_chars", 50_000)),
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