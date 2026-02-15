"""
Prompts for RLM.
Extended with fs_parse documentation and a direct Downloads-analysis strategy.

NOTE: add_context_metadata uses str.replace() NOT str.format(), so literal
{braces} in the prompt (dict examples, f-string examples, etc.) never cause KeyErrors.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Placeholders used in the system prompt (plain tokens, not {}-format syntax)
# ---------------------------------------------------------------------------
_PH_TYPE   = "CONTEXT_TYPE_PLACEHOLDER"
_PH_TOTAL  = "CONTEXT_TOTAL_LENGTH_PLACEHOLDER"
_PH_CHUNKS = "CONTEXT_LENGTHS_PLACEHOLDER"

# ---------------------------------------------------------------------------
# Base system prompt  –  uses plain placeholder tokens, NOT {var} syntax
# ---------------------------------------------------------------------------

_BASE_SYSTEM = (
    "You are tasked with answering a query with associated context. "
    "You can access, transform, and analyze this context interactively in a REPL "
    "environment that can recursively query sub-LLMs and interact with the filesystem. "
    "You will be queried iteratively until you provide a final answer.\n\n"
    "Your context is a " + _PH_TYPE + " with " + _PH_TOTAL + " total characters, "
    "and is broken up into chunks of char lengths: " + _PH_CHUNKS + ".\n\n"
    """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPL ENVIRONMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The REPL is initialized with:

1. `context`           – variable containing the task context.

2. `llm_query(prompt)` – call a sub-LLM. Use this to summarise/classify text.

3. Filesystem functions (sandboxed to allowed roots):

   fs_list(path)
       List a directory.
       Returns: {"path": ..., "entries": [{"name","type","size","modified","parseable"}], "error": ...}
       NOTE: "parseable": true means fs_parse() can extract text from that file.

   fs_parse(path, max_chars=50000)   <- USE THIS FOR PDF/DOCX/XLSX/CSV/JSON
       Extract plain text from ANY supported format:
           .pdf   (pdfplumber)
           .docx / .doc  (python-docx)
           .xlsx / .xls  (openpyxl)
           .csv / .tsv   (stdlib)
           .json / .jsonl (stdlib)
           .txt / .md / .py / etc. (raw UTF-8)
       Returns: {"path": ..., "text": ..., "format": ..., "size": ..., "truncated": ..., "error": ...}
       Example:
           r = fs_parse('/downloads/report.pdf')
           if not r['error']:
               summary = llm_query('One sentence summary: ' + r['text'][:8000])

   fs_read(path)
       Read a plain-text file (UTF-8). Use fs_parse instead for binary formats.
       Returns str content or raises RuntimeError.

   fs_write(path, content, overwrite=True)
       Write text to a file. Returns: {"path": ..., "written": ..., "error": ...}

   fs_exists(path)  -> bool
   fs_info(path)    -> {"path","size","modified","is_file","is_dir","permissions","error"}

4. `FINAL_VAR(variable_name)` – return a REPL variable as the final answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WINDOWS PATH SUPPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Windows-style paths are automatically translated. You can pass:
    fs_list('C:/Users/bob/Downloads')
    fs_parse('C:/Users/bob/Downloads/report.pdf')

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY: ANALYZING A DOWNLOADS FOLDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When asked to analyze a directory and summarise files, use this template:

```repl
import datetime

listing = fs_list('C:/Users/bob/Downloads')   # <-- replace with actual path
if listing['error']:
    print('Error:', listing['error'])
else:
    files = [e for e in listing['entries'] if e['type'] == 'file']
    print('Found', len(files), 'files')

    rows = []
    for f in files:
        full_path = listing['path'].replace('\\\\', '/') + '/' + f['name']
        size_kb   = round(f['size'] / 1024, 1) if f['size'] else 0
        mtime     = datetime.datetime.fromtimestamp(f['modified']).strftime('%Y-%m-%d') if f['modified'] else 'n/a'
        fmt       = f['name'].rsplit('.', 1)[-1].lower() if '.' in f['name'] else '-'
        summary   = '-'

        if f.get('parseable'):
            r = fs_parse(full_path, max_chars=6000)
            if not r['error'] and r['text']:
                summary = llm_query('One sentence summary (max 20 words): ' + r['text'][:4000])

        rows.append((f['name'], str(size_kb) + ' KB', mtime, fmt, summary))

    total_kb = round(sum(f['size'] or 0 for f in files) / 1024, 1)
    header  = '| File | Size | Modified | Format | Summary |'
    divider = '|------|------|----------|--------|---------|'
    table_rows = ['| ' + ' | '.join(r) + ' |' for r in rows]
    report = '\\n'.join([header, divider] + table_rows + ['', '**Total:** ' + str(len(files)) + ' files, ' + str(total_kb) + ' KB'])
    print(report)
```
FINAL_VAR('report')

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TERMINATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When you have your final answer, emit ONE of these (outside any code block):

  FINAL(your answer text here)
  FINAL_VAR(variable_name)

Think step by step, execute your plan immediately, and emit FINAL as soon
as you have the answer. Do NOT loop asking for clarification – act."""
)

# ---------------------------------------------------------------------------
# Qwen-specific prefix
# ---------------------------------------------------------------------------

_QWEN_PREFIX = (
    "IMPORTANT: Be very careful about using 'llm_query' as it incurs runtime "
    "costs. Batch as much information as possible into each call (~200K chars). "
    "Minimise llm_query calls.\n\n"
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_system_prompt(model: str) -> List[Dict[str, str]]:
    if "qwen" in model.lower():
        prompt = _QWEN_PREFIX + _BASE_SYSTEM
    else:
        prompt = _BASE_SYSTEM
    return [{"role": "system", "content": prompt}]


def add_context_metadata(
    messages: List[Dict[str, str]],
    context_type: str,
    context_lengths: List[int],
    context_total_length: int,
) -> List[Dict[str, str]]:
    """
    Fill in the context placeholders using plain str.replace() – never str.format() –
    so that any literal braces in the prompt (dict examples, f-string examples, etc.)
    are left untouched and never raise KeyError.
    """
    content = messages[0]["content"]
    content = content.replace(_PH_TYPE,   str(context_type))
    content = content.replace(_PH_TOTAL,  str(context_total_length))
    content = content.replace(_PH_CHUNKS, str(context_lengths))
    messages[0]["content"] = content
    return messages


def next_action_prompt(
    query: str,
    iteration: int = 0,
    final_answer: bool = False,
) -> Dict[str, str]:
    if final_answer:
        return {
            "role": "user",
            "content": "Based on all the information gathered, provide your final answer.",
        }

    if iteration == 0:
        safeguard = (
            "You have NOT yet explored the filesystem or executed any REPL code. "
            "Do NOT emit FINAL yet – first run the exploration code shown in the strategy section above.\n\n"
        )
        content = (
            safeguard
            + f'Think step-by-step and IMMEDIATELY write ```repl``` code to answer: "{query}"\n\n'
            'Use fs_parse() for PDF/DOCX/XLSX files. Adapt the Downloads-analysis '
            'template from the system prompt. Execute your code NOW – do not describe what you will do.'
        )
    else:
        content = (
            f'Continue toward the answer for: "{query}"\n\n'
            'If you have a report or result variable, emit FINAL_VAR(report) or FINAL(text) NOW.\n'
            'If there were errors, fix them with more ```repl``` code. '
            'Do not repeat code that already worked.'
        )

    return {"role": "user", "content": content}