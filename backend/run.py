"""
run.py – RLM entry point.

Selectable via CLI:
    python run.py pipeline   – sub-agent pipeline on a huge prompt
    python run.py downloads  – list + parse ~/Downloads with fs_parse()
    python run.py            – runs both sequentially (demo mode)

Environment: copy .env.example → .env and fill in your values.
"""

from __future__ import annotations

import os
import sys
import subprocess
import json
import textwrap
from pathlib import Path


# ── Load .env ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[run] .env loaded via python-dotenv")
except ImportError:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
        print("[run] .env loaded manually")
    else:
        print("[run] Warning: no .env file found.")


# ── Optional dependency installer ─────────────────────────────────────────────

def _ensure_parsers():
    """
    Install file-parsing libraries if not already present.
    Tries uv (project tool) first, then falls back to pip.
    """
    pkgs = {
        "pdfplumber": "pdfplumber",
        "docx":       "python-docx",
        "openpyxl":   "openpyxl",
    }
    for import_name, pip_name in pkgs.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"[run] Installing {pip_name} for file parsing...")
            # Try uv first (used by this project), then pip as fallback
            installed = False
            for cmd in (
                ["uv", "add", pip_name],
                [sys.executable, "-m", "pip", "install", pip_name, "-q"],
            ):
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"[run] {pip_name} installed OK")
                    installed = True
                    break
            if not installed:
                print(f"[run] WARNING: could not install {pip_name} – "
                      f"run manually: uv add {pip_name}")


def banner(title: str):
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ============================================================================
# USE CASE 1 – Huge prompt → sub-agent pipeline
# ============================================================================

HUGE_PROMPT = """\
QUARTERLY EARNINGS REPORT – FY 2025 Q3
=======================================

Executive Summary
-----------------
Total revenue for Q3 2025 reached $4.82 billion, representing a 12.4 % year-over-year
increase. Operating income grew 18 % to $1.1 billion. Net income attributable to
shareholders was $890 million ($1.23 per diluted share), compared to $730 million
($1.01 per diluted share) in Q3 2024.

Business Segments
-----------------

Cloud & Infrastructure (34 % of revenue)
  Revenue: $1.64 B  (+22 % YoY)
  Operating margin: 31 %
  Key drivers: enterprise migration contracts (+47 %), managed Kubernetes demand,
  GPU-cluster rental (AI workloads) up 3x vs prior year.
  Action item: Expand Tier-2 datacenter in Frankfurt (Q4 target).

Software & Licensing (28 % of revenue)
  Revenue: $1.35 B  (+8 % YoY)
  Operating margin: 52 %
  Key drivers: seat-based SaaS renewals, new SMB licensing tier launched July 2025.
  Risk: Three enterprise clients (combined $120 M ARR) up for renewal in Q4.
  Action item: Executive engagement program for top 10 at-risk accounts.

Professional Services (21 % of revenue)
  Revenue: $1.01 B  (+6 % YoY)
  Operating margin: 18 %
  Key drivers: AI implementation engagements, compliance consulting (EU AI Act).
  Action item: Hire 200 additional AI consultants by end of Q4.

Hardware (17 % of revenue)
  Revenue: $820 M  (-3 % YoY)
  Operating margin: 9 %
  Key drivers: Supply chain normalization offset by demand shift to cloud.
  Action item: Wind-down legacy server SKUs; invest in edge-compute devices.

Financial Highlights
--------------------
  Cash & equivalents:   $3.2 B  (up from $2.8 B in Q2)
  Free cash flow:       $1.05 B  (FCF margin 21.8 %)
  R&D spend:            $480 M   (10 % of revenue)
  CapEx:                $310 M   (datacenter expansion, AI infrastructure)
  Headcount:            41,200   (+4 % YoY)
  Employee turnover:    8.2 %    (industry avg ~13 %)

Guidance – Q4 2025
-------------------
  Revenue:       $5.1 – $5.3 B
  Operating income: $1.15 – $1.25 B
  EPS (diluted): $1.28 – $1.40

Risk Factors
------------
1. Macroeconomic slowdown reducing enterprise IT budgets.
2. Competitive pressure in cloud from Hyperscaler X (recent price cuts –15 %).
3. Regulatory uncertainty: EU AI Act compliance costs estimated $30–50 M in 2026.
4. FX headwinds: 40 % of revenue denominated in EUR/GBP.
5. Key-person risk: VP of Cloud departing end of October; succession plan in place.

Upcoming Milestones
-------------------
- Oct 15 2025: Board approval for Frankfurt datacenter expansion.
- Nov 1  2025: SMB licensing tier general availability.
- Nov 30 2025: Q4 interim revenue update.
- Dec 15 2025: Annual all-hands and FY2026 strategy presentation.

Appendix – Regional Breakdown
------------------------------
  Americas:    $2.65 B  (55 %)   +14 % YoY
  EMEA:        $1.54 B  (32 %)   + 9 % YoY
  APAC:        $0.63 B  (13 %)   +18 % YoY

Appendix – Top 5 Customers by Revenue
--------------------------------------
  1. GlobalBank Inc.        $210 M  (Cloud + Software)
  2. HealthNet Systems      $185 M  (Cloud + Services)
  3. RetailCo International $140 M  (Software + Hardware)
  4. GovCloud Agency        $120 M  (Cloud)
  5. AutoManufacture GmbH   $98 M   (Hardware + Services)

[END OF REPORT]
""" * 3  # Tripled to simulate a large prompt (~9 KB)


def run_pipeline():
    banner("USE CASE 1 – Sub-Agent Pipeline on Huge Prompt")

    from rlm.agents import PromptPipeline

    print(f"Prompt size: {len(HUGE_PROMPT):,} chars\n")

    pipeline = PromptPipeline()
    task = (
        "Extract every explicit ACTION ITEM mentioned in the report. "
        "Return them as a numbered markdown list with the responsible segment "
        "and target date (if mentioned)."
    )
    print(f"Task: {task}\n")

    result = pipeline.run(prompt=HUGE_PROMPT, task=task, parallel=True)

    print()
    print("─" * 70)
    print("FINAL OUTPUT")
    print("─" * 70)
    print(result.final_output)
    print()
    print("─" * 70)
    print(result.cost_summary())
    print()


# ============================================================================
# USE CASE 2 – Extract and summarise content from ~/Downloads
# ============================================================================



def run_downloads():
    banner("USE CASE 2 – Analyse & Summarise ~/Downloads")

    _ensure_parsers()

    downloads_path = str(Path.home() / "Downloads")
    # Forward slashes prevent \b \n \t escape bugs in model-generated string literals
    downloads_fwd  = downloads_path.replace("\\", "/")
    allowed = [downloads_path, "/tmp/rlm_scratch"]

    from rlm.rlm_repl import RLM_REPL

    rlm = RLM_REPL(
        allowed_roots=allowed,
        max_iterations=int(os.getenv("RLM_MAX_ITERATIONS", "15")),
        # Pre-inject path as a REPL variable so the model never types a
        # backslash path literal (which it consistently mangles).
        extra_locals={"DOWNLOADS": downloads_fwd},
    )

    # The query IS the code – wrapped in a repl fence so it executes on iter 0.
    # Using single-quoted strings throughout to avoid escape headaches.
    query = (
        "The variable DOWNLOADS is already defined in the REPL. "
        "Copy and run this code block exactly as-is:\n\n"
        "```repl\n"
        "import datetime\n"
        "listing = fs_list(DOWNLOADS)\n"
        "if listing['error']:\n"
        "    report = 'Error listing directory: ' + listing['error']\n"
        "else:\n"
        "    files = [e for e in listing['entries'] if e['type'] == 'file']\n"
        "    rows = []\n"
        "    for f in files:\n"
        "        fp      = DOWNLOADS + '/' + f['name']\n"
        "        size_kb = str(round(f['size'] / 1024, 1)) + ' KB' if f['size'] else '0 KB'\n"
        "        mtime   = datetime.datetime.fromtimestamp(f['modified']).strftime('%Y-%m-%d') if f['modified'] else 'n/a'\n"
        "        fmt     = f['name'].rsplit('.', 1)[-1].lower() if '.' in f['name'] else '-'\n"
        "        summary = '-'\n"
        "        if f.get('parseable'):\n"
        "            r = fs_parse(fp, max_chars=6000)\n"
        "            if not r['error'] and r['text']:\n"
        "                summary = llm_query('One sentence summary (max 20 words): ' + r['text'][:4000])\n"
        "        rows.append((f['name'], size_kb, mtime, fmt, summary))\n"
        "    total_kb = round(sum(f['size'] or 0 for f in files) / 1024, 1)\n"
        "    lines  = ['| File | Size | Modified | Format | Summary |']\n"
        "    lines += ['|------|------|----------|--------|---------|']\n"
        "    lines += ['| ' + ' | '.join(r) + ' |' for r in rows]\n"
        "    lines += ['', '**Total:** ' + str(len(files)) + ' files, ' + str(total_kb) + ' KB']\n"
        "    report = '\\n'.join(lines)\n"
        "    print(report)\n"
        "```\n"
        "FINAL_VAR('report')"
    )

    print(f"Downloads path : {downloads_path}")
    print(f"REPL variable  : DOWNLOADS = '{downloads_fwd}'")
    print(f"Allowed roots  : {allowed}\n")

    context = f"Target directory: {downloads_fwd}"
    result  = rlm.completion(context=context, query=query)

    print()
    print("─" * 70)
    print("FINAL OUTPUT")
    print("─" * 70)
    if result:
        print(result)
    else:
        print("(RLM reached max iterations without a final answer.)")
        print()
        print("Tip: increase RLM_MAX_ITERATIONS in your .env")
    print()

    costs = rlm.cost_summary()
    print(
        f"Cost: ${costs['total_cost']:.6f}  |  "
        f"Root calls: {costs['root_llm_calls']}  |  "
        f"Sub-LLM calls: {costs['sub_llm_calls']}"
    )



# ============================================================================
# CLI dispatch
# ============================================================================

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("pipeline", "1"):
        run_pipeline()
    elif mode in ("downloads", "2"):
        run_downloads()
    else:
        run_pipeline()
        run_downloads()


if __name__ == "__main__":
    main()