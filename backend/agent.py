"""
agent.py – Natural Language File Agent
Integrates: RLM (REPL execution) + Mem0 (memory) + Cognee (knowledge graph)

Usage:
    uv run agent.py                          # interactive loop
    uv run agent.py "analyse my budget"      # single shot
    uv run agent.py --dir C:/path "query"    # custom directory
    uv run agent.py --build-graph            # build + visualise graph only
"""

from __future__ import annotations
import os, sys, json, asyncio, datetime, textwrap, re
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

def _ensure_deps():
    import subprocess
    for imp, pip in {"pdfplumber":"pdfplumber","docx":"python-docx",
                     "openpyxl":"openpyxl","networkx":"networkx","pyvis":"pyvis"}.items():
        try:
            __import__(imp)
        except ImportError:
            print(f"[agent] Installing {pip}...")
            for cmd in (["uv","add",pip],[sys.executable,"-m","pip","install",pip,"-q"]):
                if subprocess.run(cmd, capture_output=True).returncode == 0:
                    break

_ensure_deps()

from rlm.fs_tools        import FSTools
from rlm.utils.llm       import get_llm_client
from rlm.rlm_repl        import RLM_REPL
from rlm.memory          import AgentMemory
from rlm.knowledge_graph import KnowledgeGraph

SEARCH_DIRS     = [str(Path.home()/d) for d in ("Downloads","Documents","Desktop")]
MAX_PARSE_CHARS = 60_000
SCRATCH_DIR     = "/tmp/rlm_scratch"
DATA_DIR        = "./data"


def discover_files(directories):
    fs, files = FSTools(allowed_roots=directories+[SCRATCH_DIR]), []
    for d in directories:
        d_fwd   = d.replace("\\","/")
        listing = fs.list(d_fwd)
        if listing["error"]: continue
        for e in listing["entries"]:
            if e["type"] != "file": continue
            files.append({
                "name":      e["name"],
                "path":      d_fwd+"/"+e["name"],
                "size_kb":   round((e["size"] or 0)/1024,1),
                "modified":  (datetime.datetime.fromtimestamp(e["modified"]).strftime("%Y-%m-%d")
                              if e["modified"] else "n/a"),
                "ext":       e["name"].rsplit(".",1)[-1].lower() if "." in e["name"] else "",
                "parseable": e.get("parseable",False),
            })
    return files


def select_relevant_files(user_request, files, llm):
    if not files: return []
    catalogue = "\n".join(f"{i+1}. {f['name']}  ({f['size_kb']} KB, .{f['ext']})"
                           for i,f in enumerate(files))
    prompt = (f"User request: {user_request}\n\nFiles:\n{catalogue}\n\n"
              "Reply ONLY with a JSON array of file numbers, e.g. [2,5]. Max 3.")
    try:
        raw = llm.completion([{"role":"user","content":prompt}]).strip()
        m   = re.search(r'\[[\d,\s]*\]', raw)
        return [files[i-1] for i in json.loads(m.group()) if 1<=i<=len(files)][:3] if m else []
    except Exception:
        return [f for f in files if f["parseable"]][:3]


def load_file_content(path, fs):
    r = fs.parse(path, max_chars=MAX_PARSE_CHARS)
    if r["error"]: print(f"  [warn] {Path(path).name}: {r['error']}"); return ""
    if r["truncated"]: print(f"  [info] {Path(path).name} truncated")
    return r["text"] or ""


def build_analysis_prompt(user_request, loaded_files, memories):
    files_desc = "\n".join(f"  - FILE_{i+1}: '{f['name']}' ({f['size_kb']} KB, .{f['ext']})"
                            for i,f in enumerate(loaded_files))
    mem_section = ("\nRELEVANT MEMORIES:\n"+"\n".join(f"  - {m}" for m in memories)+"\n"
                   if memories else "")
    return textwrap.dedent(f"""
        You are a data analysis agent. Files are pre-loaded in the REPL.
        USER REQUEST: {user_request}
        {mem_section}
        LOADED FILES:
        {files_desc}
        Each FILE_N variable has the extracted text. FILE_N_PATH and FILE_N_NAME also available.
        INSTRUCTIONS:
        1. Inspect files with repl code blocks
        2. Parse CSV with csv module; JSON with json.loads()
        3. Perform the requested analysis with Python arithmetic
        4. Print intermediate results so user can follow
        5. End with FINAL(your answer)
        TOOLS: llm_query(), fs_list(), fs_parse(), fs_write()
    """).strip()


async def build_knowledge_graph(directories, output_html="./data/knowledge_graph.html", mode="auto"):
    print(f"\n[graph] Building knowledge graph from: {directories}")
    kg = KnowledgeGraph(data_dir=DATA_DIR, mode=mode)
    for d in directories:
        fs = FSTools(allowed_roots=[d, SCRATCH_DIR])
        kg.ingest_directory(d, fs_tools=fs)
    if not kg._documents:
        print("[graph] No parseable files found."); return kg
    print(f"[graph] Ingested {len(kg._documents)} file(s). Building graph...")
    graph_mode = await kg.build()
    html       = await kg.visualize(output_html)
    print(f"[graph] Mode={graph_mode}  Stats={kg.stats()}")
    print(f"[graph] HTML → {html}")
    return kg


def run_agent(user_request, extra_dirs=None, user_id="default_user", kg=None):
    search_dirs = [d for d in (extra_dirs or [])+SEARCH_DIRS if Path(d).exists()]
    if not search_dirs: print("[agent] No accessible directories."); return None

    model = os.getenv("RLM_ROOT_MODEL","") or os.getenv("RLM_WORKER_MODEL","")
    llm   = get_llm_client(None, model)

    mem      = AgentMemory(user_id=user_id)
    memories = mem.search(user_request, limit=5)
    if memories: print(f"[agent] {len(memories)} relevant memory/memories retrieved")

    print(f"\n[agent] Scanning: {search_dirs}")
    all_files = discover_files(search_dirs)
    print(f"[agent] Found {len(all_files)} file(s)")
    if not all_files: return None

    relevant = select_relevant_files(user_request, all_files, llm)
    if not relevant:
        print("[agent] No relevant files found."); return None
    print(f"[agent] Selected:")
    for f in relevant: print(f"         ✓ {f['name']} ({f['size_kb']} KB)")

    fs           = FSTools(allowed_roots=search_dirs+[SCRATCH_DIR])
    extra_locals = {}
    loaded_files = []
    for i,f in enumerate(relevant):
        var = f"FILE_{i+1}"
        print(f"[agent] Loading {f['name']} → {var} ...", end=" ", flush=True)
        content = load_file_content(f["path"], fs)
        if content:
            extra_locals[var]           = content
            extra_locals[f"{var}_PATH"] = f["path"]
            extra_locals[f"{var}_NAME"] = f["name"]
            loaded_files.append(f)
            print(f"{len(content):,} chars")
        else:
            print("skipped")

    if not extra_locals: print("[agent] No file content loaded."); return None

    system_prompt = build_analysis_prompt(user_request, loaded_files, memories)
    rlm = RLM_REPL(
        allowed_roots=search_dirs+[SCRATCH_DIR],
        max_iterations=int(os.getenv("RLM_MAX_ITERATIONS","12")),
        extra_locals=extra_locals,
    )
    rlm._custom_system = system_prompt
    print(f"\n[agent] Running analysis (max {rlm.max_iterations} iterations)...")
    print("─"*60)
    result = rlm.completion(context=system_prompt, query=user_request)

    if result:
        mem.add([{"role":"user","content":user_request},
                 {"role":"assistant","content":result[:500]}],
                metadata={"files":[f["name"] for f in loaded_files],
                          "date": datetime.datetime.now().isoformat()})
        print(f"\n[memory] Stored interaction ({mem._backend_name})")
    return result


def interactive_loop(extra_dirs=None, kg=None):
    user_id = os.getenv("AGENT_USER_ID","default_user")
    mem     = AgentMemory(user_id=user_id)
    print("\n"+"═"*65)
    print("  AI-MINDS Agent  ·  RLM + Mem0 + Cognee")
    print(f"  Memory  : {mem._backend_name}")
    print(f"  Graph   : {'built' if kg and kg._built else 'not built'}")
    print("  Commands: memories | clear memory | graph | quit")
    print("═"*65)
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[agent] Goodbye."); break
        if not user_input: continue
        if user_input.lower() in ("quit","exit","q"):
            print("[agent] Goodbye."); break
        if user_input.lower() == "memories":
            mems = mem.get_all()
            print(f"\n{len(mems)} memories:" if mems else "No memories yet.")
            for i,m in enumerate(mems,1): print(f"  {i}. {str(m.get('memory',m))[:120]}")
            continue
        if user_input.lower() == "clear memory":
            mem.clear(); continue
        if user_input.lower() == "graph" and kg and kg._built:
            loop = asyncio.new_event_loop()
            p = loop.run_until_complete(kg.visualize()); loop.close()
            print(f"Graph → {p}"); continue

        result = run_agent(user_input, extra_dirs=extra_dirs, user_id=user_id, kg=kg)
        print("\n"+"─"*60+"\nRESULT\n"+"─"*60)
        print(result if result else "(No result)")
        print("─"*60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI-MINDS Natural Language File Agent")
    parser.add_argument("request",       nargs="?")
    parser.add_argument("--dir","-d",    action="append", dest="dirs", metavar="PATH")
    parser.add_argument("--build-graph", action="store_true")
    parser.add_argument("--graph-mode",  default="auto", choices=["auto","cognee","networkx"])
    parser.add_argument("--user",        default=os.getenv("AGENT_USER_ID","default_user"))
    args = parser.parse_args()

    extra_dirs  = args.dirs or []
    search_dirs = [d for d in extra_dirs+SEARCH_DIRS if Path(d).exists()]

    kg   = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    kg = loop.run_until_complete(
        build_knowledge_graph(
            search_dirs or [str(Path.home()/"Downloads")],
            output_html=str(Path(DATA_DIR)/"knowledge_graph.html"),
            mode=args.graph_mode,
        )
    )
    if args.build_graph: return

    if args.request:
        result = run_agent(args.request, extra_dirs=extra_dirs, user_id=args.user, kg=kg)
        print("\n"+"─"*60+"\n"+(result or "(No result)")+"\n"+"─"*60)
    else:
        interactive_loop(extra_dirs=extra_dirs, kg=kg)

if __name__ == "__main__":
    main()