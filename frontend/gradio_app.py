"""
app.py - Unified Gradio Interface for RLM Tools

Integrates all AI-MINDS capabilities:
- Budget Advisor
- File Analysis (PDF, DOCX, XLSX, CSV)
- Knowledge Graph Builder
- Agent Testing
- Memory Management
- REPL Playground

Architecture:
    frontend/
        app.py          ← This file
    backend/
        rlm/            ← RLM core modules
        budget_advisor.py
        agent.py
        run.py

Run: python app.py (from frontend/)
"""

import gradio as gr
import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add backend to path
FRONTEND_DIR = Path(__file__).parent
BACKEND_DIR = FRONTEND_DIR.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

print(f"[app] Frontend dir: {FRONTEND_DIR}")
print(f"[app] Backend dir: {BACKEND_DIR}")

# Import RLM components from backend
try:
    from rlm.fs_tools import FSTools
    from rlm.memory import AgentMemory
    from rlm.rlm_repl import RLM_REPL
    from rlm.agents import PromptPipeline
    print("[app] ✓ RLM components imported")
except Exception as e:
    print(f"[app] ✗ Error importing RLM: {e}")
    raise

# Import specialized tools
try:
    from budget_advisor import BudgetAdvisor
    BUDGET_AVAILABLE = True
    print("[app] ✓ Budget Advisor available")
except Exception as e:
    print(f"[app] ✗ Budget Advisor not available: {e}")
    BUDGET_AVAILABLE = False

try:
    from rlm.knowledge_graph import KnowledgeGraph
    KG_AVAILABLE = True
    print("[app] ✓ Knowledge Graph available")
except Exception as e:
    print(f"[app] ✗ Knowledge Graph not available: {e}")
    KG_AVAILABLE = False


# ============================================================================
# Global State
# ============================================================================

class AppState:
    def __init__(self):
        self.budget_advisor = None
        self.memory = AgentMemory("gradio_user")
        self.fs_tools = FSTools(allowed_roots=[
            str(Path.home() / "Downloads"),
            str(Path.home() / "Documents"),
            "/tmp/rlm_scratch"
        ])
        self.knowledge_graph = None
        self.conversation_history = []
        self.rlm_repl = None

state = AppState()


# ============================================================================
# Budget Advisor Tab
# ============================================================================

def init_budget_advisor(budget_file, safety_buffer):
    """Initialize budget advisor."""
    if not BUDGET_AVAILABLE:
        return "❌ Budget Advisor not available. Check budget_advisor.py"
    
    try:
        state.budget_advisor = BudgetAdvisor(
            budget_file=budget_file.replace("\\", "/"),
            user_id="gradio_user",
            safety_buffer=safety_buffer / 100.0
        )
        return f"✅ Budget Advisor initialized!\n📁 {budget_file}\n🛡️ Safety: {safety_buffer}%"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def budget_chat(message, history):
    """Handle budget advisor chat."""
    if state.budget_advisor is None:
        return history + [[message, "❌ Please initialize Budget Advisor first"]], ""
    
    # Parse message for purchase intent
    import re
    patterns = [
        r'buy\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)*?).*?\$?(\d+(?:\.\d+)?)',
        r'afford.*?(\w+(?:\s+\w+)*?).*?\$(\d+(?:\.\d+)?)',
        r'\$(\d+(?:\.\d+)?)\s+(\w+(?:\s+\w+)*)',
    ]
    
    item, price = None, None
    for pattern in patterns:
        match = re.search(pattern, message.lower())
        if match:
            groups = match.groups()
            try:
                item, price = groups[0].strip(), float(groups[1])
                break
            except:
                try:
                    item, price = groups[1].strip(), float(groups[0])
                    break
                except:
                    continue
    
    if item and price:
        result = state.budget_advisor.can_afford(item, price, explain=False)
        
        response = f"""## {"✅ Affordable" if result['affordable'] else "❌ Not Affordable"}: {item} (${price:.2f})

### 💰 Budget Snapshot
- **Monthly Income**: ${result.get('monthly_income', 0):,.2f}
- **Total Expenses**: ${result.get('total_expenses', 0):,.2f}
- **Available Funds**: ${result.get('available_funds', 0):,.2f}
- **Confidence**: {result.get('confidence', 0):.0%}

### 💡 {result.get('recommendation', 'No recommendation')}
"""
        
        if result.get('warnings'):
            response += "\n### ⚠️ Warnings\n" + "\n".join(f"- {w}" for w in result['warnings'])
        
        return history + [[message, response]], ""
    
    # Special commands
    if message.lower() == "summary":
        summary = state.budget_advisor.get_budget_summary()
        return history + [[message, summary]], ""
    
    return history + [[message, "Try: 'Can I afford a smartwatch for $250?'"]], ""


def get_budget_summary():
    """Get budget summary."""
    if state.budget_advisor is None:
        return "❌ Please initialize Budget Advisor first"
    return state.budget_advisor.get_budget_summary()


# ============================================================================
# File Analysis Tab
# ============================================================================

def list_files(directory):
    """List files in directory."""
    try:
        directory = directory.replace("\\", "/")
        result = state.fs_tools.list(directory)
        
        if result["error"]:
            return f"❌ Error: {result['error']}"
        
        output = f"## 📁 Files in {result['path']}\n\n"
        output += "| Name | Type | Size | Parseable |\n"
        output += "|------|------|------|----------|\n"
        
        for entry in result["entries"][:50]:  # Limit to 50
            size_kb = f"{entry['size']/1024:.1f} KB" if entry.get('size') else "—"
            parseable = "✅" if entry.get('parseable') else "—"
            output += f"| {entry['name']} | {entry['type']} | {size_kb} | {parseable} |\n"
        
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


def parse_file(filepath, max_chars):
    """Parse a file and extract text."""
    try:
        filepath = filepath.replace("\\", "/")
        result = state.fs_tools.parse(filepath, max_chars=max_chars)
        
        if result["error"]:
            return f"❌ Error: {result['error']}"
        
        output = f"""## 📄 File: {Path(filepath).name}

**Format**: {result['format']}  
**Size**: {result['size']:,} bytes  
**Truncated**: {result['truncated']}

### Content:
```
{result['text']}
```
"""
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


def analyze_files(directory, query):
    """Analyze files in directory using RLM."""
    try:
        directory = directory.replace("\\", "/")
        
        # Initialize RLM for file analysis
        rlm = RLM_REPL(
            allowed_roots=[directory, "/tmp/rlm_scratch"],
            max_iterations=8,
            extra_locals={"target_dir": directory}
        )
        
        analysis_query = f"""
Analyze the files in the directory stored in 'target_dir' variable.

User query: {query}

Use fs_list() to see available files, then fs_parse() to read them.
Provide a clear answer to the user's query based on the file contents.

Execute your analysis and emit FINAL(your_answer).
"""
        
        result = rlm.completion(
            context=f"File analysis in {directory}",
            query=analysis_query
        )
        
        return result or "No result from analysis"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# Knowledge Graph Tab
# ============================================================================

async def build_knowledge_graph_async(directory, mode):
    """Build knowledge graph from directory."""
    if not KG_AVAILABLE:
        return "❌ Knowledge Graph not available", None
    
    try:
        directory = directory.replace("\\", "/")
        
        state.knowledge_graph = KnowledgeGraph(data_dir="./data", mode=mode)
        state.knowledge_graph.ingest_directory(directory, fs_tools=state.fs_tools)
        
        await state.knowledge_graph.build()
        html_path = await state.knowledge_graph.visualize("./data/knowledge_graph.html")
        
        stats = state.knowledge_graph.stats()
        
        summary = f"""## ✅ Knowledge Graph Built!

**Mode**: {stats.get('mode', 'unknown')}  
**Documents**: {stats.get('documents', 0)}  
**Nodes**: {stats.get('nodes', 0)}  
**Edges**: {stats.get('edges', 0)}  
**File Nodes**: {stats.get('file_nodes', 0)}  
**Term Nodes**: {stats.get('term_nodes', 0)}

📊 Visualization saved to: {html_path}
"""
        
        return summary, html_path
    except Exception as e:
        return f"❌ Error: {str(e)}", None


def build_knowledge_graph(directory, mode):
    """Sync wrapper for knowledge graph building."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(build_knowledge_graph_async(directory, mode))
    loop.close()
    return result


async def search_knowledge_graph_async(query, limit):
    """Search the knowledge graph."""
    if state.knowledge_graph is None:
        return "❌ Please build knowledge graph first"
    
    try:
        results = await state.knowledge_graph.search(query, limit=limit)
        
        if not results:
            return f"No results found for: {query}"
        
        output = f"## 🔍 Search Results for '{query}'\n\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result}\n\n"
        
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


def search_knowledge_graph(query, limit):
    """Sync wrapper for knowledge graph search."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(search_knowledge_graph_async(query, limit))
    loop.close()
    return result


# ============================================================================
# Memory Tab
# ============================================================================

def add_memory(text, metadata_json):
    """Add a memory."""
    try:
        metadata = eval(metadata_json) if metadata_json else None
        state.memory.add(text, metadata=metadata)
        return f"✅ Memory added: {text[:100]}..."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def search_memory(query, limit):
    """Search memories."""
    try:
        results = state.memory.search(query, limit=limit)
        
        if not results:
            return f"No memories found for: {query}"
        
        output = f"## 🔍 Memory Search: '{query}'\n\n"
        for i, mem in enumerate(results, 1):
            output += f"{i}. {mem}\n\n"
        
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_all_memories():
    """Get all memories."""
    try:
        memories = state.memory.get_all()
        
        if not memories:
            return "No memories stored yet."
        
        output = f"## 🧠 All Memories ({len(memories)})\n\n"
        for i, mem in enumerate(memories, 1):
            mem_text = mem.get('memory', str(mem))
            output += f"{i}. {mem_text}\n\n"
        
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


def clear_memories():
    """Clear all memories."""
    try:
        state.memory.clear()
        return "✅ All memories cleared"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# REPL Playground Tab
# ============================================================================

def run_repl_code(code, context):
    """Execute code in REPL environment."""
    try:
        if state.rlm_repl is None:
            state.rlm_repl = RLM_REPL(
                allowed_roots=[
                    str(Path.home() / "Downloads"),
                    "/tmp/rlm_scratch"
                ],
                max_iterations=5
            )
        
        query = f"""
Execute this code:

```python
{code}
```

FINAL(result of execution)
"""
        
        result = state.rlm_repl.completion(
            context=context or "REPL execution",
            query=query
        )
        
        return result or "No output"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def run_file_analysis_repl(directory, task):
    """Run file analysis using REPL."""
    try:
        directory = directory.replace("\\", "/")
        
        rlm = RLM_REPL(
            allowed_roots=[directory, "/tmp/rlm_scratch"],
            max_iterations=10,
            extra_locals={"work_dir": directory}
        )
        
        query = f"""
Task: {task}

Working directory is in 'work_dir' variable.

Use these tools:
- fs_list(work_dir) - list files
- fs_parse(file_path) - read files
- llm_query(prompt) - ask sub-LLM questions

Complete the task and emit FINAL(your_answer).
"""
        
        result = rlm.completion(
            context=f"File analysis task",
            query=query
        )
        
        return result or "No result"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# Agent Pipeline Tab  
# ============================================================================

def run_agent_pipeline(prompt, task, chunk_size, max_workers):
    """Run the agent pipeline on large prompt."""
    try:
        pipeline = PromptPipeline(
            chunk_size=chunk_size,
            max_workers=max_workers
        )
        
        result = pipeline.run(prompt=prompt, task=task, parallel=True)
        
        output = f"""## ✅ Pipeline Complete

**Chunks Processed**: {result.chunks_processed}  
**Total Iterations**: {result.total_tokens}

### Final Output:
{result.final_output}
"""
        
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# Testing Tab
# ============================================================================

def run_budget_tests(verbose):
    """Run budget advisor tests."""
    try:
        import subprocess
        
        # Run tests from backend directory
        test_file = BACKEND_DIR / "test_budget_advisor.py"
        
        if not test_file.exists():
            return f"❌ Test file not found: {test_file}"
        
        cmd = ["pytest", str(test_file), "-v"]
        if verbose:
            cmd.append("-s")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BACKEND_DIR))
        return f"```\n{result.stdout}\n{result.stderr}\n```"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def run_rlm_tests(verbose):
    """Run RLM REPL tests."""
    try:
        import subprocess
        
        # Run tests from backend directory
        test_file = BACKEND_DIR / "test_rlm_repl.py"
        
        if not test_file.exists():
            return f"❌ Test file not found: {test_file}"
        
        cmd = ["pytest", str(test_file), "-v"]
        if verbose:
            cmd.append("-s")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BACKEND_DIR))
        return f"```\n{result.stdout}\n{result.stderr}\n```"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# Dark Theme CSS
# ============================================================================

DARK_CSS = """
/* ── Root palette ── */
:root {
    --bg-primary:    #0d0f13;
    --bg-secondary:  #141720;
    --bg-surface:    #1c1f2a;
    --bg-elevated:   #232738;
    --border:        #2e3347;
    --accent:        #5b7fff;
    --accent-glow:   rgba(91, 127, 255, 0.18);
    --accent-dim:    #3d57cc;
    --success:       #34d399;
    --warning:       #fbbf24;
    --danger:        #f87171;
    --text-primary:  #e8eaf0;
    --text-secondary:#9aa0b8;
    --text-muted:    #5c6280;
    --font-mono:     'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
}

/* ── Base overrides ── */
body, .gradio-container {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', 'Outfit', system-ui, sans-serif !important;
}

/* ── Header ── */
.gradio-container > .prose h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 40%, #5b7fff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}

/* ── Panels & blocks ── */
.block, .form, .panel, .wrap, .gap,
.svelte-1gfkfd6, .svelte-10ogue4 {
    background: var(--bg-secondary) !important;
    border-color: var(--border) !important;
}

/* ── Tabs ── */
.tab-nav {
    background: var(--bg-surface) !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: 10px 10px 0 0 !important;
    overflow: hidden;
}

.tab-nav button {
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 18px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px;
    transition: color 0.2s, background 0.2s !important;
}

.tab-nav button:hover {
    color: var(--text-primary) !important;
    background: var(--accent-glow) !important;
}

.tab-nav button.selected {
    color: var(--accent) !important;
    background: var(--bg-elevated) !important;
    border-bottom: 2px solid var(--accent) !important;
    font-weight: 600 !important;
}

/* ── Inputs ── */
input[type="text"], input[type="number"],
textarea, .input-wrap textarea, .scroll-hide {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: inherit !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

input[type="text"]:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}

/* ── Labels ── */
label span, .label-wrap span {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}

/* ── Buttons ── */
button.primary, .btn-primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
}

button.primary:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
}

button.primary:active {
    transform: translateY(0) !important;
}

button.stop, .btn-stop {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--danger) !important;
    color: var(--danger) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background 0.2s, box-shadow 0.2s !important;
}

button.stop:hover {
    background: rgba(248, 113, 113, 0.1) !important;
    box-shadow: 0 0 0 3px rgba(248, 113, 113, 0.15) !important;
}

button:not(.primary):not(.stop):not(.selected) {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    transition: border-color 0.2s, background 0.2s !important;
}

button:not(.primary):not(.stop):not(.selected):hover {
    border-color: var(--accent) !important;
    background: var(--accent-glow) !important;
}

/* ── Chatbot ── */
.chatbot {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

.chatbot .message.user {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px 10px 2px 10px !important;
}

.chatbot .message.bot {
    background: linear-gradient(135deg, #1a1e30 0%, #1c2035 100%) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px 10px 10px 2px !important;
}

/* ── Markdown output ── */
.prose, .md {
    color: var(--text-primary) !important;
}

.prose h1, .prose h2, .prose h3 {
    color: var(--text-primary) !important;
}

.prose code, code {
    background: var(--bg-elevated) !important;
    color: #a5b4fc !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82em !important;
    padding: 1px 5px !important;
}

.prose pre, pre {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 14px !important;
}

.prose table {
    border-color: var(--border) !important;
}

.prose th {
    background: var(--bg-elevated) !important;
    color: var(--text-secondary) !important;
}

.prose td {
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}

.prose tr:nth-child(even) {
    background: var(--bg-surface) !important;
}

/* ── Code editor ── */
.codemirror-wrapper, .cm-editor {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
}

.cm-gutters {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
}

.cm-content {
    color: var(--text-primary) !important;
}

/* ── Sliders ── */
input[type="range"] {
    accent-color: var(--accent) !important;
}

/* ── Checkboxes ── */
input[type="checkbox"] {
    accent-color: var(--accent) !important;
}

/* ── Radio buttons ── */
.gr-radio input[type="radio"]:checked + span,
.radio-group input[type="radio"]:checked + label {
    color: var(--accent) !important;
}

/* ── File upload ── */
.upload-container, .file-preview {
    background: var(--bg-elevated) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    transition: border-color 0.2s !important;
}

.upload-container:hover {
    border-color: var(--accent) !important;
}

/* ── Scrollbars ── */
* {
    scrollbar-width: thin;
    scrollbar-color: var(--border) var(--bg-primary);
}
*::-webkit-scrollbar { width: 6px; height: 6px; }
*::-webkit-scrollbar-track { background: var(--bg-primary); }
*::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
*::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Footer ── */
footer .prose {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}

/* ── Gradio dark mode toggle (force dark) ── */
.dark { color-scheme: dark; }
"""


# ============================================================================
# Create Gradio Interface
# ============================================================================

def create_app():
    """Create the full Gradio interface."""
    
    with gr.Blocks(
        title="AI-MINDS RLM Tools",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        ).set(
            # Core backgrounds
            body_background_fill="#0d0f13",
            body_background_fill_dark="#0d0f13",
            block_background_fill="#141720",
            block_background_fill_dark="#141720",
            block_border_color="#2e3347",
            block_border_color_dark="#2e3347",
            # Input fields
            input_background_fill="#232738",
            input_background_fill_dark="#232738",
            input_border_color="#2e3347",
            input_border_color_dark="#2e3347",
            input_border_color_focus="#5b7fff",
            input_border_color_focus_dark="#5b7fff",
            # Buttons
            button_primary_background_fill="#5b7fff",
            button_primary_background_fill_dark="#5b7fff",
            button_primary_background_fill_hover="#4a6aee",
            button_primary_background_fill_hover_dark="#4a6aee",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill="#232738",
            button_secondary_background_fill_dark="#232738",
            button_secondary_border_color="#2e3347",
            button_secondary_border_color_dark="#2e3347",
            button_secondary_text_color="#e8eaf0",
            button_secondary_text_color_dark="#e8eaf0",
            # Text
            body_text_color="#e8eaf0",
            body_text_color_dark="#e8eaf0",
            body_text_color_subdued="#9aa0b8",
            body_text_color_subdued_dark="#9aa0b8",
            # Borders & shadows
            border_color_primary="#2e3347",
            border_color_primary_dark="#2e3347",
            shadow_drop="0 4px 24px rgba(0,0,0,0.4)",
            shadow_drop_lg="0 8px 40px rgba(0,0,0,0.5)",
            # Block labels
            block_label_text_color="#9aa0b8",
            block_label_text_color_dark="#9aa0b8",
        ),
        css=DARK_CSS,
    ) as app:
        
        gr.Markdown("""
        # 🤖 AI-MINDS — RLM Tools Suite
        ### Unified interface for all Recursive Language Model capabilities
        
        **Available Tools**: Budget Advisor • File Analysis • Knowledge Graph • Memory • REPL • Testing
        """)
        
        with gr.Tabs():
            
            # ================================================================
            # TAB 1: Budget Advisor
            # ================================================================
            with gr.Tab("💰 Budget Advisor"):
                gr.Markdown("## Smart Financial Assistant")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        budget_file = gr.Textbox(
                            label="Budget File",
                            value=str(Path.home() / "Downloads" / "budget.xlsx"),
                            placeholder="C:/Users/bani_/Downloads/budget.xlsx"
                        )
                        safety_buffer = gr.Slider(0, 50, 20, step=5, label="Safety Buffer (%)")
                        init_budget_btn = gr.Button("🚀 Initialize", variant="primary")
                        budget_status = gr.Textbox(label="Status", lines=3, interactive=False)
                        
                        gr.Markdown("### Quick Actions")
                        summary_btn = gr.Button("📊 View Summary")
                        summary_out = gr.Textbox(label="Summary", lines=8, visible=False)
                    
                    with gr.Column(scale=2):
                        budget_chatbot = gr.Chatbot(label="Chat", height=500)
                        budget_msg = gr.Textbox(label="Message", placeholder="Can I afford a smartwatch for $250?")
                        with gr.Row():
                            budget_send = gr.Button("Send", variant="primary")
                            budget_clear = gr.Button("Clear")
                
                init_budget_btn.click(init_budget_advisor, [budget_file, safety_buffer], budget_status)
                budget_send.click(budget_chat, [budget_msg, budget_chatbot], [budget_chatbot, budget_msg])
                budget_msg.submit(budget_chat, [budget_msg, budget_chatbot], [budget_chatbot, budget_msg])
                budget_clear.click(lambda: [], outputs=budget_chatbot)
                summary_btn.click(get_budget_summary, outputs=summary_out).then(
                    lambda: gr.update(visible=True), outputs=summary_out
                )
            
            # ================================================================
            # TAB 2: File Analysis
            # ================================================================
            with gr.Tab("📁 File Analysis"):
                gr.Markdown("## Parse and Analyze Files (PDF, DOCX, XLSX, CSV)")
                
                with gr.Row():
                    with gr.Column():
                        file_dir = gr.Textbox(
                            label="Directory",
                            value=str(Path.home() / "Downloads"),
                            placeholder="C:/Users/bani_/Downloads"
                        )
                        list_btn = gr.Button("📂 List Files", variant="primary")
                        file_list_out = gr.Markdown(label="Files")
                    
                    with gr.Column():
                        file_path = gr.Textbox(label="File Path", placeholder="Full path to file")
                        max_chars = gr.Slider(1000, 100000, 50000, step=1000, label="Max Characters")
                        parse_btn = gr.Button("📄 Parse File", variant="primary")
                        parse_out = gr.Markdown(label="Content")
                
                gr.Markdown("### AI-Powered Analysis")
                with gr.Row():
                    analysis_query = gr.Textbox(
                        label="Query",
                        placeholder="Summarize all files in this directory",
                        scale=3
                    )
                    analyze_btn = gr.Button("🤖 Analyze", variant="primary", scale=1)
                
                analysis_out = gr.Markdown(label="Analysis Result")
                
                list_btn.click(list_files, file_dir, file_list_out)
                parse_btn.click(parse_file, [file_path, max_chars], parse_out)
                analyze_btn.click(analyze_files, [file_dir, analysis_query], analysis_out)
            
            # ================================================================
            # TAB 3: Knowledge Graph
            # ================================================================
            with gr.Tab("🕸️ Knowledge Graph"):
                gr.Markdown("## Build and Search Knowledge Graphs from Documents")
                
                with gr.Row():
                    with gr.Column():
                        kg_dir = gr.Textbox(
                            label="Directory",
                            value=str(Path.home() / "Downloads")
                        )
                        kg_mode = gr.Radio(
                            ["auto", "cognee", "networkx"],
                            value="auto",
                            label="Graph Mode"
                        )
                        build_kg_btn = gr.Button("🏗️ Build Graph", variant="primary")
                        kg_status = gr.Markdown(label="Status")
                        kg_viz = gr.File(label="Visualization HTML")
                    
                    with gr.Column():
                        kg_query = gr.Textbox(label="Search Query", placeholder="budget expenses")
                        kg_limit = gr.Slider(1, 20, 5, step=1, label="Max Results")
                        search_kg_btn = gr.Button("🔍 Search", variant="primary")
                        kg_results = gr.Markdown(label="Search Results")
                
                build_kg_btn.click(build_knowledge_graph, [kg_dir, kg_mode], [kg_status, kg_viz])
                search_kg_btn.click(search_knowledge_graph, [kg_query, kg_limit], kg_results)
            
            # ================================================================
            # TAB 4: Memory
            # ================================================================
            with gr.Tab("🧠 Memory"):
                gr.Markdown("## Manage Agent Memory (Mem0)")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Add Memory")
                        mem_text = gr.Textbox(label="Text", placeholder="User prefers CSV format", lines=3)
                        mem_meta = gr.Textbox(label="Metadata (JSON)", placeholder='{"category": "preference"}')
                        add_mem_btn = gr.Button("➕ Add Memory", variant="primary")
                        add_mem_out = gr.Textbox(label="Status", lines=2)
                    
                    with gr.Column():
                        gr.Markdown("### Search Memory")
                        search_query = gr.Textbox(label="Query", placeholder="budget")
                        search_limit = gr.Slider(1, 20, 5, step=1, label="Max Results")
                        search_mem_btn = gr.Button("🔍 Search", variant="primary")
                        search_mem_out = gr.Markdown(label="Results")
                
                gr.Markdown("### All Memories")
                with gr.Row():
                    get_all_btn = gr.Button("📋 Get All Memories")
                    clear_mem_btn = gr.Button("🗑️ Clear All", variant="stop")
                
                all_mem_out = gr.Markdown(label="All Memories")
                
                add_mem_btn.click(add_memory, [mem_text, mem_meta], add_mem_out)
                search_mem_btn.click(search_memory, [search_query, search_limit], search_mem_out)
                get_all_btn.click(get_all_memories, outputs=all_mem_out)
                clear_mem_btn.click(clear_memories, outputs=all_mem_out)
            
            # ================================================================
            # TAB 5: REPL Playground
            # ================================================================
            with gr.Tab("⚡ REPL Playground"):
                gr.Markdown("## Interactive Python REPL with RLM")
                
                gr.Markdown("### Code Execution")
                repl_code = gr.Code(
                    label="Python Code",
                    language="python",
                    value="# Example\nresult = 100 * 25\nprint(f'Result: {result}')",
                    lines=10
                )
                repl_context = gr.Textbox(label="Context (optional)", placeholder="Working on financial analysis")
                run_repl_btn = gr.Button("▶️ Run Code", variant="primary")
                repl_out = gr.Textbox(label="Output", lines=10)
                
                gr.Markdown("### File Analysis Task")
                with gr.Row():
                    task_dir = gr.Textbox(label="Directory", value=str(Path.home() / "Downloads"), scale=2)
                    task_desc = gr.Textbox(label="Task", placeholder="Find all CSV files and sum their totals", scale=3)
                
                run_task_btn = gr.Button("🚀 Run Task", variant="primary")
                task_out = gr.Textbox(label="Result", lines=10)
                
                run_repl_btn.click(run_repl_code, [repl_code, repl_context], repl_out)
                run_task_btn.click(run_file_analysis_repl, [task_dir, task_desc], task_out)
            
            # ================================================================
            # TAB 6: Agent Pipeline
            # ================================================================
            with gr.Tab("🔄 Agent Pipeline"):
                gr.Markdown("## Process Large Prompts with Sub-Agents")
                
                pipeline_prompt = gr.Textbox(
                    label="Large Prompt",
                    placeholder="Paste large document here...",
                    lines=10
                )
                pipeline_task = gr.Textbox(
                    label="Task",
                    placeholder="Extract all action items and dates",
                    lines=3
                )
                
                with gr.Row():
                    pipeline_chunks = gr.Slider(10000, 200000, 80000, step=10000, label="Chunk Size")
                    pipeline_workers = gr.Slider(1, 10, 4, step=1, label="Max Workers")
                
                pipeline_btn = gr.Button("🚀 Run Pipeline", variant="primary")
                pipeline_out = gr.Markdown(label="Result")
                
                pipeline_btn.click(
                    run_agent_pipeline,
                    [pipeline_prompt, pipeline_task, pipeline_chunks, pipeline_workers],
                    pipeline_out
                )
            
            # ================================================================
            # TAB 7: Testing
            # ================================================================
            with gr.Tab("🧪 Testing"):
                gr.Markdown("## Agent-Based Testing with Judge Agents")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Budget Advisor Tests")
                        budget_test_verbose = gr.Checkbox(label="Verbose Output", value=False)
                        run_budget_test_btn = gr.Button("▶️ Run Budget Tests", variant="primary")
                        budget_test_out = gr.Code(label="Results", language="shell", lines=15)
                    
                    with gr.Column():
                        gr.Markdown("### RLM REPL Tests")
                        rlm_test_verbose = gr.Checkbox(label="Verbose Output", value=False)
                        run_rlm_test_btn = gr.Button("▶️ Run RLM Tests", variant="primary")
                        rlm_test_out = gr.Code(label="Results", language="shell", lines=15)
                
                run_budget_test_btn.click(run_budget_tests, budget_test_verbose, budget_test_out)
                run_rlm_test_btn.click(run_rlm_tests, rlm_test_verbose, rlm_test_out)
        
        gr.Markdown("""
        ---
        **AI-MINDS RLM Tools** | Powered by Ollama + RLM + Mem0 + Cognee  
        All processing happens locally with your Ollama models 🔒
        """)
    
    return app


# ============================================================================
# Launch
# ============================================================================

if __name__ == "__main__":
    app = create_app()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              AI-MINDS RLM TOOLS SUITE                         ║
    ║                                                               ║
    ║  🚀 Starting unified web interface...                        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    app.launch(
        server_name="127.0.0.1",  # Use localhost for Windows
        server_port=7860,
        share=False,
        show_error=True
    )