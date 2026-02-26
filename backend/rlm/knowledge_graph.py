"""
knowledge_graph.py – Cognee knowledge graph layer with graceful fallback.

Ingests files from any directory, builds a semantic knowledge graph,
and exports an interactive HTML visualisation.

Architecture:
  Primary:  cognee  (LLM-based entity/relation extraction → graph DB)
  Fallback: networkx (structural graph: files as nodes, shared-terms as edges)

Usage:
    from rlm.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    await kg.ingest_directory("C:/Users/bani/Downloads")
    await kg.build()
    html_path = await kg.visualize("./data/graph.html")
    results   = await kg.search("budget expenses")
"""

from __future__ import annotations

import asyncio
import os
import json
import hashlib
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import cognee
    from cognee.api.v1.visualize.visualize import visualize_graph as _cognee_visualize
    _COGNEE_AVAILABLE = True
except ImportError:
    _COGNEE_AVAILABLE = False

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

try:
    from pyvis.network import Network as _PyvisNetwork
    _PYVIS_AVAILABLE = True
except ImportError:
    _PYVIS_AVAILABLE = False


# ── Cognee configurator ───────────────────────────────────────────────────────

def _configure_cognee_for_ollama():
    """
    Apply Ollama configuration to cognee at runtime.
    All env vars already loaded from .env.
    """
    ollama_url  = os.getenv("LLM_ENDPOINT",
                            os.getenv("RLM_API_URL", "http://localhost:11434/v1"))
    llm_model   = os.getenv("LLM_MODEL",
                            os.getenv("RLM_ROOT_MODEL", "gemma3:latest"))
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    embed_dims  = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

    # Cognee picks these up as env vars via its Pydantic settings
    os.environ.setdefault("LLM_PROVIDER",          "ollama")
    os.environ.setdefault("LLM_MODEL",             llm_model)
    os.environ.setdefault("LLM_ENDPOINT",          ollama_url)
    os.environ.setdefault("LLM_API_KEY",           "ollama")
    os.environ.setdefault("EMBEDDING_PROVIDER",    "ollama")
    os.environ.setdefault("EMBEDDING_MODEL",       embed_model)
    os.environ.setdefault("EMBEDDING_ENDPOINT",    ollama_url.replace("/v1", "/api/embeddings"))
    os.environ.setdefault("EMBEDDING_DIMENSIONS",  str(embed_dims))
    os.environ.setdefault("HUGGINGFACE_TOKENIZER", "bert-base-uncased")

    # Use local SQLite + LanceDB so no external DB is needed
    os.environ.setdefault("GRAPH_DATABASE_PROVIDER",     "networkx")
    os.environ.setdefault("VECTOR_DB_PROVIDER",          "lancedb")
    os.environ.setdefault("LANCEDB_URI",                 "./data/cognee_lancedb")


# ── Fallback: pure networkx graph builder ─────────────────────────────────────

def _build_networkx_graph(documents: List[Dict]) -> "nx.Graph":
    """
    Build a structural graph from document metadata + content without LLM.

    Nodes:
      - Each file  (type=file)
      - Each unique word/term with len>5, freq>1 (type=term)

    Edges:
      - file → term  if term appears in file content
      - file → file  if they share 3+ terms (co-occurrence similarity)
    """
    if not _NX_AVAILABLE:
        raise ImportError("networkx not installed. Run: uv add networkx")

    G = nx.Graph()
    term_files: Dict[str, List[str]] = {}

    for doc in documents:
        fname = doc["name"]
        G.add_node(fname, type="file",
                   size_kb=doc.get("size_kb", 0),
                   ext=doc.get("ext", ""),
                   modified=doc.get("modified", ""),
                   title=fname)

        # Extract significant terms from content
        content = doc.get("content", "")
        words   = [w.lower().strip(".,;:!?\"'()[]") for w in content.split()]
        freq: Dict[str, int] = {}
        for w in words:
            if len(w) > 4:
                freq[w] = freq.get(w, 0) + 1

        top_terms = sorted(freq, key=lambda w: freq[w], reverse=True)[:30]
        for term in top_terms:
            if term not in G:
                G.add_node(term, type="term", title=f'"{term}"')
            G.add_edge(fname, term, weight=freq[term])
            term_files.setdefault(term, []).append(fname)

    # File–file edges via shared significant terms
    for term, files in term_files.items():
        if len(files) >= 2:
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    if G.has_edge(files[i], files[j]):
                        G[files[i]][files[j]]["weight"] += 1
                    else:
                        G.add_edge(files[i], files[j], weight=1, shared_term=term)

    return G


def _export_networkx_html(G: "nx.Graph", output_path: str) -> str:
    """Export a networkx graph to an interactive pyvis HTML file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if _PYVIS_AVAILABLE:
        net = _PyvisNetwork(height="750px", width="100%",
                            bgcolor="#1a1a2e", font_color="white",
                            notebook=False)
        net.from_nx(G)

        # Style nodes by type
        for node in net.nodes:
            nid  = node["id"]
            data = G.nodes[nid]
            if data.get("type") == "file":
                ext   = data.get("ext", "")
                color = {"csv": "#00b4d8", "pdf": "#e63946",
                         "docx": "#457b9d", "xlsx": "#2a9d8f",
                         "txt": "#f4a261", "md": "#e9c46a"}.get(ext, "#a8dadc")
                node.update({"color": color, "size": 25, "shape": "box",
                             "font": {"size": 14, "color": "white"}})
            else:
                node.update({"color": "#6c757d", "size": 8, "shape": "dot",
                             "font": {"size": 10, "color": "#adb5bd"}})

        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 150
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 200}
          }
        }
        """)
        net.save_graph(output_path)

    else:
        # Pure HTML fallback using D3-force via CDN
        nodes_json = json.dumps([
            {"id": n, "group": 1 if G.nodes[n].get("type") == "file" else 2,
             "label": n[:30]}
            for n in G.nodes
        ])
        links_json = json.dumps([
            {"source": u, "target": v}
            for u, v in G.edges
        ])
        html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>AI-MINDS Knowledge Graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  body {{ background: #1a1a2e; margin: 0; }}
  svg  {{ width: 100vw; height: 100vh; }}
  .file-node  {{ fill: #00b4d8; }}
  .term-node  {{ fill: #6c757d; }}
  line  {{ stroke: #444; stroke-opacity: .6; }}
  text  {{ fill: #fff; font: 11px sans-serif; pointer-events: none; }}
</style>
</head><body>
<svg id="graph"></svg>
<script>
const nodes = {nodes_json};
const links = {links_json};
const svg   = d3.select("#graph");
const W = window.innerWidth, H = window.innerHeight;
const sim = d3.forceSimulation(nodes)
  .force("link",   d3.forceLink(links).id(d=>d.id).distance(80))
  .force("charge", d3.forceManyBody().strength(-120))
  .force("center", d3.forceCenter(W/2, H/2));
const link = svg.append("g").selectAll("line").data(links).join("line");
const node = svg.append("g").selectAll("circle").data(nodes).join("circle")
  .attr("r",     d => d.group===1 ? 14 : 6)
  .attr("class", d => d.group===1 ? "file-node" : "term-node")
  .call(d3.drag()
    .on("start", (e,d) => {{ if(!e.active) sim.alphaTarget(.3).restart(); d.fx=d.x; d.fy=d.y; }})
    .on("drag",  (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
    .on("end",   (e,d) => {{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }}));
const label = svg.append("g").selectAll("text").data(nodes).join("text")
  .text(d=>d.label).attr("dy","0.35em");
sim.on("tick", () => {{
  link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
      .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  node.attr("cx",d=>d.x).attr("cy",d=>d.y);
  label.attr("x",d=>d.x+12).attr("y",d=>d.y);
}});
</script></body></html>"""
        Path(output_path).write_text(html, encoding="utf-8")

    return output_path


# ── Main KnowledgeGraph class ─────────────────────────────────────────────────

class KnowledgeGraph:
    """
    Knowledge graph over a set of files.

    Two modes:
      "cognee"   – full LLM-based entity/relation extraction (requires cognee + good model)
      "networkx" – structural co-occurrence graph (always works, no LLM needed)

    The mode is auto-selected: tries cognee first, falls back to networkx.
    """

    def __init__(self, data_dir: str = "./data", mode: str = "auto"):
        self.data_dir  = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._documents: List[Dict] = []
        self._graph: Optional[Any]  = None   # nx.Graph or cognee graph
        self._mode  = mode   # "auto" | "cognee" | "networkx"
        self._built = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_file(self, path: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a single file's content to the pending ingestion queue.

        path:    absolute file path (used as unique id)
        content: extracted plain text (from FSTools.parse)
        """
        p    = Path(path)
        meta = metadata or {}
        self._documents.append({
            "id":       hashlib.md5(path.encode()).hexdigest()[:8],
            "name":     p.name,
            "path":     str(p),
            "ext":      p.suffix.lstrip(".").lower(),
            "size_kb":  meta.get("size_kb", 0),
            "modified": meta.get("modified", ""),
            "content":  content[:50_000],   # cap for graph building
        })
        print(f"[kg] Queued for ingestion: {p.name} ({len(content):,} chars)")

    def ingest_directory(self, directory: str, fs_tools=None):
        """
        Scan a directory using FSTools and queue all parseable files.
        Requires an FSTools instance (from rlm.fs_tools).
        """
        if fs_tools is None:
            from rlm.fs_tools import FSTools
            fs_tools = FSTools(allowed_roots=[directory])

        dir_fwd = directory.replace("\\", "/")
        listing = fs_tools.list(dir_fwd)
        if listing["error"]:
            print(f"[kg] Cannot list {directory}: {listing['error']}")
            return

        for entry in listing["entries"]:
            if entry["type"] != "file" or not entry.get("parseable"):
                continue
            fpath = dir_fwd + "/" + entry["name"]
            r = fs_tools.parse(fpath, max_chars=40_000)
            if r["error"] or not r["text"]:
                continue
            self.ingest_file(
                path=fpath,
                content=r["text"],
                metadata={
                    "size_kb":  round((entry["size"] or 0) / 1024, 1),
                    "modified": (datetime.datetime.fromtimestamp(entry["modified"]).strftime("%Y-%m-%d")
                                 if entry["modified"] else "n/a"),
                },
            )

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------

    async def build(self) -> str:
        """
        Build the knowledge graph from all ingested documents.
        Returns the mode used: "cognee" or "networkx".
        """
        if not self._documents:
            print("[kg] No documents queued – nothing to build")
            return "empty"

        # Determine mode
        use_cognee = False
        if self._mode == "auto":
            use_cognee = _COGNEE_AVAILABLE
        elif self._mode == "cognee":
            use_cognee = True

        if use_cognee:
            result = await self._build_cognee()
            if result:
                return "cognee"
            print("[kg] cognee build failed – falling back to networkx")

        self._build_networkx()
        return "networkx"

    async def _build_cognee(self) -> bool:
        try:
            _configure_cognee_for_ollama()
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)

            for doc in self._documents:
                await cognee.add(doc["content"], dataset_name=doc["name"])

            print(f"[kg] cognee.cognify() on {len(self._documents)} document(s)...")
            await cognee.cognify()
            self._built = True
            self._graph = "cognee"
            print("[kg] cognee graph built [ok]")
            return True
        except Exception as e:
            print(f"[kg] cognee build error: {e}")
            return False

    def _build_networkx(self):
        if not _NX_AVAILABLE:
            print("[kg] networkx not available – skipping graph build")
            return
        print(f"[kg] Building networkx graph from {len(self._documents)} document(s)...")
        self._graph = _build_networkx_graph(self._documents)
        self._built = True
        n = self._graph.number_of_nodes()
        e = self._graph.number_of_edges()
        print(f"[kg] networkx graph built [ok]  nodes={n}  edges={e}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(self, query: str, limit: int = 10) -> List[str]:
        """Search the knowledge graph. Returns a list of text results."""
        if not self._built:
            return ["Graph not built yet. Call build() first."]

        if self._graph == "cognee" and _COGNEE_AVAILABLE:
            try:
                from cognee import SearchType
                results = await cognee.search(SearchType.INSIGHTS, query_text=query)
                return [str(r) for r in results[:limit]]
            except Exception as e:
                return [f"cognee search error: {e}"]

        if _NX_AVAILABLE and isinstance(self._graph, nx.Graph):
            q = query.lower()
            hits = []
            # Direct node name match
            for node, data in self._graph.nodes(data=True):
                if q in node.lower():
                    neighbours = list(self._graph.neighbors(node))
                    hits.append(f"[{data.get('type','?')}] {node}  ->  {', '.join(str(n) for n in neighbours[:8])}")
                    if len(hits) >= limit:
                        break
            return hits or [f"No nodes matching '{query}' found in graph."]

        return ["Graph not available."]

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    async def visualize(self, output_path: Optional[str] = None) -> str:
        """
        Export the graph to an interactive HTML file.
        Returns the path to the HTML file.
        """
        if output_path is None:
            output_path = str(self.data_dir / "knowledge_graph.html")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if not self._built:
            msg = "<h2 style='color:white;font-family:sans-serif'>Graph not built yet.</h2>"
            Path(output_path).write_text(msg)
            return output_path

        # cognee built-in visualizer
        if self._graph == "cognee" and _COGNEE_AVAILABLE:
            try:
                await _cognee_visualize(output_path)
                print(f"[kg] cognee graph exported -> {output_path}")
                return output_path
            except Exception as e:
                print(f"[kg] cognee visualize failed ({e}) – falling back to networkx export")
                self._build_networkx()

        # networkx / pyvis export
        if _NX_AVAILABLE and isinstance(self._graph, nx.Graph):
            _export_networkx_html(self._graph, output_path)
            print(f"[kg] Graph exported -> {output_path}")
            return output_path

        Path(output_path).write_text("<p>No graph available.</p>")
        return output_path

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        s = {
            "documents": len(self._documents),
            "built":     self._built,
            "mode":      self._graph if isinstance(self._graph, str) else "networkx",
        }
        if _NX_AVAILABLE and isinstance(self._graph, nx.Graph):
            s["nodes"] = self._graph.number_of_nodes()
            s["edges"] = self._graph.number_of_edges()
            s["file_nodes"]  = sum(1 for _, d in self._graph.nodes(data=True) if d.get("type") == "file")
            s["term_nodes"]  = sum(1 for _, d in self._graph.nodes(data=True) if d.get("type") == "term")
        return s
