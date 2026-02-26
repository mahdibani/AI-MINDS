from __future__ import annotations

import uuid
from pathlib import Path

from api.core.dependencies import container
from rlm.fs_tools import FSTools
from rlm.knowledge_graph import KnowledgeGraph


async def build_knowledge_graph(directory: str, mode: str) -> dict:
    directory = directory.replace("\\", "/")
    graph_id = uuid.uuid4().hex
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    html_path = str(data_dir / f"knowledge_graph_{graph_id}.html")

    fs_tools = FSTools(allowed_roots=[directory, "/tmp/rlm_scratch", str(Path.home())])
    kg = KnowledgeGraph(data_dir="./data", mode=mode)
    kg.ingest_directory(directory, fs_tools=fs_tools)
    await kg.build()
    await kg.visualize(html_path)
    stats = kg.stats()
    container.put_knowledge_graph(graph_id, kg)

    summary = (
        "Knowledge Graph Built\n"
        f"Mode: {stats.get('mode', 'unknown')}\n"
        f"Documents: {stats.get('documents', 0)}\n"
        f"Nodes: {stats.get('nodes', 0)}\n"
        f"Edges: {stats.get('edges', 0)}"
    )
    return {"graph_id": graph_id, "summary": summary, "html_path": html_path, "stats": stats}


async def search_knowledge_graph(graph_id: str, query: str, limit: int) -> list[str]:
    kg = container.get_knowledge_graph(graph_id)
    if kg is None:
        return [f"Graph session not found: {graph_id}. Build a graph first."]
    return await kg.search(query, limit=limit)
