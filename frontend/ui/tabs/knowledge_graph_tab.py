from __future__ import annotations

import gradio as gr
from pathlib import Path


def build_knowledge_graph_tab(client) -> None:
    with gr.Tab("Knowledge Graph"):
        with gr.Row():
            directory = gr.Textbox(label="Directory", value=str(Path.home() / "Downloads"))
            mode = gr.Radio(["auto", "cognee", "networkx"], value="auto", label="Mode")
            build_btn = gr.Button("Build Graph", variant="primary")
        summary_out = gr.Markdown()
        graph_id_state = gr.State("")
        graph_id_out = gr.Textbox(label="Graph Session ID", interactive=False)
        html_path_out = gr.Textbox(label="HTML Path", interactive=False)
        html_file_out = gr.File(label="Visualization HTML")
        stats_out = gr.JSON(label="Graph Stats")

        with gr.Row():
            query = gr.Textbox(label="Search Query", placeholder="budget expenses")
            limit = gr.Slider(1, 20, value=5, step=1, label="Limit")
            search_btn = gr.Button("Search", variant="primary")
        search_out = gr.Markdown()

        def run_build(dir_val, mode_val):
            resp = client.knowledge_graph_build({"directory": dir_val, "mode": mode_val})
            return (
                f"## Build Complete\n\n{resp.get('summary', '')}",
                resp.get("graph_id", ""),
                resp.get("graph_id", ""),
                resp.get("html_path", ""),
                resp.get("html_path", ""),
                resp.get("stats", {}),
            )

        def run_search(graph_id_val, query_val, limit_val):
            if not graph_id_val:
                return "Build a graph first."
            resp = client.knowledge_graph_search(
                {"graph_id": graph_id_val, "query": query_val, "limit": limit_val}
            )
            results = resp.get("results", [])
            if not results:
                return "No results."
            lines = [f"{i}. {text}" for i, text in enumerate(results, 1)]
            return "## Search Results\n\n" + "\n\n".join(lines)

        build_btn.click(
            run_build,
            inputs=[directory, mode],
            outputs=[summary_out, graph_id_state, graph_id_out, html_path_out, html_file_out, stats_out],
        )
        search_btn.click(run_search, inputs=[graph_id_state, query, limit], outputs=search_out)
