from __future__ import annotations

import gradio as gr

from api_client import ApiClient
from ui.theme import APP_CSS
from ui.tabs.overview_tab import build_overview_tab
from ui.tabs.budget_tab import build_budget_tab
from ui.tabs.files_tab import build_files_tab
from ui.tabs.memory_tab import build_memory_tab
from ui.tabs.pipeline_tab import build_pipeline_tab
from ui.tabs.knowledge_graph_tab import build_knowledge_graph_tab
from ui.tabs.repl_tab import build_repl_tab
from ui.tabs.testing_tab import build_testing_tab


def create_app() -> gr.Blocks:
    client = ApiClient()

    with gr.Blocks(title="AI-MINDS", css=APP_CSS) as app:
        gr.Markdown("# AI-MINDS Frontend")
        gr.Markdown("Microservices-driven UI over the AI-MINDS API gateway.")

        with gr.Tabs():
            build_overview_tab(client)
            build_budget_tab(client)
            build_files_tab(client)
            build_memory_tab(client)
            build_pipeline_tab(client)
            build_knowledge_graph_tab(client)
            build_repl_tab(client)
            build_testing_tab(client)

    return app


if __name__ == "__main__":
    ui = create_app()
    ui.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)
