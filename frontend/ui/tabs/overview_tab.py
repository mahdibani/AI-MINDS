from __future__ import annotations

import gradio as gr


def build_overview_tab(client) -> None:
    with gr.Tab("Overview"):
        gr.Markdown("## AI-MINDS Platform")
        gr.Markdown(
            "Backend now runs as API microservices (budget, files, memory, pipeline) and this UI consumes those APIs."
        )
        status_out = gr.JSON(label="Gateway Health")
        refresh_btn = gr.Button("Check API Health", variant="primary")
        refresh_btn.click(lambda: client.health(), outputs=status_out)

