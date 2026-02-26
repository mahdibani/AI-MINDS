from __future__ import annotations

import gradio as gr
from pathlib import Path


def build_files_tab(client) -> None:
    with gr.Tab("Files"):
        with gr.Row():
            directory = gr.Textbox(label="Directory", value=str(Path.home() / "Downloads"))
            list_btn = gr.Button("List Files", variant="primary")
        files_out = gr.JSON(label="Directory Listing")

        with gr.Row():
            filepath = gr.Textbox(label="File Path")
            max_chars = gr.Slider(1000, 100000, value=50000, step=1000, label="Max Chars")
            parse_btn = gr.Button("Parse File", variant="primary")
        parse_out = gr.JSON(label="Parsed")

        analysis_query = gr.Textbox(label="Analysis Query", placeholder="Summarize risks and opportunities")
        analyze_btn = gr.Button("Analyze Directory", variant="primary")
        analysis_out = gr.Markdown()

        list_btn.click(lambda d: client.files_list({"directory": d}), inputs=directory, outputs=files_out)
        parse_btn.click(
            lambda fp, mc: client.files_parse({"filepath": fp, "max_chars": mc}),
            inputs=[filepath, max_chars],
            outputs=parse_out,
        )
        analyze_btn.click(
            lambda d, q: client.files_analyze({"directory": d, "query": q}).get("result", ""),
            inputs=[directory, analysis_query],
            outputs=analysis_out,
        )

