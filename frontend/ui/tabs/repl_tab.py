from __future__ import annotations

import gradio as gr
from pathlib import Path


def build_repl_tab(client) -> None:
    with gr.Tab("REPL"):
        code = gr.Code(
            label="Python Code",
            language="python",
            value="# Example\nresult = 100 * 25\nprint(f'Result: {result}')",
            lines=10,
        )
        context = gr.Textbox(label="Context", value="REPL execution")
        max_iters_code = gr.Slider(2, 20, value=5, step=1, label="Max Iterations (Code)")
        run_code_btn = gr.Button("Run Code", variant="primary")
        code_out = gr.Textbox(label="Output", lines=10)

        with gr.Row():
            directory = gr.Textbox(label="Directory", value=str(Path.home() / "Downloads"), scale=2)
            task = gr.Textbox(label="Task", placeholder="Find all CSV files and summarize totals", scale=3)
        max_iters_task = gr.Slider(2, 30, value=10, step=1, label="Max Iterations (Task)")
        run_task_btn = gr.Button("Run File Task", variant="primary")
        task_out = gr.Textbox(label="Task Result", lines=10)

        run_code_btn.click(
            lambda c, ctx, it: client.repl_code({"code": c, "context": ctx, "max_iterations": it}).get("result", ""),
            inputs=[code, context, max_iters_code],
            outputs=code_out,
        )
        run_task_btn.click(
            lambda d, t, it: client.repl_task(
                {"directory": d, "task": t, "max_iterations": it}
            ).get("result", ""),
            inputs=[directory, task, max_iters_task],
            outputs=task_out,
        )

