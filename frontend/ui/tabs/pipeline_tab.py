from __future__ import annotations

import gradio as gr


def build_pipeline_tab(client) -> None:
    with gr.Tab("Pipeline"):
        prompt = gr.Textbox(label="Large Prompt", lines=10)
        task = gr.Textbox(label="Task", value="Extract action items and dates")
        with gr.Row():
            chunk_size = gr.Slider(10000, 200000, value=80000, step=5000, label="Chunk Size")
            max_workers = gr.Slider(1, 12, value=4, step=1, label="Workers")
        run_btn = gr.Button("Run Pipeline", variant="primary")
        output = gr.Markdown()
        metrics = gr.JSON(label="Metrics")

        def run_pipeline(prompt_val, task_val, chunk_val, worker_val):
            resp = client.pipeline_run(
                {
                    "prompt": prompt_val,
                    "task": task_val,
                    "chunk_size": chunk_val,
                    "max_workers": worker_val,
                    "parallel": True,
                }
            )
            formatted = f"## Output\n\n{resp.get('final_output', '')}"
            stats = {
                "chunks_processed": resp.get("chunks_processed"),
                "total_tokens": resp.get("total_tokens"),
                "total_cost": resp.get("total_cost"),
                "elapsed_total": resp.get("elapsed_total"),
            }
            return formatted, stats

        run_btn.click(run_pipeline, inputs=[prompt, task, chunk_size, max_workers], outputs=[output, metrics])

