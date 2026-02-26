from __future__ import annotations

import gradio as gr


def build_testing_tab(client) -> None:
    with gr.Tab("Testing"):
        verbose = gr.Checkbox(label="Verbose Output", value=False)
        with gr.Row():
            run_budget_btn = gr.Button("Run Budget Tests", variant="primary")
            run_rlm_btn = gr.Button("Run RLM Tests", variant="primary")
        budget_out = gr.Code(label="Budget Test Output", language="shell", lines=16)
        rlm_out = gr.Code(label="RLM Test Output", language="shell", lines=16)

        def run_budget(v):
            resp = client.testing_budget({"verbose": v})
            return f"$ {resp.get('command', '')}\n(exit={resp.get('return_code')})\n\n{resp.get('output', '')}"

        def run_rlm(v):
            resp = client.testing_rlm({"verbose": v})
            return f"$ {resp.get('command', '')}\n(exit={resp.get('return_code')})\n\n{resp.get('output', '')}"

        run_budget_btn.click(run_budget, inputs=verbose, outputs=budget_out)
        run_rlm_btn.click(run_rlm, inputs=verbose, outputs=rlm_out)

