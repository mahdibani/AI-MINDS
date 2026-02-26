from __future__ import annotations

import gradio as gr
from pathlib import Path


def build_budget_tab(client) -> None:
    with gr.Tab("Budget"):
        with gr.Row():
            budget_file = gr.Textbox(label="Budget File", value=str(Path.home() / "Downloads" / "budget.xlsx"))
            safety = gr.Slider(0, 50, value=20, step=5, label="Safety Buffer %")
            user_id = gr.Textbox(label="User ID", value="gradio_user")

        with gr.Row():
            item = gr.Textbox(label="Item")
            price = gr.Number(label="Price", value=250)
            check_btn = gr.Button("Check Affordability", variant="primary")

        result_out = gr.JSON(label="Result")

        summary_btn = gr.Button("Budget Summary")
        summary_out = gr.Markdown()

        def run_check(budget_file_val, safety_val, user_val, item_val, price_val):
            return client.budget_check(
                {
                    "item": item_val,
                    "price": price_val,
                    "budget_file": budget_file_val,
                    "safety_buffer_percent": safety_val,
                    "user_id": user_val,
                    "explain": True,
                }
            )

        def run_summary(budget_file_val, safety_val, user_val):
            resp = client.budget_summary(
                {
                    "budget_file": budget_file_val,
                    "safety_buffer_percent": safety_val,
                    "user_id": user_val,
                }
            )
            return resp.get("summary", "")

        check_btn.click(
            run_check,
            inputs=[budget_file, safety, user_id, item, price],
            outputs=result_out,
        )
        summary_btn.click(
            run_summary,
            inputs=[budget_file, safety, user_id],
            outputs=summary_out,
        )

