from __future__ import annotations

import json
import gradio as gr


def build_memory_tab(client) -> None:
    with gr.Tab("Memory"):
        user_id = gr.Textbox(label="User ID", value="gradio_user")

        with gr.Row():
            mem_text = gr.Textbox(label="Memory", lines=3)
            metadata = gr.Textbox(label="Metadata JSON", value='{"source":"ui"}')
        add_btn = gr.Button("Add Memory", variant="primary")
        add_out = gr.JSON(label="Add Result")

        with gr.Row():
            query = gr.Textbox(label="Search Query")
            limit = gr.Slider(1, 20, value=5, step=1, label="Limit")
            search_btn = gr.Button("Search")
        search_out = gr.JSON(label="Search Results")

        with gr.Row():
            list_btn = gr.Button("List All")
            clear_btn = gr.Button("Clear", variant="stop")
        list_out = gr.JSON(label="Memories")

        def add_memory(uid, text, meta):
            parsed = json.loads(meta) if meta.strip() else None
            return client.memory_add({"user_id": uid, "text": text, "metadata": parsed})

        add_btn.click(add_memory, inputs=[user_id, mem_text, metadata], outputs=add_out)
        search_btn.click(
            lambda uid, q, l: client.memory_search({"user_id": uid, "query": q, "limit": l}),
            inputs=[user_id, query, limit],
            outputs=search_out,
        )
        list_btn.click(lambda uid: client.memory_list({"user_id": uid}), inputs=user_id, outputs=list_out)
        clear_btn.click(lambda uid: client.memory_clear({"user_id": uid}), inputs=user_id, outputs=list_out)

