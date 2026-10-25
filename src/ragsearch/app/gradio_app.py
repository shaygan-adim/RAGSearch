"""Gradio UI: an image-to-image tab (gallery) and a text-to-text tab (cards), both paginated."""

from __future__ import annotations

import html
import math

import gradio as gr

from ragsearch.app.engine import SearchEngine, SearchHit


def total_pages(n_results: int, page_size: int) -> int:
    return max(1, math.ceil(n_results / page_size))


def page_slice(results: list[SearchHit], page: int, page_size: int) -> list[SearchHit]:
    start = (page - 1) * page_size
    return results[start : start + page_size]


def render_text_cards(hits: list[SearchHit]) -> str:
    """HTML cards for the text tab. Product text comes from the data, so it is escaped."""
    if not hits:
        return "<p>No results found.</p>"
    cards = []
    for hit in hits:
        cards.append(
            "<div style='border:1px solid #ddd; padding:10px; border-radius:8px; background:#f9f9f9;'>"
            f"<h3 style='margin:0; color:#333;'>{html.escape(hit.persian_name)}</h3>"
            f"<p style='margin:5px 0;'><b>Category:</b> {html.escape(hit.category_name)} | "
            f"<b>Brand:</b> {html.escape(hit.brand_name)}</p>"
            f"<p style='font-size:0.8em; color:#666;'>Similarity Score: {hit.score:.4f}</p>"
            "</div>"
        )
    return "<div style='display:flex; flex-direction:column; gap:15px;'>" + "".join(cards) + "</div>"


def render_gallery(hits: list[SearchHit]) -> list[tuple[str, str]]:
    return [(h.image_path, f"{h.persian_name}\n{h.category_name} | {h.brand_name}") for h in hits if h.image_path]


def build_demo(engine: SearchEngine, page_size: int) -> gr.Blocks:
    def search_handler(search_fn, render):
        def handler(query):
            results = search_fn(query)
            pages = total_pages(len(results), page_size)
            return (
                results, render(page_slice(results, 1, page_size)), 1, pages, f"Page 1 of {pages}",
                gr.update(interactive=False), gr.update(interactive=pages > 1),
            )
        return handler

    def page_handler(direction, render):
        def handler(results, page, pages):
            if not results:
                return gr.update(), page, gr.update(), gr.update(), gr.update()
            page = min(pages, page + 1) if direction == "next" else max(1, page - 1)
            return (
                render(page_slice(results, page, page_size)), page, f"Page {page} of {pages}",
                gr.update(interactive=page > 1), gr.update(interactive=page < pages),
            )
        return handler

    def search_tab(label, make_input, button_label, make_output, search_fn, render):
        """One tab: input + search button on the left, results + pager on the right."""
        with gr.TabItem(label):
            with gr.Row():
                with gr.Column(scale=1):
                    query = make_input()
                    search_btn = gr.Button(button_label, variant="primary")
                with gr.Column(scale=2):
                    results_state, page_state, pages_state = gr.State([]), gr.State(1), gr.State(1)
                    output = make_output()
                    with gr.Row():
                        prev_btn = gr.Button("Previous", interactive=False)
                        page_info = gr.Label("Page 1 of 1", show_label=False)
                        next_btn = gr.Button("Next", interactive=False)

            pager_outputs = [output, page_state, page_info, prev_btn, next_btn]
            search_btn.click(
                search_handler(search_fn, render), inputs=[query],
                outputs=[results_state, output, page_state, pages_state, page_info, prev_btn, next_btn],
            )
            for button, direction in ((prev_btn, "prev"), (next_btn, "next")):
                button.click(page_handler(direction, render), inputs=[results_state, page_state, pages_state],
                             outputs=pager_outputs)

    with gr.Blocks(title="AI Product Retrieval") as demo:
        gr.Markdown("# 🛍️ AI Vector Product Retrieval System")
        gr.Markdown("Dual-Encoder Architecture (DistilBERT + SigLIP)")
        with gr.Tabs():
            search_tab("🖼️ Image-to-Image Search", lambda: gr.Image(type="pil", label="Upload Query Image"),
                       "Search Similar Products",
                       lambda: gr.Gallery(label="Results", columns=2, height=600, object_fit="contain"),
                       engine.search_image, render_gallery)
            search_tab("📝 Text-to-Text Search",
                       lambda: gr.Textbox(label="Enter Query", placeholder="e.g., فرش ماشینی آبی"),
                       "Search Products", gr.HTML, engine.search_text, render_text_cards)
    return demo
