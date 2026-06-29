import os

import constants as const

_dll_directory = None
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    _app_directory = os.path.dirname(os.path.abspath(__file__))
    _dll_directory = os.add_dll_directory(_app_directory)

os.environ.setdefault(const.ENV_GRADIO_ANALYTICS, "False")
os.environ.setdefault(const.ENV_HF_OFFLINE, "1")

import gradio as gr

from engine import LlmEngine, Channel, app_dir, find_gguf_models
from documents import extract_document_text


FOLDER = app_dir()
MODELS_ROOT = os.environ.get(const.ENV_MODELS_ROOT, os.path.expanduser("~"))
engine = LlmEngine()
_loaded = {"path": None, "ctx": None}

UI_CSS = """
#main-layout { align-items: stretch; }
#chat-panel { order: 1; min-width: 0; }
#settings-panel {
    order: 2;
    min-width: 320px;
    max-width: 400px;
}
@media (max-width: 900px) {
    #settings-panel {
        min-width: 100%;
        max-width: none;
    }
}
"""


def _gradio_major():
    try:
        return int(gr.__version__.split(".", 1)[0])
    except (AttributeError, ValueError):
        return 0


def _resolve_model(model):
    if isinstance(model, (list, tuple)):
        model = model[0] if model else None
    if not model:
        return None
    model = model.strip()
    for cand in (model, os.path.join(MODELS_ROOT, model), os.path.join(FOLDER, model)):
        if cand.lower().endswith(const.GGUF_EXT) and os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def _native_pick(start_dir):
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            parent=root,
            title=const.DIALOG_TITLE,
            initialdir=start_dir,
            filetypes=(("GGUF", "*.gguf"), ("Все файлы", "*.*")),
        )
    except Exception:
        return None
    finally:
        if root is not None:
            root.destroy()
    return str(path) if path else None


def _browse(current):
    start = MODELS_ROOT
    if current:
        parent = os.path.dirname(current.strip())
        if os.path.isdir(parent):
            start = parent
    picked = _native_pick(start)
    return picked or current


def _ensure_loaded(path, ctx):
    want = {"path": path, "ctx": int(ctx)}
    if engine.is_loaded() and _loaded == want:
        return
    engine.load(path, n_ctx=int(ctx))
    _loaded.update(want)


def _with_document(system, document, ctx, max_tokens):
    if not document:
        return system
    try:
        text = extract_document_text(document)
    except Exception as exc:
        return (system + "\n\n" + const.MSG_DOC_ERROR % exc) if system else system
    if not text.strip():
        return system
    reserve = (int(max_tokens) + const.DOC_RESERVE_TOKENS) * const.DOC_CHARS_PER_TOKEN
    budget = max(const.DOC_MIN_BUDGET, int(ctx) * const.DOC_CHARS_PER_TOKEN - reserve)
    block = const.DOC_CONTEXT_HEADER + text[:budget]
    return (system + "\n\n" + block) if system else block


def _context_usage(usage, limit):
    limit = int(limit)
    if not usage:
        return const.CONTEXT_USAGE_UNKNOWN % limit
    used = int(usage.get("total_tokens") or (
        int(usage.get("prompt_tokens") or 0)
        + int(usage.get("completion_tokens") or 0)
    ))
    percent = used * 100.0 / limit if limit > 0 else 0.0
    return const.CONTEXT_USAGE_TEMPLATE % (used, limit, percent)


def respond(message, history, model, system, document,
            temperature, top_p, top_k, repeat_penalty, max_tokens, n_ctx):
    usage_text = _context_usage(None, n_ctx)
    path = _resolve_model(model)
    if not path:
        yield const.MSG_NO_MODEL, usage_text
        return

    if not (engine.is_loaded() and _loaded == {"path": path, "ctx": int(n_ctx)}):
        yield const.MSG_LOADING, usage_text
    try:
        _ensure_loaded(path, n_ctx)
    except Exception as exc:
        yield const.MSG_LOAD_FAILED % exc, usage_text
        return

    sys_text = _with_document(system.strip(), document, n_ctx, max_tokens)
    messages = [{"role": "system", "content": sys_text}] if sys_text else []
    messages += history
    messages.append({"role": "user", "content": message})
    messages = [{"role": str(m["role"]), "content": m["content"]} for m in messages]

    params = dict(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repeat_penalty=float(repeat_penalty),
        max_tokens=int(max_tokens),
    )
    reasoning, answer = "", ""
    try:
        for channel, text in engine.stream_chat(messages, **params):
            if channel == Channel.REASONING:
                reasoning += text
            else:
                answer += text
            yield _bubbles(reasoning, answer), usage_text
        yield _bubbles(reasoning, answer), _context_usage(engine.last_usage, n_ctx)
    except Exception as exc:
        yield (
            _bubbles(reasoning, answer) + [gr.ChatMessage(content=const.MSG_GEN_ERROR % exc)],
            usage_text,
        )


def _bubbles(reasoning, answer):
    out = []
    if reasoning:
        done = bool(answer)
        out.append(gr.ChatMessage(
            content=reasoning,
            metadata={"title": const.THINKING_DONE if done else const.THINKING_PROGRESS,
                      "status": "done" if done else "pending"},
        ))
    if answer or not out:
        out.append(gr.ChatMessage(content=answer))
    return out


def build_ui():
    local = find_gguf_models(FOLDER)
    default_model = os.path.join(FOLDER, local[0]) if local else None
    blocks_args = {"css": UI_CSS} if _gradio_major() < 6 else {}
    with gr.Blocks(title=const.APP_TITLE, fill_height=True, **blocks_args) as demo:
        gr.Markdown("### " + const.APP_TITLE)
        with gr.Row(elem_id="main-layout"):
            with gr.Column(scale=1, elem_id="settings-panel"):
                with gr.Row():
                    model = gr.Textbox(
                        label=const.LABEL_MODEL, value=(default_model or ""),
                        placeholder=const.PLACEHOLDER_MODEL, scale=5,
                    )
                    browse_btn = gr.Button(const.LABEL_BROWSE, scale=1, min_width=100)
                system = gr.Textbox(
                    label=const.LABEL_SYSTEM,
                    value=const.DEFAULT_SYSTEM_PROMPT,
                    lines=3,
                )
                document = gr.File(
                    label=const.LABEL_DOCUMENT,
                    type="filepath",
                    file_types=const.DOC_FILE_TYPES,
                )
                with gr.Accordion(const.LABEL_PARAMS, open=True):
                    temperature = gr.Slider(0.0, 2.0, value=const.DEFAULT_TEMPERATURE, step=0.05, label=const.LABEL_TEMPERATURE)
                    top_p = gr.Slider(0.0, 1.0, value=const.DEFAULT_TOP_P, step=0.01, label=const.LABEL_TOP_P)
                    top_k = gr.Number(value=const.DEFAULT_TOP_K, precision=0, label=const.LABEL_TOP_K)
                    repeat_penalty = gr.Number(value=const.DEFAULT_REPEAT_PENALTY, label=const.LABEL_REPEAT_PENALTY)
                    max_tokens = gr.Number(value=const.DEFAULT_MAX_TOKENS, precision=0, label=const.LABEL_MAX_TOKENS)
                    n_ctx = gr.Number(value=const.DEFAULT_N_CTX, precision=0, label=const.LABEL_N_CTX)
                context_usage = gr.Markdown(_context_usage(None, const.DEFAULT_N_CTX))

            with gr.Column(scale=4, elem_id="chat-panel"):
                gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[model, system, document, temperature, top_p, top_k,
                                       repeat_penalty, max_tokens, n_ctx],
                    additional_outputs=[context_usage],
                    fill_height=True,
                )

        browse_btn.click(_browse, inputs=[model], outputs=[model])
    return demo


def main():
    launch_args = {"css": UI_CSS} if _gradio_major() >= 6 else {}
    build_ui().launch(inbrowser=True, **launch_args)


if __name__ == "__main__":
    main()
