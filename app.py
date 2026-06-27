import os
import shutil
import subprocess

import constants as const

os.environ.setdefault(const.ENV_GRADIO_ANALYTICS, "False")
os.environ.setdefault(const.ENV_HF_OFFLINE, "1")

import gradio as gr

from engine import LlmEngine, Role, Channel, app_dir, find_gguf_models
from documents import extract_document_text


FOLDER = app_dir()
MODELS_ROOT = os.environ.get(const.ENV_MODELS_ROOT, os.path.expanduser("~"))
engine = LlmEngine()
_loaded = {"path": None, "ctx": None, "ngl": None}


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
    if os.name == "nt":
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$d = New-Object System.Windows.Forms.OpenFileDialog;"
            "$d.Filter = '%s';"
            "$d.InitialDirectory = '%s';"
            "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK)"
            " { [Console]::Out.Write($d.FileName) }"
        ) % (const.DIALOG_FILTER_WIN, start_dir.replace("'", "''"))
        cmd = ["powershell", "-NoProfile", "-STA", "-Command", ps]
    elif shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", "--title=" + const.DIALOG_TITLE,
               "--file-filter=" + const.DIALOG_FILTER_ZENITY_GGUF,
               "--file-filter=" + const.DIALOG_FILTER_ZENITY_ALL,
               "--filename=" + os.path.join(start_dir, "")]
    elif shutil.which("kdialog"):
        cmd = ["kdialog", "--getopenfilename", start_dir, const.DIALOG_FILTER_KDIALOG]
    else:
        return None
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=const.DIALOG_TIMEOUT)
    except Exception:
        return None
    path = done.stdout.strip()
    return path or None


def _browse(current):
    start = MODELS_ROOT
    if current:
        parent = os.path.dirname(current.strip())
        if os.path.isdir(parent):
            start = parent
    picked = _native_pick(start)
    return picked or current


def _ensure_loaded(path, ctx, ngl):
    want = {"path": path, "ctx": int(ctx), "ngl": int(ngl)}
    if engine.is_loaded() and _loaded == want:
        return
    engine.load(path, n_ctx=int(ctx), requested_ngl=int(ngl))
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


def respond(message, history, model, system, document,
            temperature, top_p, top_k, repeat_penalty, max_tokens, seed, n_ctx, n_gpu_layers):
    path = _resolve_model(model)
    if not path:
        yield const.MSG_NO_MODEL
        return

    if not (engine.is_loaded() and _loaded == {"path": path, "ctx": int(n_ctx), "ngl": int(n_gpu_layers)}):
        yield const.MSG_LOADING
    try:
        _ensure_loaded(path, n_ctx, n_gpu_layers)
    except Exception as exc:
        yield const.MSG_LOAD_FAILED % exc
        return

    sys_text = _with_document(system.strip(), document, n_ctx, max_tokens)
    messages = [{"role": Role.SYSTEM, "content": sys_text}] if sys_text else []
    messages += history
    messages.append({"role": Role.USER, "content": message})
    messages = [{"role": str(m["role"]), "content": m["content"]} for m in messages]

    params = dict(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repeat_penalty=float(repeat_penalty),
        max_tokens=int(max_tokens),
    )
    if int(seed) >= 0:
        params["seed"] = int(seed)

    reasoning, answer = "", ""
    try:
        for channel, text in engine.stream_chat(messages, **params):
            if channel == Channel.REASONING:
                reasoning += text
            else:
                answer += text
            yield _bubbles(reasoning, answer)
    except Exception as exc:
        yield _bubbles(reasoning, answer) + [gr.ChatMessage(content=const.MSG_GEN_ERROR % exc)]


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
    with gr.Blocks(title=const.APP_TITLE, fill_height=True) as demo:
        gr.Markdown("### " + const.APP_TITLE)
        with gr.Row():
            model = gr.Textbox(
                label=const.LABEL_MODEL, value=(default_model or ""),
                placeholder=const.PLACEHOLDER_MODEL, scale=5,
            )
            browse_btn = gr.Button(const.LABEL_BROWSE, scale=1, min_width=120)
        browse_btn.click(_browse, inputs=[model], outputs=[model])
        system = gr.Textbox(label=const.LABEL_SYSTEM, value=const.DEFAULT_SYSTEM_PROMPT, lines=2)
        document = gr.File(label=const.LABEL_DOCUMENT, type="filepath", file_types=const.DOC_FILE_TYPES)
        with gr.Accordion(const.LABEL_PARAMS, open=False):
            temperature = gr.Slider(0.0, 2.0, value=const.DEFAULT_TEMPERATURE, step=0.05, label=const.LABEL_TEMPERATURE)
            top_p = gr.Slider(0.0, 1.0, value=const.DEFAULT_TOP_P, step=0.01, label=const.LABEL_TOP_P)
            top_k = gr.Number(value=const.DEFAULT_TOP_K, precision=0, label=const.LABEL_TOP_K)
            repeat_penalty = gr.Number(value=const.DEFAULT_REPEAT_PENALTY, label=const.LABEL_REPEAT_PENALTY)
            max_tokens = gr.Number(value=const.DEFAULT_MAX_TOKENS, precision=0, label=const.LABEL_MAX_TOKENS)
            seed = gr.Number(value=const.DEFAULT_SEED, precision=0, label=const.LABEL_SEED)
        with gr.Accordion(const.LABEL_LOAD_SECTION, open=False):
            n_ctx = gr.Number(value=const.DEFAULT_N_CTX, precision=0, label=const.LABEL_N_CTX)
            n_gpu_layers = gr.Number(value=const.DEFAULT_N_GPU_LAYERS, precision=0, label=const.LABEL_N_GPU)

        gr.ChatInterface(
            fn=respond,
            additional_inputs=[model, system, document, temperature, top_p, top_k,
                               repeat_penalty, max_tokens, seed, n_ctx, n_gpu_layers],
            fill_height=True,
        )
    return demo


def main():
    build_ui().launch(inbrowser=True)


if __name__ == "__main__":
    main()
