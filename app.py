#!/usr/bin/env python3
import os
import sys
import enum
import json
import time
import queue
import socket
import threading
import traceback
import subprocess
import urllib.error
import urllib.request

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox


APP_TITLE = "GGUF Чат"
SHIFT_MASK = 0x0001
CREATE_NO_WINDOW = 0x08000000
SERVER_HOST = "127.0.0.1"


def _console_attached() -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:
        return False


DEBUG = bool(os.environ.get("GGUF_DEBUG")) or _console_attached()

LOAD_TIMEOUT = 300.0
GEN_TIMEOUT = 600.0


def dbg(msg):
    if not DEBUG:
        return
    try:
        print("[gguf] " + str(msg), file=sys.stderr, flush=True)
    except Exception:
        pass


class Msg(enum.Enum):
    TOKEN = enum.auto()
    DONE = enum.auto()
    STOPPED = enum.auto()
    LOG = enum.auto()
    LOADED = enum.auto()
    ERROR = enum.auto()
    FATAL = enum.auto()


class Tag(enum.StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    ERROR = "error"
    MUTED = "muted"


class Role(enum.StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


def app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def bundle_dir() -> str:
    return getattr(sys, "_MEIPASS", None) or app_dir()


def find_gguf_models(folder: str):
    try:
        names = [
            n for n in os.listdir(folder)
            if n.lower().endswith(".gguf") and os.path.isfile(os.path.join(folder, n))
        ]
    except OSError:
        names = []
    return sorted(names)


def _read_text_file(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _extract_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_docx(path):
    import docx
    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs)


def extract_document_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    return _read_text_file(path)


class LlmEngine:
    def __init__(self):
        self.proc = None
        self.port = None
        self.status = "Модель не загружена."
        self._loaded = False
        self._stderr = []
        self._cancel = threading.Event()

    def _server_exe(self):
        base = bundle_dir()
        for name in ("llama-server.exe", "llama-server", "server.exe", "server"):
            path = os.path.join(base, "llama_server", name)
            if os.path.isfile(path):
                return path
        env = os.environ.get("LLAMA_SERVER")
        if env and os.path.isfile(env):
            return env
        return None

    @staticmethod
    def _free_port():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((SERVER_HOST, 0))
            return s.getsockname()[1]
        finally:
            s.close()

    def _base_url(self):
        return "http://%s:%d" % (SERVER_HOST, self.port)

    def _drain_stderr(self, proc):
        try:
            for line in proc.stderr:
                line = line.rstrip("\n")
                self._stderr.append(line)
                if len(self._stderr) > 400:
                    del self._stderr[:200]
                dbg("server: " + line)
        except Exception:
            pass

    def _stderr_tail(self, n=8):
        return "\n".join(self._stderr[-n:])

    def _health_ok(self):
        try:
            with urllib.request.urlopen(self._base_url() + "/health", timeout=2) as r:
                body = r.read().decode("utf-8", "replace")
            return r.status == 200 and '"ok"' in body
        except Exception:
            return False

    def is_loaded(self) -> bool:
        return self._loaded and self.proc is not None and self.proc.poll() is None

    def shutdown(self):
        proc = self.proc
        self.proc = None
        self._loaded = False
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _start(self, exe, model_path, n_ctx, ngl, log):
        self.port = self._free_port()
        self._stderr = []
        argv = [
            exe,
            "-m", model_path,
            "-c", str(int(n_ctx)),
            "-ngl", str(int(ngl)),
            "--host", SERVER_HOST,
            "--port", str(self.port),
        ]
        win = sys.platform.startswith("win")
        flags = CREATE_NO_WINDOW if (win and not DEBUG) else 0
        dbg("server spawn: %s" % " ".join(argv))
        self.proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=flags,
        )
        threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True).start()

        deadline = time.monotonic() + LOAD_TIMEOUT
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return False, self._stderr_tail()
            if self._health_ok():
                return True, ""
            time.sleep(0.4)
        return False, "таймаут загрузки"

    def _status_from_log(self, ngl):
        text = "\n".join(self._stderr).lower()
        gpu = "ggml_cuda_init" in text or "using device cuda" in text or "cuda0" in text
        if ngl == 0:
            return "Работает на CPU"
        if gpu:
            return "Загружено на видеокарту (CUDA)"
        return "Загружено (CPU — видеокарта не задействована)"

    def load(self, model_path: str, n_ctx: int, requested_ngl: int, on_log=None):
        def log(msg):
            if on_log:
                on_log(msg)

        self.shutdown()
        exe = self._server_exe()
        if not exe:
            raise RuntimeError("Не найден llama-server (папка llama_server рядом с программой).")

        req = 999 if int(requested_ngl) >= 900 else int(requested_ngl)
        attempts = [req] if req == 0 else [req, 0]
        last = ""
        for ngl in attempts:
            log("Запускаю движок (слоёв на GPU: %s) …" % ngl)
            ok, err = self._start(exe, model_path, n_ctx, ngl, log)
            if ok:
                self._loaded = True
                self.status = self._status_from_log(ngl)
                log(self.status)
                return self.status
            last = err
            self.shutdown()
            if ngl != attempts[-1]:
                log("Не вышло, пробую запасной вариант (CPU) …")

        raise RuntimeError("Не удалось загрузить модель.\n\n" + (last or "неизвестная ошибка"))

    def cancel(self):
        self._cancel.set()

    def stream_chat(self, messages, **params):
        if not self.is_loaded():
            raise RuntimeError("Модель не загружена.")
        self._cancel.clear()

        payload = {"messages": messages, "stream": True}
        for key in ("temperature", "top_p", "top_k", "max_tokens", "seed", "repeat_penalty"):
            if params.get(key) is not None:
                payload[key] = params[key]

        req = urllib.request.Request(
            self._base_url() + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=GEN_TIMEOUT)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError("ошибка движка: " + detail[:300])
        except Exception as exc:
            raise RuntimeError("движок недоступен: " + str(exc))

        try:
            for raw in resp:
                if self._cancel.is_set():
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield delta
        finally:
            try:
                resp.close()
            except Exception:
                pass


class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x740")
        self.minsize(740, 540)

        self.engine = LlmEngine()
        self.folder = app_dir()
        self.history = []
        self.doc_name = None
        self.doc_text = ""

        self.token_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.busy = False
        self.generating = False
        self._reply_acc = []

        self._build_widgets()
        self._refresh_model_list()
        self.after(40, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_widgets(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(root)
        top.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(top, text="Модель:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, state="readonly", width=46)
        self.model_combo.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Обновить", command=self._refresh_model_list).pack(side=tk.LEFT)
        ttk.Button(top, text="Обзор…", command=self._browse_model).pack(side=tk.LEFT, padx=(6, 0))
        self.load_btn = ttk.Button(top, text="Загрузить модель", command=self._load_model)
        self.load_btn.pack(side=tk.LEFT, padx=6)

        body = ttk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True)

        chat_col = ttk.Frame(body)
        chat_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.transcript = scrolledtext.ScrolledText(
            chat_col, wrap=tk.WORD, state=tk.DISABLED, font=("TkDefaultFont", 11)
        )
        self.transcript.pack(fill=tk.BOTH, expand=True)
        self.transcript.tag_configure(Tag.USER, foreground="#1565c0", font=("TkDefaultFont", 11, "bold"))
        self.transcript.tag_configure(Tag.ASSISTANT, foreground="#2e7d32", font=("TkDefaultFont", 11, "bold"))
        self.transcript.tag_configure(Tag.ERROR, foreground="#c62828")
        self.transcript.tag_configure(Tag.MUTED, foreground="#888888")

        attach_row = ttk.Frame(chat_col)
        attach_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(attach_row, text="Документ…", command=self._attach_doc).pack(side=tk.LEFT)
        self.clear_doc_btn = ttk.Button(attach_row, text="Убрать", command=self._clear_doc, state=tk.DISABLED)
        self.clear_doc_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.doc_label = ttk.Label(attach_row, text="документ не прикреплён", foreground="#888888")
        self.doc_label.pack(side=tk.LEFT, padx=(8, 0))

        in_row = ttk.Frame(chat_col)
        in_row.pack(fill=tk.X, pady=(6, 0))
        self.input_box = tk.Text(in_row, height=3, wrap=tk.WORD, font=("TkDefaultFont", 11))
        self.input_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input_box.bind("<Return>", self._on_return)

        btn_col = ttk.Frame(in_row)
        btn_col.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0))
        self.send_btn = ttk.Button(btn_col, text="Отправить", command=self._on_send)
        self.send_btn.pack(fill=tk.X)
        self.stop_btn = ttk.Button(btn_col, text="Стоп", command=self._on_stop, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btn_col, text="Очистить", command=self._clear_chat).pack(fill=tk.X, pady=(4, 0))

        self._build_params(body)

        self.status_var = tk.StringVar(value="Модель не загружена.")
        ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, pady=(6, 0))

    def _build_params(self, parent):
        panel = ttk.LabelFrame(parent, text="Параметры", padding=8)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))

        self.p_temp = tk.DoubleVar(value=0.7)
        self.p_top_p = tk.DoubleVar(value=0.95)
        self.p_top_k = tk.IntVar(value=40)
        self.p_repeat = tk.DoubleVar(value=1.1)
        self.p_seed = tk.IntVar(value=-1)
        self.p_ctx = tk.IntVar(value=4096)
        self.p_max = tk.IntVar(value=512)
        self.p_gpu = tk.IntVar(value=999)

        self._slider(panel, "Температура", self.p_temp, 0.0, 2.0)
        self._slider(panel, "Top-p", self.p_top_p, 0.0, 1.0)
        self._int_field(panel, "Top-k", self.p_top_k)
        self._float_field(panel, "Штраф за повтор", self.p_repeat)
        self._int_field(panel, "Сид (-1 = случайный)", self.p_seed)
        self._int_field(panel, "Макс. токенов ответа", self.p_max)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(panel, text="Применяется при загрузке модели:", foreground="#888888").pack(anchor=tk.W)
        self._int_field(panel, "Длина контекста (n_ctx)", self.p_ctx)
        self._int_field(panel, "Слоёв на GPU (999=все, 0=CPU)", self.p_gpu)

        ttk.Label(panel, text="Системный промпт:").pack(anchor=tk.W, pady=(8, 0))
        self.system_box = tk.Text(panel, height=7, width=30, wrap=tk.WORD)
        self.system_box.pack(fill=tk.X)
        self.system_box.insert("1.0", "Ты — полезный ассистент.")

    def _slider(self, parent, label, var, lo, hi):
        head = ttk.Frame(parent)
        head.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(head, text=label).pack(side=tk.LEFT)
        val = ttk.Label(head, text=f"{var.get():.2f}")
        val.pack(side=tk.RIGHT)
        ttk.Scale(
            parent, from_=lo, to=hi, variable=var,
            command=lambda _v, l=val, v=var: l.config(text=f"{float(v.get()):.2f}"),
        ).pack(fill=tk.X)

    def _int_field(self, parent, label, var):
        ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(6, 0))
        ttk.Entry(parent, textvariable=var, width=12).pack(anchor=tk.W)

    def _float_field(self, parent, label, var):
        self._int_field(parent, label, var)

    def _refresh_model_list(self):
        models = find_gguf_models(self.folder)
        self.model_combo["values"] = models
        if models and not self.model_var.get():
            self.model_var.set(models[0])
        if not models:
            self.status_var.set(f"Файлы .gguf не найдены в: {self.folder}")

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Выберите модель GGUF",
            initialdir=self.folder,
            filetypes=[("Модели GGUF", "*.gguf"), ("Все файлы", "*.*")],
        )
        if path:
            self.folder = os.path.dirname(path)
            name = os.path.basename(path)
            vals = list(self.model_combo["values"])
            if name not in vals:
                vals.insert(0, name)
                self.model_combo["values"] = vals
            self.model_var.set(name)

    def _attach_doc(self):
        path = filedialog.askopenfilename(
            title="Выберите документ",
            initialdir=self.folder,
            filetypes=[("Документы", "*.pdf *.docx *.txt *.md"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            text = extract_document_text(path)
        except ImportError as exc:
            messagebox.showerror(APP_TITLE, f"Нет модуля для этого формата: {exc}")
            return
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Не удалось прочитать документ:\n{exc}")
            return
        if not text.strip():
            messagebox.showwarning(APP_TITLE, "В документе не найден текст (возможно, скан без OCR).")
            return
        self.doc_name = os.path.basename(path)
        self.doc_text = text
        self.doc_label.config(text=f"📎 {self.doc_name} ({len(text)} симв.)", foreground="#2e7d32")
        self.clear_doc_btn.config(state=tk.NORMAL)

    def _clear_doc(self):
        self.doc_name = None
        self.doc_text = ""
        self.doc_label.config(text="документ не прикреплён", foreground="#888888")
        self.clear_doc_btn.config(state=tk.DISABLED)

    def _with_document(self, system):
        if not self.doc_text:
            return system
        reserve_chars = (int(self.p_max.get()) + 600) * 4
        budget = max(2000, int(self.p_ctx.get()) * 4 - reserve_chars)
        doc = self.doc_text[:budget]
        block = f"Контекст из документа «{self.doc_name}»:\n{doc}"
        return (system + "\n\n" + block) if system else block

    def _load_model(self):
        if self.busy:
            return
        name = self.model_var.get()
        if not name:
            messagebox.showwarning(APP_TITLE, "Сначала выберите модель .gguf.")
            return
        path = name if os.path.isabs(name) else os.path.join(self.folder, name)
        if not os.path.isfile(path):
            messagebox.showerror(APP_TITLE, f"Файл не найден:\n{path}")
            return

        self._set_busy(True)
        self.load_btn.config(state=tk.DISABLED)
        self.status_var.set(f"Загрузка {os.path.basename(path)} …")

        def work():
            try:
                self.engine.load(
                    path,
                    n_ctx=self.p_ctx.get(),
                    requested_ngl=self.p_gpu.get(),
                    on_log=lambda m: self.token_queue.put((Msg.LOG, m)),
                )
                self.token_queue.put((Msg.LOADED, self.engine.status))
            except Exception:
                self.token_queue.put((Msg.FATAL, traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _on_return(self, event):
        if event.state & SHIFT_MASK:
            return None
        self._on_send()
        return "break"

    def _on_send(self):
        if self.busy:
            return
        if not self.engine.is_loaded():
            messagebox.showinfo(APP_TITLE, "Сначала загрузите модель.")
            return
        text = self.input_box.get("1.0", tk.END).strip()
        if not text:
            return
        self.input_box.delete("1.0", tk.END)

        self.history.append({"role": Role.USER, "content": text})
        self._append("\nВы:\n", Tag.USER)
        self._append(text + "\n")
        if self.doc_text:
            self._append(f"📎 в контексте: {self.doc_name}\n", Tag.MUTED)
        self._append("\nАссистент:\n", Tag.ASSISTANT)

        system = self._with_document(self.system_box.get("1.0", tk.END).strip())
        turns = ([{"role": Role.SYSTEM, "content": system}] if system else []) + list(self.history)
        messages = [{"role": str(t["role"]), "content": t["content"]} for t in turns]

        params = dict(
            temperature=float(self.p_temp.get()),
            top_p=float(self.p_top_p.get()),
            top_k=int(self.p_top_k.get()),
            repeat_penalty=float(self.p_repeat.get()),
            max_tokens=int(self.p_max.get()),
        )
        seed = int(self.p_seed.get())
        if seed >= 0:
            params["seed"] = seed

        self.stop_event.clear()
        self.generating = True
        self._set_busy(True)
        self.status_var.set("Генерация …")
        self._reply_acc = []

        def work():
            try:
                for piece in self.engine.stream_chat(messages, **params):
                    self.token_queue.put((Msg.TOKEN, piece))
                self.token_queue.put((Msg.STOPPED if self.stop_event.is_set() else Msg.DONE, None))
            except Exception:
                self.token_queue.put((Msg.ERROR, traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _on_stop(self):
        self.stop_event.set()
        self.engine.cancel()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.token_queue.get_nowait()
                if kind is Msg.TOKEN:
                    self._reply_acc.append(payload)
                    self._append(payload)
                elif kind is Msg.DONE:
                    self._finish_reply()
                elif kind is Msg.STOPPED:
                    self._append("\n[остановлено]\n", Tag.MUTED)
                    self._finish_reply()
                elif kind is Msg.LOG:
                    self.status_var.set(payload)
                elif kind is Msg.LOADED:
                    self._set_busy(False)
                    self.load_btn.config(state=tk.NORMAL)
                    self.status_var.set(payload)
                elif kind is Msg.ERROR:
                    self._append("\n" + self._friendly_error(payload) + "\n", Tag.ERROR)
                    self._finish_reply()
                elif kind is Msg.FATAL:
                    self._set_busy(False)
                    self.load_btn.config(state=tk.NORMAL)
                    self.status_var.set("Не удалось загрузить модель.")
                    messagebox.showerror(APP_TITLE, payload)
        except queue.Empty:
            pass
        self.after(40, self._poll_queue)

    def _friendly_error(self, tb_text: str) -> str:
        low = tb_text.lower()
        if "context" in low and ("exceed" in low or "window" in low or "n_ctx" in low):
            return (
                "[контекст переполнен] Диалог длиннее окна контекста (n_ctx). "
                "Нажмите «Очистить», либо увеличьте n_ctx и перезагрузите модель."
            )
        return "[ошибка]\n" + tb_text

    def _finish_reply(self):
        reply = "".join(self._reply_acc)
        if reply:
            self.history.append({"role": Role.ASSISTANT, "content": reply})
        self._append("\n")
        self.generating = False
        self._set_busy(False)
        self.status_var.set(self.engine.status if self.engine.is_loaded() else "Готово.")

    def _append(self, text, tag=None):
        self.transcript.config(state=tk.NORMAL)
        if tag is not None:
            self.transcript.insert(tk.END, text, tag)
        else:
            self.transcript.insert(tk.END, text)
        self.transcript.see(tk.END)
        self.transcript.config(state=tk.DISABLED)

    def _clear_chat(self):
        if self.busy:
            return
        self.history.clear()
        self.transcript.config(state=tk.NORMAL)
        self.transcript.delete("1.0", tk.END)
        self.transcript.config(state=tk.DISABLED)

    def _set_busy(self, busy):
        self.busy = busy
        self.send_btn.config(state=tk.DISABLED if busy else tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL if (busy and self.generating) else tk.DISABLED)

    def _on_close(self):
        self.stop_event.set()
        try:
            self.engine.shutdown()
        except Exception:
            pass
        self.destroy()


def main():
    dbg("GUI start; DEBUG=%s platform=%s" % (DEBUG, sys.platform))
    ChatApp().mainloop()


if __name__ == "__main__":
    main()
