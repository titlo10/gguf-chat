#!/usr/bin/env python3
import os
import re
import sys
import enum
import json
import queue
import importlib
import threading
import traceback
import faulthandler
import subprocess

for _k in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox


APP_TITLE = "GGUF Чат"
SHIFT_MASK = 0x0001
CREATE_NO_WINDOW = 0x08000000


def _console_attached() -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:
        return False


DEBUG = bool(os.environ.get("GGUF_DEBUG")) or _console_attached()

META_TIMEOUT = 30.0 if DEBUG else 60.0
LOAD_TIMEOUT = 120.0 if DEBUG else 300.0
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


class Backend(enum.Enum):
    VULKAN = enum.auto()
    CPU = enum.auto()
    DEFAULT = enum.auto()


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


def log_path() -> str:
    return os.path.join(app_dir(), "gguf-chat.log")


def cpu_has_avx2() -> bool:
    try:
        if sys.platform.startswith("win"):
            import ctypes
            return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
        with open("/proc/cpuinfo", "r") as f:
            return re.search(r"\bavx2\b", f.read()) is not None
    except Exception:
        return False


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


class NativeLog:
    def __init__(self, path):
        self.path = path
        self.text = ""
        self._ok = False
        self._fh = None
        self._saved = None
        self._start = 0

    def __enter__(self):
        if DEBUG:
            self._ok = False
            return self
        try:
            self._fh = open(self.path, "a+b")
            self._start = self._fh.seek(0, os.SEEK_END)
            self._saved = os.dup(2)
            os.dup2(self._fh.fileno(), 2)
            self._ok = True
        except Exception:
            self._ok = False
        return self

    def __exit__(self, *exc):
        if not self._ok:
            return False
        try:
            os.dup2(self._saved, 2)
            os.close(self._saved)
            self._fh.seek(self._start)
            self.text = self._fh.read().decode("utf-8", "replace")
            self._fh.close()
        except Exception:
            pass
        return False


_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to\s+GPU", re.I)
_LAYER_DEV_RE = re.compile(r"assigned to device\s+([A-Za-z0-9_]+)", re.I)


def parse_offloaded(log_text: str):
    text = log_text or ""
    m = _OFFLOAD_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    devices = _LAYER_DEV_RE.findall(text)
    if not devices:
        return None
    gpu = sum(1 for d in devices if d.upper() != "CPU")
    if gpu == 0:
        return None
    return gpu, len(devices)


class BackendManager:
    def __init__(self):
        self._injected = []

    def candidates(self):
        base = bundle_dir()
        out = []
        vk = os.path.join(base, "llama_vulkan")
        cpu = os.path.join(base, "llama_cpp_cpu")
        if os.path.isdir(os.path.join(vk, "llama_cpp")):
            out.append((Backend.VULKAN, vk))
        if os.path.isdir(os.path.join(cpu, "llama_cpp")):
            out.append((Backend.CPU, cpu))
        if not out:
            out.append((Backend.DEFAULT, None))
        return out

    def import_backend(self, path):
        for name in [m for m in sys.modules if m == "llama_cpp" or m.startswith("llama_cpp.")]:
            del sys.modules[name]
        for p in self._injected:
            try:
                sys.path.remove(p)
            except ValueError:
                pass
        self._injected.clear()
        if path:
            sys.path.insert(0, path)
            self._injected.append(path)
            lib = os.path.join(path, "llama_cpp", "lib")
            if sys.platform.startswith("win") and os.path.isdir(lib):
                try:
                    os.add_dll_directory(lib)
                except Exception:
                    pass
        importlib.invalidate_caches()
        dbg("import_backend: import_module('llama_cpp') from %s" % path)
        mod = importlib.import_module("llama_cpp")
        dbg("import_backend: import_module done")
        return mod


class Attempt:
    def __init__(self, label, path, n_gpu_layers):
        self.label = label
        self.path = path
        self.n_gpu_layers = n_gpu_layers


def _prewarm_backend_threadlib():
    base = bundle_dir()
    cpu = os.path.join(base, "llama_cpp_cpu")
    if not os.path.isdir(os.path.join(cpu, "llama_cpp")):
        return
    sys.path.insert(0, cpu)
    dbg("worker: pre-warm path=%s" % cpu)
    try:
        import numpy  # noqa: F401
        dbg("worker: pre-warm numpy OK (%s)" % getattr(numpy, "__version__", "?"))
    except Exception as exc:
        dbg("worker: pre-warm numpy failed: " + repr(exc))
    finally:
        try:
            sys.path.remove(cpu)
        except ValueError:
            pass


def worker_main():
    try:
        _prewarm_backend_threadlib()
        _run_worker()
    except Exception:
        try:
            with open(log_path(), "a", encoding="utf-8") as f:
                f.write("\n[worker fatal] " + traceback.format_exc() + "\n")
        except Exception:
            pass


def _run_worker():
    stdin = os.fdopen(0, "r", buffering=1, encoding="utf-8", newline="\n")
    stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", newline="\n")

    out_lock = threading.Lock()

    def send(obj):
        with out_lock:
            stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stdout.flush()

    cmds = queue.Queue()
    cancel = threading.Event()

    backends = BackendManager()
    state = {"llm": None}
    dbg("worker ready")
    if DEBUG:
        faulthandler.enable()

    def reader():
        try:
            for line in stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                if msg.get("cmd") == "cancel":
                    cancel.set()
                else:
                    cmds.put(msg)
        except Exception:
            pass
        cmds.put({"cmd": "quit"})

    threading.Thread(target=reader, daemon=True).start()

    while True:
        msg = cmds.get()
        cmd = msg.get("cmd")
        if cmd == "quit":
            break
        elif cmd == "meta":
            if DEBUG:
                faulthandler.dump_traceback_later(25, repeat=True)
            try:
                dbg("meta: import backend " + str(msg.get("backend_path")))
                lc = backends.import_backend(msg.get("backend_path"))
                dbg("meta: vocab_only Llama")
                m = lc.Llama(model_path=msg["model"], vocab_only=True, verbose=False)
                n_layer = None
                try:
                    for k, v in dict(m.metadata).items():
                        if k.endswith(".block_count"):
                            n_layer = int(v)
                            break
                except Exception:
                    pass
                send({"ev": "meta", "n_layer": n_layer})
            except Exception as exc:
                dbg("meta: FAILED " + repr(exc))
                send({"ev": "error", "msg": repr(exc)})
            finally:
                if DEBUG:
                    faulthandler.cancel_dump_traceback_later()
        elif cmd == "load":
            if DEBUG:
                faulthandler.dump_traceback_later(25, repeat=True)
            try:
                dbg("load: import backend " + str(msg.get("backend_path")))
                lc = backends.import_backend(msg.get("backend_path"))
                cap = NativeLog(log_path())
                dbg("load: constructing Llama ngl=%s n_ctx=%s" % (msg.get("ngl"), msg.get("n_ctx")))
                with cap:
                    m = lc.Llama(
                        model_path=msg["model"],
                        n_ctx=int(msg["n_ctx"]),
                        n_gpu_layers=int(msg["ngl"]),
                        verbose=True,
                    )
                dbg("load: Llama constructed OK")
                try:
                    m.verbose = False
                except Exception:
                    pass
                gpu_ok = False
                try:
                    gpu_ok = bool(lc.llama_supports_gpu_offload())
                except Exception:
                    pass
                state["llm"] = m
                dbg("load: sending loaded")
                send({"ev": "loaded", "gpu_ok": gpu_ok, "offloaded": parse_offloaded(cap.text)})
            except Exception as exc:
                dbg("load: FAILED " + repr(exc))
                send({"ev": "error", "msg": repr(exc)})
            finally:
                if DEBUG:
                    faulthandler.cancel_dump_traceback_later()
        elif cmd == "gen":
            m = state["llm"]
            if m is None:
                send({"ev": "error", "msg": "Модель не загружена."})
                continue
            cancel.clear()
            try:
                stream = m.create_chat_completion(
                    messages=msg["messages"], stream=True, **msg.get("params", {})
                )
                stopped = False
                for chunk in stream:
                    if cancel.is_set():
                        stopped = True
                        try:
                            stream.close()
                        except Exception:
                            pass
                        break
                    piece = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    if piece:
                        send({"ev": "token", "t": piece})
                send({"ev": "stopped" if stopped else "done"})
            except Exception as exc:
                send({"ev": "error", "msg": repr(exc)})


class LlmEngine:
    def __init__(self):
        self.backends = BackendManager()
        self.proc = None
        self._q = None
        self.status = "Модель не загружена."
        self._loaded = False
        self._wlock = threading.Lock()

    def _worker_argv(self):
        if getattr(sys, "frozen", False):
            return [sys.executable, "--worker"]
        return [sys.executable, os.path.abspath(__file__), "--worker"]

    def _new_worker(self):
        win = sys.platform.startswith("win")
        thread_env = {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        if DEBUG:
            flags = 0
            werr = None
            wenv = {**os.environ, **thread_env, "GGUF_DEBUG": "1"}
        else:
            flags = CREATE_NO_WINDOW if win else 0
            werr = subprocess.DEVNULL
            wenv = {**os.environ, **thread_env}
        proc = subprocess.Popen(
            self._worker_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=werr,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=flags,
            env=wenv,
        )
        dbg("spawned worker pid=%s" % proc.pid)
        q = queue.Queue()

        def pump():
            try:
                for line in proc.stdout:
                    q.put(line)
            except Exception:
                pass
            q.put(None)

        threading.Thread(target=pump, daemon=True).start()
        return proc, q

    def _send(self, proc, obj):
        with self._wlock:
            proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            proc.stdin.flush()

    @staticmethod
    def _next(q, timeout):
        try:
            line = q.get(timeout=timeout)
        except queue.Empty:
            return "timeout", None
        if line is None:
            return "eof", None
        return "line", line

    @staticmethod
    def _kill(proc):
        if proc is None:
            return
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass

    def shutdown(self):
        self._kill(self.proc)
        self.proc = None
        self._q = None
        self._loaded = False

    def _safe_probe_path(self):
        cands = dict(self.backends.candidates())
        if Backend.CPU in cands:
            return cands[Backend.CPU]
        if Backend.DEFAULT in cands:
            return cands[Backend.DEFAULT]
        if Backend.VULKAN in cands and cpu_has_avx2():
            return cands[Backend.VULKAN]
        return cands.get(Backend.VULKAN)

    def _gpu_ladder(self, requested_ngl, n_layer):
        if requested_ngl <= 0:
            return []
        if not n_layer:
            return [requested_ngl]
        top = n_layer if requested_ngl >= 900 else min(requested_ngl, n_layer)
        steps = []
        for frac in (1.0, 0.75, 0.5, 0.25):
            v = max(1, int(round(top * frac)))
            if v not in steps:
                steps.append(v)
        return steps

    def _plan(self, requested_ngl, n_layer):
        cands = dict(self.backends.candidates())
        plan = []
        if requested_ngl != 0:
            if Backend.VULKAN in cands and cpu_has_avx2():
                path, label = cands[Backend.VULKAN], "Vulkan GPU"
            elif Backend.DEFAULT in cands:
                path, label = cands[Backend.DEFAULT], "GPU"
            else:
                path, label = None, None
            if label:
                for v in self._gpu_ladder(requested_ngl, n_layer):
                    plan.append(Attempt(f"{label} ({v} сл.)", path, v))
        if Backend.CPU in cands:
            plan.append(Attempt("CPU", cands[Backend.CPU], 0))
        elif Backend.VULKAN in cands:
            plan.append(Attempt("CPU (сборка vulkan)", cands[Backend.VULKAN], 0))
        else:
            plan.append(Attempt("CPU", cands.get(Backend.DEFAULT), 0))
        return plan

    def _probe_n_layer(self, model_path):
        proc = None
        try:
            proc, q = self._new_worker()
            self._send(proc, {"cmd": "meta", "backend_path": self._safe_probe_path(), "model": model_path})
            while True:
                kind, line = self._next(q, META_TIMEOUT)
                if kind != "line":
                    return None
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("ev") == "meta":
                    return ev.get("n_layer")
                if ev.get("ev") == "error":
                    return None
        except Exception:
            return None
        finally:
            self._kill(proc)

    def _crash_hint(self):
        try:
            with open(log_path(), errors="replace") as f:
                tail = [s.strip() for s in f.read().splitlines() if s.strip()][-3:]
            joined = " | ".join(tail)
            return "нативный сбой (нехватка VRAM/драйвер). " + joined[-200:]
        except Exception:
            return "нативный сбой (нехватка VRAM/драйвер)"

    def _await_loaded(self, q):
        while True:
            kind, line = self._next(q, LOAD_TIMEOUT)
            if kind == "eof":
                dbg("worker EOF before load result (crashed)")
                return {"ev": "error", "msg": self._crash_hint()}
            if kind == "timeout":
                dbg("LOAD_TIMEOUT %ss with no result" % LOAD_TIMEOUT)
                return {"ev": "error", "msg": "таймаут загрузки (возможно, зависание драйвера)"}
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("ev") in ("loaded", "error"):
                return ev

    def _status_for(self, attempt, offloaded, gpu_ok):
        if offloaded and offloaded[0] > 0:
            return f"{attempt.label}: на GPU выгружено {offloaded[0]}/{offloaded[1]} слоёв"
        if attempt.n_gpu_layers == 0:
            return f"{attempt.label}: работает на CPU"
        if gpu_ok:
            return f"{attempt.label}: загружено (выгрузка на GPU запрошена)"
        return f"{attempt.label}: работает на CPU (без выгрузки)"

    def load(self, model_path: str, n_ctx: int, requested_ngl: int, on_log=None):
        def log(msg):
            if on_log:
                on_log(msg)

        try:
            open(log_path(), "w").close()
        except Exception:
            pass

        self.shutdown()

        n_layer = None
        if requested_ngl != 0:
            log("Определяю число слоёв модели …")
            n_layer = self._probe_n_layer(model_path)

        errors = []
        for attempt in self._plan(requested_ngl, n_layer):
            log(f"Пробую: {attempt.label} …")
            dbg("attempt: %s backend=%s ngl=%s" % (attempt.label, attempt.path, attempt.n_gpu_layers))
            proc, q = self._new_worker()
            try:
                self._send(proc, {
                    "cmd": "load",
                    "backend_path": attempt.path,
                    "model": model_path,
                    "n_ctx": int(n_ctx),
                    "ngl": int(attempt.n_gpu_layers),
                })
                ev = self._await_loaded(q)
            except Exception as exc:
                ev = {"ev": "error", "msg": str(exc)}

            dbg("attempt result: %s %s" % (ev.get("ev"), (ev.get("msg") or "")[:300]))
            if ev.get("ev") == "loaded":
                self.proc = proc
                self._q = q
                self._loaded = True
                self.status = self._status_for(attempt, ev.get("offloaded"), ev.get("gpu_ok"))
                log(self.status)
                return self.status

            reason = (ev.get("msg") or "").splitlines()[0] if ev.get("msg") else "не удалось"
            errors.append(f"{attempt.label}: {reason}")
            log(f"{attempt.label}: не вышло — пробую дальше")
            self._kill(proc)

        raise RuntimeError("Не удалось загрузить модель ни одним способом.\n\n" + "\n".join(errors))

    def is_loaded(self) -> bool:
        return self._loaded and self.proc is not None

    def cancel(self):
        if self.proc is not None:
            try:
                self._send(self.proc, {"cmd": "cancel"})
            except Exception:
                pass

    def stream_chat(self, messages, **params):
        proc, q = self.proc, self._q
        if proc is None or q is None or not self._loaded:
            raise RuntimeError("Модель не загружена.")
        self._send(proc, {"cmd": "gen", "messages": messages, "params": params})
        while True:
            kind, line = self._next(q, GEN_TIMEOUT)
            if kind == "eof":
                self._loaded = False
                raise RuntimeError("процесс генерации аварийно завершился")
            if kind == "timeout":
                raise RuntimeError("таймаут генерации")
            try:
                ev = json.loads(line)
            except Exception:
                continue
            kind = ev.get("ev")
            if kind == "token":
                yield ev.get("t", "")
            elif kind in ("done", "stopped"):
                return
            elif kind == "error":
                raise RuntimeError(ev.get("msg", "ошибка генерации"))


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
    if "--worker" in sys.argv[1:]:
        worker_main()
        return
    dbg("GUI start; DEBUG=%s platform=%s" % (DEBUG, sys.platform))
    ChatApp().mainloop()


if __name__ == "__main__":
    main()
