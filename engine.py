import os
import sys
import enum
import json
import time
import socket
import threading
import subprocess
import urllib.error
import urllib.request

import constants as const


DEBUG = bool(os.environ.get(const.ENV_DEBUG))


def dbg(msg):
    if not DEBUG:
        return
    try:
        print(const.DBG_PREFIX + str(msg), file=sys.stderr, flush=True)
    except Exception:
        pass


class Role(enum.StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Channel(enum.StrEnum):
    CONTENT = "content"
    REASONING = "reasoning"


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
            if n.lower().endswith(const.GGUF_EXT) and os.path.isfile(os.path.join(folder, n))
        ]
    except OSError:
        names = []
    return sorted(names)


class LlmEngine:
    def __init__(self):
        self.proc = None
        self.port = None
        self.status = const.STATUS_NOT_LOADED
        self._loaded = False
        self._stderr = []
        self._cancel = threading.Event()

    def _server_exe(self):
        bases = (bundle_dir(), os.path.join(app_dir(), const.BUILD_SUBDIR))
        for base in bases:
            for name in const.SERVER_NAMES:
                path = os.path.join(base, const.SERVER_SUBDIR, name)
                if os.path.isfile(path):
                    return path
        env = os.environ.get(const.ENV_SERVER)
        if env and os.path.isfile(env):
            return env
        return None

    @staticmethod
    def _free_port():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((const.SERVER_HOST, 0))
            return s.getsockname()[1]
        finally:
            s.close()

    def _base_url(self):
        return const.URL_TEMPLATE % (const.SERVER_HOST, self.port)

    def _drain_stderr(self, proc):
        try:
            for line in proc.stderr:
                line = line.rstrip("\n")
                self._stderr.append(line)
                if len(self._stderr) > const.STDERR_MAX_LINES:
                    del self._stderr[:const.STDERR_TRIM_LINES]
                dbg("server: " + line)
        except Exception:
            pass

    def _stderr_tail(self, n=const.STDERR_TAIL_LINES):
        return "\n".join(self._stderr[-n:])

    def _health_ok(self):
        try:
            with urllib.request.urlopen(self._base_url() + const.HEALTH_PATH, timeout=const.HEALTH_TIMEOUT) as r:
                body = r.read().decode("utf-8", "replace")
            return r.status == 200 and const.HEALTH_OK_MARKER in body
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
            proc.wait(timeout=const.STOP_TIMEOUT)
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
            "--host", const.SERVER_HOST,
            "--port", str(self.port),
        ]
        win = sys.platform.startswith("win")
        flags = const.CREATE_NO_WINDOW if (win and not DEBUG) else 0
        env = os.environ.copy()
        if not win:
            exe_dir = os.path.dirname(os.path.abspath(exe))
            prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = exe_dir + (os.pathsep + prev if prev else "")
        dbg("server spawn: %s" % " ".join(argv))
        self.proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=flags,
        )
        threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True).start()

        deadline = time.monotonic() + const.LOAD_TIMEOUT
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return False, self._stderr_tail()
            if self._health_ok():
                return True, ""
            time.sleep(const.POLL_INTERVAL)
        return False, const.ERR_TIMEOUT

    def _status_from_log(self, ngl):
        text = "\n".join(self._stderr).lower()
        gpu = any(marker in text for marker in const.GPU_MARKERS)
        if ngl == 0:
            return const.STATUS_CPU
        if gpu:
            return const.STATUS_GPU
        return const.STATUS_CPU_FALLBACK

    def load(self, model_path: str, n_ctx: int, requested_ngl: int, on_log=None):
        def log(msg):
            if on_log:
                on_log(msg)

        self.shutdown()
        exe = self._server_exe()
        if not exe:
            raise RuntimeError(const.ERR_NO_SERVER)

        req = const.NGL_ALL if int(requested_ngl) >= const.NGL_ALL_THRESHOLD else int(requested_ngl)
        attempts = [req] if req == 0 else [req, 0]
        last = ""
        for ngl in attempts:
            log(const.LOG_STARTING % ngl)
            ok, err = self._start(exe, model_path, n_ctx, ngl, log)
            if ok:
                self._loaded = True
                self.status = self._status_from_log(ngl)
                log(self.status)
                return self.status
            last = err
            self.shutdown()
            if ngl != attempts[-1]:
                log(const.LOG_FALLBACK)

        raise RuntimeError(const.ERR_LOAD_FAILED + (last or const.ERR_UNKNOWN))

    def cancel(self):
        self._cancel.set()

    def stream_chat(self, messages, **params):
        if not self.is_loaded():
            raise RuntimeError(const.STATUS_NOT_LOADED)
        self._cancel.clear()

        payload = {"messages": messages, "stream": True}
        for key in const.STREAM_PARAM_KEYS:
            if params.get(key) is not None:
                payload[key] = params[key]

        req = urllib.request.Request(
            self._base_url() + const.CHAT_PATH,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=const.GEN_TIMEOUT)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(const.ERR_ENGINE + detail[:const.ENGINE_ERROR_LIMIT])
        except Exception as exc:
            raise RuntimeError(const.ERR_ENGINE_UNREACHABLE + str(exc))

        try:
            for raw in resp:
                if self._cancel.is_set():
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith(const.SSE_PREFIX):
                    continue
                chunk = line[len(const.SSE_PREFIX):].strip()
                if chunk == const.SSE_DONE:
                    break
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    yield Channel.REASONING, reasoning
                content = delta.get("content")
                if content:
                    yield Channel.CONTENT, content
        finally:
            try:
                resp.close()
            except Exception:
                pass
