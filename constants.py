SERVER_HOST = "127.0.0.1"
HEALTH_PATH = "/health"
CHAT_PATH = "/v1/chat/completions"
HEALTH_OK_MARKER = '"ok"'
SSE_PREFIX = "data:"
SSE_DONE = "[DONE]"
URL_TEMPLATE = "http://%s:%d"

LOAD_TIMEOUT = 300.0
GEN_TIMEOUT = 600.0
HEALTH_TIMEOUT = 2.0
POLL_INTERVAL = 0.4
STOP_TIMEOUT = 5
DIALOG_TIMEOUT = 600

CREATE_NO_WINDOW = 0x08000000

SERVER_SUBDIR = "llama_server"
BUILD_SUBDIR = "build"
SERVER_NAMES = ("llama-server", "llama-server.exe", "server", "server.exe")
GPU_MARKERS = ("ggml_cuda_init", "using device cuda", "cuda0")
STREAM_PARAM_KEYS = ("temperature", "top_p", "top_k", "max_tokens", "seed", "repeat_penalty")

STDERR_MAX_LINES = 400
STDERR_TRIM_LINES = 200
STDERR_TAIL_LINES = 8
ENGINE_ERROR_LIMIT = 300
NGL_ALL = 999
NGL_ALL_THRESHOLD = 900

GGUF_EXT = ".gguf"
PDF_EXT = ".pdf"
DOCX_EXT = ".docx"
DOC_ENCODINGS = ("utf-8", "cp1251", "latin-1")
DOC_FILE_TYPES = [".pdf", ".docx", ".txt", ".md"]

ENV_DEBUG = "GGUF_DEBUG"
ENV_SERVER = "LLAMA_SERVER"
ENV_MODELS_ROOT = "GGUF_MODELS_ROOT"
ENV_GRADIO_ANALYTICS = "GRADIO_ANALYTICS_ENABLED"
ENV_HF_OFFLINE = "HF_HUB_OFFLINE"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 1.1
DEFAULT_MAX_TOKENS = 512
DEFAULT_SEED = -1
DEFAULT_N_CTX = 4096
DEFAULT_N_GPU_LAYERS = 999

DOC_RESERVE_TOKENS = 600
DOC_CHARS_PER_TOKEN = 4
DOC_MIN_BUDGET = 2000

APP_TITLE = "GGUF Чат"
DEFAULT_SYSTEM_PROMPT = "Ты — полезный ассистент."

LABEL_MODEL = "Модель .gguf"
PLACEHOLDER_MODEL = "Путь к файлу .gguf"
LABEL_BROWSE = "📂 Обзор…"
LABEL_SYSTEM = "Системный промпт"
LABEL_DOCUMENT = "Документ (pdf / docx / txt / md)"
LABEL_PARAMS = "Параметры генерации"
LABEL_TEMPERATURE = "Температура"
LABEL_TOP_P = "Top-p"
LABEL_TOP_K = "Top-k"
LABEL_REPEAT_PENALTY = "Штраф за повтор"
LABEL_MAX_TOKENS = "Макс. токенов ответа"
LABEL_SEED = "Сид (-1 = случайный)"
LABEL_LOAD_SECTION = "Загрузка модели (применяется при первом сообщении)"
LABEL_N_CTX = "Длина контекста (n_ctx)"
LABEL_N_GPU = "Слоёв на GPU (999=все, 0=CPU)"
DIALOG_TITLE = "Выберите модель GGUF"
DIALOG_FILTER_WIN = "GGUF (*.gguf)|*.gguf|Все файлы (*.*)|*.*"
DIALOG_FILTER_ZENITY_GGUF = "GGUF | *.gguf *.GGUF"
DIALOG_FILTER_ZENITY_ALL = "Все файлы | *"
DIALOG_FILTER_KDIALOG = "*.gguf | GGUF"

MSG_NO_MODEL = "Выберите модель .gguf в проводнике слева."
MSG_LOADING = "⏳ Загружаю модель …"
MSG_LOAD_FAILED = "❌ Не удалось загрузить модель:\n\n%s"
MSG_GEN_ERROR = "❌ Ошибка генерации:\n\n%s"
MSG_DOC_ERROR = "[не удалось прочитать документ: %s]"
DOC_CONTEXT_HEADER = "Контекст из документа:\n"
THINKING_DONE = "💭 Размышления"
THINKING_PROGRESS = "💭 Размышляю …"

STATUS_NOT_LOADED = "Модель не загружена."
STATUS_CPU = "Работает на CPU"
STATUS_GPU = "Загружено на видеокарту (CUDA)"
STATUS_CPU_FALLBACK = "Загружено (CPU — видеокарта не задействована)"
LOG_STARTING = "Запускаю движок (слоёв на GPU: %s) …"
LOG_FALLBACK = "Не вышло, пробую запасной вариант (CPU) …"
ERR_NO_SERVER = "Не найден llama-server (папка llama_server рядом с программой)."
ERR_LOAD_FAILED = "Не удалось загрузить модель.\n\n"
ERR_UNKNOWN = "неизвестная ошибка"
ERR_TIMEOUT = "таймаут загрузки"
ERR_ENGINE = "ошибка движка: "
ERR_ENGINE_UNREACHABLE = "движок недоступен: "
DBG_PREFIX = "[gguf] "
