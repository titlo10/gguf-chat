import os
import sys

IS_WIN = sys.platform.startswith("win")
APP_NAME = "GGUFChat"

datas = []
binaries = []
hiddenimports = ["pypdf", "docx"]

try:
    from PyInstaller.utils.hooks import collect_data_files
    datas += collect_data_files("docx")
except Exception:
    pass

if IS_WIN:
    datas += [
        (os.path.join("build", "llama_vulkan"), "llama_vulkan"),
        (os.path.join("build", "llama_cpu"), "llama_cpp_cpu"),
    ]
    _vk = os.path.join("redist", "vulkan-1.dll")
    if os.path.isfile(_vk):
        binaries += [(_vk, ".")]
else:
    from PyInstaller.utils.hooks import collect_all
    _d, _b, _h = collect_all("llama_cpp")
    datas += _d
    binaries += _b
    hiddenimports += _h

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

console = (not IS_WIN) or bool(os.environ.get("GGUF_CONSOLE"))

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=console,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
