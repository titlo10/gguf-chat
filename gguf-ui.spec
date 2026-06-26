import os
import sys

IS_WIN = sys.platform.startswith("win")
APP_NAME = os.environ.get("GGUF_APP_NAME", "GGUFChat")

datas = []
binaries = []
hiddenimports = ["pypdf", "docx"]

try:
    from PyInstaller.utils.hooks import collect_data_files
    datas += collect_data_files("docx")
except Exception:
    pass

_srv = os.path.join("build", "llama_server")
if os.path.isdir(_srv):
    datas += [(_srv, "llama_server")]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
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
