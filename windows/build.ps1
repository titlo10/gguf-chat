$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
Set-Location $Repo

$Py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
if (-not (Test-Path ".venv")) {
    Write-Host "==> Создаю venv"
    & $Py -3 -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "==> Скачиваю официальный llama.cpp (CUDA) + CUDA runtime"
$rel = Invoke-RestMethod -Headers @{ "User-Agent" = "gguf-ui" } `
       -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

$cuda = $rel.assets |
    Where-Object { $_.name -match '(?i)bin-win-cuda-.*x64.*\.zip$' -and $_.name -notmatch '(?i)cudart' } |
    Sort-Object name -Descending | Select-Object -First 1
if (-not $cuda) { throw "Не найден ассет llama.cpp CUDA в релизе $($rel.tag_name)" }

$ver = ([regex]::Match($cuda.name, '(?i)cuda-(\d+\.\d+)')).Groups[1].Value
$cudart = $rel.assets | Where-Object { $_.name -match '(?i)cudart' -and $_.name -match [regex]::Escape($ver) } | Select-Object -First 1
if (-not $cudart) {
    $cudart = $rel.assets | Where-Object { $_.name -match '(?i)cudart' } | Sort-Object name -Descending | Select-Object -First 1
}
if (-not $cudart) { throw "Не найден ассет cudart в релизе $($rel.tag_name)" }

Write-Host "Релиз: $($rel.tag_name)  |  CUDA: $($cuda.name)  |  cudart: $($cudart.name)"

Remove-Item -Recurse -Force build\_dl, build\llama_server -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force build\_dl, build\llama_server | Out-Null
Invoke-WebRequest -Uri $cuda.browser_download_url   -OutFile build\_dl\cuda.zip
Invoke-WebRequest -Uri $cudart.browser_download_url -OutFile build\_dl\cudart.zip
Expand-Archive build\_dl\cuda.zip   -DestinationPath build\_dl\cuda   -Force
Expand-Archive build\_dl\cudart.zip -DestinationPath build\_dl\cudart -Force

$exe = Get-ChildItem build\_dl\cuda -Recurse -File |
    Where-Object { $_.Name -match '(?i)llama-server\.exe$' } | Select-Object -First 1
if (-not $exe) { throw "llama-server.exe не найден в архиве" }

Copy-Item (Join-Path $exe.Directory.FullName '*') build\llama_server -Recurse -Force
Get-ChildItem build\_dl\cudart -Recurse -Filter *.dll | ForEach-Object {
    Copy-Item $_.FullName build\llama_server -Force
}
Remove-Item -Recurse -Force build\_dl

Write-Host ""
Write-Host "Готово. Положите .gguf в $Repo и запустите:  windows\start.bat"
