# Запуск RAG server с переменными из .env
#
#   copy env.example .env
#   .\start.ps1
#
# Параметры:
#   .\start.ps1 -Port 8010
#   .\start.ps1 -EnvFile .env.local

param(
    [string]$EnvFile = ".env",
    [string]$BindHost = "",
    [int]$Port = 0
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return
    }

    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }

        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) {
            return
        }

        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        Set-Item -Path "env:$name" -Value $value
    }
}

Import-DotEnv (Join-Path $Root $EnvFile)

if (-not $env:RAG_ADMIN_TOKEN) {
    Write-Error "RAG_ADMIN_TOKEN is not set. Copy env.example to .env and edit the token."
}

if ($BindHost) {
    $env:RAG_HOST = $BindHost
}
if ($Port -gt 0) {
    $env:RAG_PORT = "$Port"
}

if (-not $env:RAG_HOST) { $env:RAG_HOST = "0.0.0.0" }
if (-not $env:RAG_PORT) { $env:RAG_PORT = "8000" }
if (-not $env:EMBEDDING_URL) { $env:EMBEDDING_URL = "http://127.0.0.1:8002/v1/embeddings" }
if (-not $env:EMBEDDING_MODEL) { $env:EMBEDDING_MODEL = "qwen3-embedding" }
if (-not $env:RAG_STAGING_DIR) { $env:RAG_STAGING_DIR = "./data/staging" }
if (-not $env:RAG_BUNDLE_STATE_DIR) { $env:RAG_BUNDLE_STATE_DIR = "./data/state" }

New-Item -ItemType Directory -Force -Path $env:RAG_STAGING_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:RAG_BUNDLE_STATE_DIR | Out-Null

Write-Host "Starting RAG server on http://$($env:RAG_HOST):$($env:RAG_PORT)"
Write-Host "Staging: $($env:RAG_STAGING_DIR)"
Write-Host "State:   $($env:RAG_BUNDLE_STATE_DIR)"
Write-Host "Embedding: $($env:EMBEDDING_URL) ($($env:EMBEDDING_MODEL))"

python -m uvicorn rag_server:app --host $env:RAG_HOST --port $env:RAG_PORT
