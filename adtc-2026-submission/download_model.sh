#!/usr/bin/env bash
set -e

MODEL_DIR="model"
MODEL_FILE="${MODEL_DIR}/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_FILE}" ]; then
    echo "Model weight already exists at ${MODEL_FILE}. Skipping download."
    exit 0
fi

echo "Downloading Qwen2.5-Coder-1.5B-Instruct (Q4_K_M GGUF)..."
if command -v curl &> /dev/null; then
    curl -L -o "${MODEL_FILE}" "${MODEL_URL}"
elif command -v wget &> /dev/null; then
    wget -O "${MODEL_FILE}" "${MODEL_URL}"
else
    echo "Error: Neither curl nor wget found on system." >&2
    exit 1
fi

echo "Download complete: ${MODEL_FILE}"
