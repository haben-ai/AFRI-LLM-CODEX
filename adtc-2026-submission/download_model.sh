#!/usr/bin/env bash
set -e

MODEL_DIR="model"
MODEL_FILE="${MODEL_DIR}/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

# NLLB-200-distilled-600M, pre-quantized to CTranslate2 INT8. Only needed for
# --pedagogical mode's regional-language translation; the default JSON-patch
# code-fixing mode never touches this. File names verified against the real
# repo listing, not assumed (community CTranslate2 conversions don't all use
# identical file names).
NLLB_DIR="${MODEL_DIR}/nllb_ct2"
NLLB_BASE_URL="https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8/resolve/main"
NLLB_FILES=(
    "config.json"
    "model.bin"
    "sentencepiece.bpe.model"
    "shared_vocabulary.txt"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer_config.json"
)

download() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "Already exists, skipping: ${dest}"
        return 0
    fi
    echo "Downloading $(basename "$dest") -> ${dest}"
    if command -v curl &> /dev/null; then
        curl -L --fail -o "$dest" "$url"
    elif command -v wget &> /dev/null; then
        wget -O "$dest" "$url"
    else
        echo "Error: Neither curl nor wget found on system." >&2
        exit 1
    fi
}

mkdir -p "${MODEL_DIR}"
mkdir -p "${NLLB_DIR}"

echo "== Primary LLM: Qwen2.5-Coder-1.5B-Instruct (Q4_K_M GGUF, ~1.1 GB) =="
download "$MODEL_URL" "$MODEL_FILE"

echo "== Translation layer (--pedagogical mode): NLLB-200-distilled-600M, CTranslate2 INT8 (~650 MB) =="
for f in "${NLLB_FILES[@]}"; do
    download "${NLLB_BASE_URL}/${f}" "${NLLB_DIR}/${f}"
done

echo "Download complete."
