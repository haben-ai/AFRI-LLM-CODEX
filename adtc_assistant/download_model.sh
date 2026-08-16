#!/usr/bin/env bash
# Downloads both model assets this project needs:
#   1. The primary LLM: Qwen2.5-Coder-1.5B-Instruct, Q4_K_M GGUF (llama.cpp).
#   2. The offline translation layer: NLLB-200-distilled-600M, pre-quantized
#      to CTranslate2 INT8 (huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8).
#
# Note on file names: this script downloads the NLLB model's REAL file names
# as published in that repo (shared_vocabulary.txt, sentencepiece.bpe.model,
# plus the transformers-compatible tokenizer.json/tokenizer_config.json/
# special_tokens_map.json trio) -- verified against the actual repo listing
# rather than assumed, since community CTranslate2 conversions don't all use
# identical file names.
set -e

MODEL_DIR="model"

GGUF_FILE="${MODEL_DIR}/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
GGUF_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

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

log() {
    echo "[download_model] $1"
}

download() {
    local url="$1"
    local dest="$2"

    if [ -f "$dest" ]; then
        log "Already present, skipping: $dest"
        return 0
    fi

    log "Downloading $(basename "$dest") -> $dest"
    if command -v curl &> /dev/null; then
        curl -L --fail -o "$dest" "$url"
    elif command -v wget &> /dev/null; then
        wget -O "$dest" "$url"
    else
        echo "Error: neither curl nor wget is available on this system." >&2
        exit 1
    fi
}

mkdir -p "$MODEL_DIR"
mkdir -p "$NLLB_DIR"

log "== Primary LLM: Qwen2.5-Coder-1.5B-Instruct (Q4_K_M GGUF, ~1.1 GB) =="
download "$GGUF_URL" "$GGUF_FILE"

log "== Translation layer: NLLB-200-distilled-600M, CTranslate2 INT8 (~650 MB total) =="
for f in "${NLLB_FILES[@]}"; do
    download "${NLLB_BASE_URL}/${f}" "${NLLB_DIR}/${f}"
done

chmod +x "$0" 2>/dev/null || true

log "All model assets present:"
log "  $GGUF_FILE"
log "  $NLLB_DIR/ ($(ls -1 "$NLLB_DIR" 2>/dev/null | wc -l | tr -d ' ') files)"
log "Done."
