#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Runtime environment. Prefer exporting the key before running:
#   export OPENAI_API_KEY="your-remote-api-key"
# Alternatively, replace the empty default below for local-only use.
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running run_index.sh}"
export OPENAI_API_KEY
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Input and model parameters.
CORPUS="/home/wcj/data/project/memgraphrag/input/smallcorpus/smallcorpus.txt"
SAVE_DIR="/home/wcj/data/project/memgraphrag/outputs/smallcorpus"
LLM_NAME="gpt-4o-mini"
LLM_BASE_URL=""
EMBEDDING_MODEL="/home/wcj/data/model/bge-large-en-v1.5"
TOKENIZER="$EMBEDDING_MODEL"

# Pipeline parameters.
CHUNK_SIZE=256
CHUNK_OVERLAP=32
ARTIFACT_MODE="default"       
MEMORY_MAX_WORKERS=20
ONTOLOGY_FILTER_MODE="percentile"
ONTOLOGY_LOW_PERCENT=20
CONFLICT_MIN_CONFIDENCE=0.85
SYNONYMY_EDGE_TOPK=100
SYNONYMY_EDGE_THRESHOLD=0.8

"$PYTHON" "$SCRIPT_DIR/index.py" \
  --corpus "$CORPUS" \
  --save-dir "$SAVE_DIR" \
  --llm-name "$LLM_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --tokenizer "$TOKENIZER" \
  --chunk-size "$CHUNK_SIZE" \
  --chunk-overlap "$CHUNK_OVERLAP" \
  --artifact-mode "$ARTIFACT_MODE" \
  --memory-max-workers "$MEMORY_MAX_WORKERS" \
  --ontology-filter-mode "$ONTOLOGY_FILTER_MODE" \
  --ontology-low-percent "$ONTOLOGY_LOW_PERCENT" \
  --conflict-min-confidence "$CONFLICT_MIN_CONFIDENCE" \
  --synonymy-edge-topk "$SYNONYMY_EDGE_TOPK" \
  --synonymy-edge-threshold "$SYNONYMY_EDGE_THRESHOLD"
