#!/usr/bin/env bash
set -euo pipefail

# 当前脚本所在目录，用于拼接项目内的相对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 使用的 GPU 编号
export CUDA_VISIBLE_DEVICES="0"

# OpenAI 兼容接口的 API Key；使用无鉴权的本地服务时可留空
export OPENAI_API_KEY=

# 问题数据集 JSON 文件路径
QUESTIONS="$SCRIPT_DIR/input/smallcorpus/small-questions.json"

# 已建好图和 embedding 索引的保存目录，直接指向当前数据集目录
SAVE_DIR="$SCRIPT_DIR/outputs/smallcorpus"

# 最终问答与评测结果 JSON 文件的保存路径
OUTPUT="$SCRIPT_DIR/results/smallcorpus/qa_results.json"

# 问题类型，可设置为 type1、type2、type3、type4；all 表示使用所有类型
QUESTION_TYPE="all"

# 采样的问题数量；0 表示使用全部问题
SAMPLE_NUM=0

# LLM 模型名称，必须与构建索引时使用的名称一致
LLM_NAME="gpt-4o-mini"

# OpenAI 兼容的 LLM 服务地址
LLM_BASE_URL="https://apis.aaife.cn/v1"

# Embedding 模型路径，必须与构建索引时使用的路径一致
EMBEDDING_MODEL="/home/wcj/data/model/bge-large-en-v1.5"

# 是否跳过 LLM 对三元组的 rerank：true 使用相似度阈值过滤，false 使用 LLM 过滤
SKIP_LLM_RERANK=true

# 跳过 LLM rerank 时，仅保留原始相似度不低于该阈值的三元组
FACT_SIMILARITY_THRESHOLD=0.6

# true：跳过 fact top-k，直接对全部 facts 使用原始分数做阈值过滤
# false：先选择候选 facts，再使用相似度阈值过滤
USE_RAW_THRESHOLD_FILTER=true

"$PYTHON" "$SCRIPT_DIR/retrieval_dataset_test.py" \
  --questions "$QUESTIONS" \
  --save-dir "$SAVE_DIR" \
  --output "$OUTPUT" \
  --llm-name "$LLM_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --question-type "$QUESTION_TYPE" \
  --sample-num "$SAMPLE_NUM" \
  --skip-fact-rerank "$SKIP_LLM_RERANK" \
  --fact-similarity-threshold "$FACT_SIMILARITY_THRESHOLD" \
  --use-raw-threshold-filter "$USE_RAW_THRESHOLD_FILTER"
