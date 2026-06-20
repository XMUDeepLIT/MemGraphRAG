# MemGraphRAG: Memory-based Multi-Agent System for Graph Retrieval-Augmented Generation

<div align="center">
  <a href="https://arxiv.org/abs/2606.00610"><img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&style=flat-square" alt="arXiv"></a>
  <a href="https://github.com/XMUDeepLIT/MemGraphRAG"><img src="https://img.shields.io/github/stars/XMUDeepLIT/MemGraphRAG?style=flat-square" alt="GitHub Stars"></a>
  <a href="https://github.com/XMUDeepLIT/MemGraphRAG/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"></a>
</div>

<br>

> A memory-enhanced GraphRAG framework that connects unstructured passages, extracted facts, and abstract schemas in a three-layer memory for reliable retrieval and generation.

## 🎉 News
- **[2026-05-17]** Our **[MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG)** for memory-enhanced RAG is accepted by KDD'26.
- **[2026-04-07]** Our **[ProbeRAG](https://github.com/LinfengGao/ProbeRAG.git)** for RAG faithfulness is accepted by ACL'26.
- **[2026-04-07]** Our **[BAPO](https://github.com/Liushiyu-0709/BAPO-Reliable-Search.git)** for reliable agentic search is accepted by ACL'26.
- **[2026-04-07]** Our **[LegalGraphRAG](https://github.com/XMUDeepLIT/LegalGraphRAG.git)** for reliable legal reasoning is accepted by ACL'26.
- **[2026-04-07]** Our **[LogicPoison](https://github.com/Jord8061/logicPoison.git)**, a GraphRAG attack model, is accepted by ACL'26.
- **[2026-01-26]** Our **[LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)** for efficient GraphRAG is accepted by ICLR’26.
- **[2026-01-26]** Our **[GraphRAG Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)** is accepted by ICLR’26.
- **[2025-11-08]** Our **[LogicRAG](https://github.com/chensyCN/LogicRAG.git)** is accepted by AAAI'26.
- **[2025-10-27]** We release **[LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)**, a relation-free graph construction method for efficient GraphRAG.
- **[2025-06-06]** We release the **[GraphRAG Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git)** for evaluating GraphRAG models.
- **[2025-05-14]** We release the [GraphRAG Benchmark dataset](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench).
- **[2025-01-21]** We release the [GraphRAG survey](https://github.com/DEEP-PolyU/Awesome-GraphRAG).

📃 Please [cite our paper](#-citation) if you find this repository helpful.

📫 Contact: `{xiangzhishang,wuchuanjie}@stu.xmu.edu.cn`, `qinggangzhang@jlu.edu.cn`

## 🏴 Overview

<p align="center">
  <img src="https://raw.githubusercontent.com/XMUDeepLIT/MemGraphRAG/main/framework.png" width="90%" alt="MemGraphRAG Framework"/>
</p>

MemGraphRAG organizes knowledge into three connected layers:

- **Schema layer:** abstract ontology triples such as `(head type, relation, tail type)`.
- **Fact layer:** concrete relation triples extracted from the corpus.
- **Passage layer:** original text chunks that support the facts.


###🔥 Key Features

- **Three-layer memory:** bidirectional links among schemas, facts, and source passages.
- **Ontology induction:** abstracts facts into reusable schemas and filters low-frequency patterns.
- **Conflict-aware construction:** detects hard conflicts with passage evidence and resolves connected conflict groups.
- **Memory-derived graph:** creates type, entity, and passage nodes after conflict resolution.
- **Graph-enhanced retrieval:** combines embedding similarity and Personalized PageRank.
- **Batch QA:** records answers, retrieved documents, scores, latency, and token usage.

---

## 🛠 Quickstart Guide

### 1. Environment Setup

Python 3.10 or later is recommended. The online pipeline uses an OpenAI-compatible LLM endpoint and a local Hugging Face embedding model.

```bash
conda create -n memgraphrag python=3.10
conda activate memgraphrag

git clone https://github.com/XMUDeepLIT/MemGraphRAG.git
cd MemGraphRAG
pip install -r requirements.txt
```

For offline OpenIE, also install `vllm` or `llama-factory`. Optional embedding backends may require packages such as `gritlm`.

### 2. Configure Models and Credentials

```bash
export OPENAI_API_KEY="your-api-key"
export CUDA_VISIBLE_DEVICES="0"

LLM_NAME="gpt-4o-mini"
LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
EMBEDDING_MODEL="/path/to/bge-large-en-v1.5"
```

Use the same `LLM_NAME` and `EMBEDDING_MODEL` for indexing and retrieval because their values identify the saved graph and embedding stores.

### 3. Run Indexing

Run from the repository root. The corpus must be a UTF-8 plain-text file.

```bash
python code/index.py \
  --corpus datasets/corpus/corpus.txt \
  --save-dir outputs/corpus \
  --llm-name "$LLM_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --tokenizer "$EMBEDDING_MODEL" \
  --chunk-size 256 \
  --chunk-overlap 32 \
  --artifact-mode default
```

Add `--force-index-from-scratch` and `--force-openie-from-scratch` to rebuild the corresponding caches.

```text
outputs/corpus/
├── openie_results_ner_<llm>.json
├── initial_memory_with_schema.json
├── memory.json
├── graph_from_memory/
│   ├── memory_graph.json
│   └── memory_graph.graphml
└── <llm>_<embedding_model>/
    ├── graph.graphml
    ├── chunk_embeddings/
    ├── entity_embeddings/
    └── fact_embeddings/
```

### 4. Run Retrieval and Question Answering

```bash
python code/retrieval_dataset_test.py \
  --questions datasets/corpus/small-questions.json \
  --save-dir outputs/corpus \
  --output results/corpus/qa_results.json \
  --llm-name "$LLM_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --question-type all \
  --sample-num 0 \
  --skip-fact-rerank true \
  --fact-similarity-threshold 0.4 \
  --use-raw-threshold-filter true
```

The result JSON contains the run summary, effective configuration, solutions, raw responses, metadata, retrieved passages. 

`code/run_index.sh` and `code/run_retrieval_test.sh` are editable launch templates. Adjust their interpreter, paths, endpoint, model, and GPU settings before use.

## 📝 Dataset Format

The QA runner accepts a flat list of question objects or questions grouped by type:

```json
[
  {
    "source": "dataset-name",
    "questions": {
      "type1": [
        {
          "id": "type1_0",
          "question": "Your question",
          "answer": "Gold answer",
          "evidence": ["Optional supporting evidence"]
        }
      ]
    }
  }
]
```

Set `--question-type type1` for one group or `--question-type all` for all groups. `--sample-num 0` loads every question.

## 📦 Code Structure

```text
📦 .
├── 📂 code
│   ├── 📂 src
│   │   ├── 📂 embedding_model        # Embedding backends
│   │   ├── 📂 evaluation             # Retrieval and QA metrics
│   │   ├── 📂 information_extraction # Online and offline OpenIE
│   │   ├── 📂 llm                    # OpenAI-compatible and vLLM backends
│   │   ├── 📂 prompts                # Extraction, linking, QA, memory prompts
│   │   ├── 📂 utils                  # Configuration and utilities
│   │   ├── MemGraphRAG.py            # Pipeline orchestration
│   │   ├── Memory.py                 # Three-layer memory
│   │   ├── embedding_store.py        # Persistent embedding stores
│   │   └── rerank.py                 # Fact reranking and filtering
│   ├── index.py                      # Indexing CLI
│   ├── retrieval_dataset_test.py     # Retrieval and QA CLI
│   ├── run_index.sh                  # Indexing template
│   └── run_retrieval_test.sh         # Retrieval template
├── 📂 datasets                       # Example corpora and QA datasets
├── 📂 outputs                        # Generated artifacts
└── 📜 README.md
```

---

## 🙏 Acknowledgements

Our framework builds upon the excellent work [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG). We also thank the open-source communities behind Hugging Face Transformers, OpenAI-compatible APIs, and igraph.

## 🍀 Citation

```bibtex
@article{wu2026memgraphrag,
  title={MemGraphRAG: Memory-based Multi-Agent System for Graph Retrieval-Augmented Generation},
  author={Wu, Chuanjie and Xiang, Zhishang and Tang, Yunbo and Chen, Zerui and Zhang, Qinggang and Su, Jinsong},
  journal={arXiv preprint arXiv:2606.00610},
  year={2026}
}
```

**Links:** [arXiv](https://arxiv.org/abs/2606.00610)

## 📄 License

This project is released under the [MIT License](https://github.com/XMUDeepLIT/MemGraphRAG/blob/main/LICENSE).
