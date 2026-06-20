#!/usr/bin/env python3
"""Command-line entry point for complete MemGraphRAG memory indexing."""

import argparse
import logging
import os
import time
from typing import List

from transformers import AutoTokenizer

from src.MemGraphRAG import MemGraphRAG
from src.utils.config_utils import BaseConfig


def split_text(text: str, tokenizer, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
    previous_limit = getattr(tokenizer, "model_max_length", None)
    try:
        tokenizer.model_max_length = int(1e9)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    finally:
        if previous_limit is not None:
            tokenizer.model_max_length = previous_limit
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(token_ids), step):
        chunk = token_ids[start:start + chunk_size]
        if not chunk:
            break
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        if start + chunk_size >= len(token_ids):
            break
    return chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a complete MemGraphRAG memory index.")
    parser.add_argument("--corpus", required=True, help="UTF-8 text corpus path")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--llm-name", default="gpt-4o-mini")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--tokenizer", default=None, help="Defaults to --embedding-model")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--openie-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--force-index-from-scratch", action="store_true")
    parser.add_argument("--force-openie-from-scratch", action="store_true")
    parser.add_argument("--artifact-mode", choices=["default", "detailed"], default="default")
    parser.add_argument("--memory-max-workers", type=int, default=20)
    parser.add_argument("--schema-sample-facts", type=int, default=0)
    parser.add_argument("--schema-force-reextract", action="store_true")
    parser.add_argument("--ontology-filter-mode", choices=["absolute", "percentile"], default="percentile")
    parser.add_argument("--ontology-min-frequency", type=int, default=2)
    parser.add_argument("--ontology-low-percent", type=float, default=20.0)
    parser.add_argument("--conflict-max-related", type=int, default=20)
    parser.add_argument("--conflict-min-confidence", type=float, default=0.85)
    parser.add_argument("--conflict-all-directions", action="store_true",
                        help="Compare candidates in both directions instead of only against earlier facts")
    parser.add_argument("--resolution-max-component-size", type=int, default=20)
    parser.add_argument("--synonymy-edge-topk", type=int, default=100)
    parser.add_argument("--synonymy-edge-threshold", type=float, default=0.8)
    parser.add_argument("--memory-graph-output-name", default="memory_graph")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with open(args.corpus, "r", encoding="utf-8") as handle:
        corpus = handle.read()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.embedding_model)
    chunks = split_text(corpus, tokenizer, args.chunk_size, args.chunk_overlap)
    docs = [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]
    if not docs:
        raise ValueError(f"Corpus contains no indexable text: {args.corpus}")

    config = BaseConfig(
        save_dir=args.save_dir,
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        openie_mode=args.openie_mode,
        force_index_from_scratch=args.force_index_from_scratch,
        force_openie_from_scratch=args.force_openie_from_scratch,
        save_openie=True,
        corpus_len=len(docs),
        memory_artifact_mode=args.artifact_mode,
        memory_max_workers=args.memory_max_workers,
        schema_extraction_sample_facts=args.schema_sample_facts,
        schema_extraction_force_reextract=args.schema_force_reextract,
        ontology_filter_mode=args.ontology_filter_mode,
        ontology_filter_min_frequency=args.ontology_min_frequency,
        ontology_filter_low_percent=args.ontology_low_percent,
        conflict_max_related_per_target=args.conflict_max_related,
        conflict_min_confidence=args.conflict_min_confidence,
        conflict_streaming_only_previous=not args.conflict_all_directions,
        conflict_resolution_max_component_size=args.resolution_max_component_size,
        synonymy_edge_topk=args.synonymy_edge_topk,
        synonymy_edge_sim_threshold=args.synonymy_edge_threshold,
        memory_graph_output_name=args.memory_graph_output_name,
    )
    logging.basicConfig(level=logging.INFO)
    started = time.time()
    result = MemGraphRAG(global_config=config).index_with_memory(docs)
    print(f"Index completed in {time.time() - started:.2f}s")
    print(f"Memory: {result['artifacts']['memory']}")
    print(f"Memory graph: {result['artifacts']['memory_graph']['json']}")
    print(f"Graph stats: {result['stats']}")


if __name__ == "__main__":
    main()
