#!/usr/bin/env python3
"""Run retrieval and question answering on a question dataset."""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

LOGGER = logging.getLogger(__name__)
DEFAULT_RERANK_DSPY_FILE = str(
    Path(__file__).resolve().parent
    / "src/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json"
)


def str_to_bool(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def load_question_samples(
    questions_path: str,
    question_type: Optional[str] = None,
    sample_num: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load flat datasets or datasets whose questions are grouped by type."""
    with open(questions_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    roots = data if isinstance(data, list) else [data]
    samples: List[Dict[str, Any]] = []
    for root in roots:
        if not isinstance(root, dict):
            continue

        grouped_questions = root.get("questions")
        if isinstance(grouped_questions, dict):
            if question_type and question_type.lower() != "all":
                if question_type not in grouped_questions:
                    available = ", ".join(sorted(grouped_questions))
                    raise ValueError(
                        f"Question type {question_type!r} was not found. "
                        f"Available types: {available}"
                    )
                groups = [(question_type, grouped_questions[question_type])]
            else:
                groups = grouped_questions.items()

            for group_name, group_samples in groups:
                for sample in group_samples:
                    item = dict(sample)
                    item.setdefault("question_type", group_name)
                    samples.append(item)
        elif "question" in root:
            samples.append(dict(root))

    if not samples:
        raise ValueError(f"No question samples were found in {questions_path}")
    if sample_num is not None and sample_num > 0:
        samples = samples[:sample_num]
    return samples


def _result_record(
    sample: Dict[str, Any],
    solution: Any,
    response: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    record = {
        key: sample[key]
        for key in (
            "id",
            "_id",
            "source",
            "question_type",
            "question",
            "answer",
            "evidence",
            "supporting_facts",
        )
        if key in sample
    }
    record["predicted_answer"] = solution.answer
    record["raw_response"] = response
    record["metadata"] = metadata
    record["retrieved_docs"] = solution.docs
    record["doc_scores"] = (
        [float(score) for score in solution.doc_scores]
        if solution.doc_scores is not None
        else None
    )
    return record


def test_dataset(
    questions_path: str,
    save_dir: str,
    output_path: str,
    llm_name: str,
    embedding_model_name: str,
    llm_base_url: Optional[str] = None,
    question_type: Optional[str] = None,
    sample_num: Optional[int] = None,
    retrieval_top_k: int = 10,
    linking_top_k: int = 50,
    passage_node_weight: float = 0.05,
    damping: float = 0.5,
    embedding_batch_size: int = 8,
    rerank_dspy_file_path: Optional[str] = DEFAULT_RERANK_DSPY_FILE,
    skip_fact_rerank: bool = True,
    fact_similarity_threshold: float = 0.2,
    use_raw_threshold_filter: bool = True,
    retrieval_max_workers: int = 5,
    qa_top_k: int = 10,
    qa_max_workers: int = 5,
    max_qa_steps: int = 3,
    max_new_tokens: Optional[int] = 500,
) -> Dict[str, Any]:
    """Run MemGraphRAG retrieval and QA without answer evaluation."""
    # Delay the heavyweight model imports so data loading and --help do not start
    # multiprocessing/model initialization.
    from src.MemGraphRAG import MemGraphRAG
    from src.utils.config_utils import BaseConfig

    samples = load_question_samples(questions_path, question_type, sample_num)
    queries = [sample["question"] for sample in samples]

    llm_label = llm_name.replace("/", "_")
    embedding_label = embedding_model_name.replace("/", "_")
    graph_path = Path(save_dir) / f"{llm_label}_{embedding_label}" / "graph.graphml"
    if not graph_path.is_file():
        raise FileNotFoundError(
            f"Existing graph was not found at {graph_path}. Check save_dir and model names."
        )

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        embedding_model_name=embedding_model_name,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        openie_mode="online",
        rerank_dspy_file_path=rerank_dspy_file_path,
        retrieval_top_k=retrieval_top_k,
        linking_top_k=linking_top_k,
        passage_node_weight=passage_node_weight,
        damping=damping,
        embedding_batch_size=embedding_batch_size,
        corpus_len=None,
        skip_fact_rerank=skip_fact_rerank,
        fact_similarity_threshold=fact_similarity_threshold,
        use_raw_threshold_filter=use_raw_threshold_filter,
        retrieval_max_workers=retrieval_max_workers,
        qa_top_k=qa_top_k,
        qa_max_workers=qa_max_workers,
        max_qa_steps=max_qa_steps,
        max_new_tokens=max_new_tokens,
    )

    LOGGER.info("Loading the existing graph and embedding stores from %s", save_dir)
    rag = MemGraphRAG(global_config=config)
    rag.prepare_retrieval_objects()

    start_time = time.perf_counter()
    (
        solutions,
        responses,
        metadata,
        retrieval_time,
    ) = rag.rag_qa(queries=queries, gold_docs=None, gold_answers=None)
    elapsed = time.perf_counter() - start_time
    qa_time = max(0.0, elapsed - retrieval_time)
    prompt_tokens = sum(int(item.get("prompt_tokens", 0) or 0) for item in metadata)
    completion_tokens = sum(int(item.get("completion_tokens", 0) or 0) for item in metadata)

    result = {
        "summary": {
            "questions_path": os.path.abspath(questions_path),
            "save_dir": os.path.abspath(save_dir),
            "num_questions": len(samples),
            "question_type": question_type or "all",
            "retrieval_seconds": round(retrieval_time, 4),
            "qa_seconds": round(qa_time, 4),
            "total_seconds": round(elapsed, 4),
            "seconds_per_question": round(elapsed / len(samples), 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "config": {
            "llm_name": llm_name,
            "embedding_model_name": embedding_model_name,
            "passage_node_weight": passage_node_weight,
            "damping": damping,
            "skip_fact_rerank": skip_fact_rerank,
            "fact_similarity_threshold": fact_similarity_threshold,
            "use_raw_threshold_filter": use_raw_threshold_filter,
            "retrieval_max_workers": retrieval_max_workers,
            "qa_max_workers": qa_max_workers,
            "max_qa_steps": max_qa_steps,
            "max_new_tokens": max_new_tokens,
        },
        "solutions": [solution.to_dict() for solution in solutions],
        "answers": responses,
        "metadata": metadata,
        "results": [
            _result_record(sample, solution, response, item_metadata)
            for sample, solution, response, item_metadata in zip(
                samples, solutions, responses, metadata
            )
        ],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    LOGGER.info("Saved %d QA results to %s", len(samples), output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="Question dataset JSON path")
    parser.add_argument("--save-dir", required=True, help="Directory containing the built index")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--llm-name", required=True)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--question-type", default="all", help="type1/type2/... or all")
    parser.add_argument("--sample-num", type=int, default=0, help="0 means all questions")
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--linking-top-k", type=int, default=50)
    parser.add_argument("--passage-node-weight", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--rerank-dspy-file", default=DEFAULT_RERANK_DSPY_FILE)
    parser.add_argument("--skip-fact-rerank", type=str_to_bool, default=True)
    parser.add_argument("--fact-similarity-threshold", type=float, default=0.6)
    parser.add_argument("--use-raw-threshold-filter", type=str_to_bool, default=True)
    parser.add_argument("--retrieval-max-workers", type=int, default=1)
    parser.add_argument("--qa-top-k", type=int, default=10)
    parser.add_argument("--qa-max-workers", type=int, default=5)
    parser.add_argument("--max-qa-steps", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    result = test_dataset(
        questions_path=args.questions,
        save_dir=args.save_dir,
        output_path=args.output,
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_model,
        question_type=args.question_type,
        sample_num=args.sample_num,
        retrieval_top_k=args.retrieval_top_k,
        linking_top_k=args.linking_top_k,
        passage_node_weight=args.passage_node_weight,
        damping=args.damping,
        embedding_batch_size=args.embedding_batch_size,
        rerank_dspy_file_path=args.rerank_dspy_file,
        skip_fact_rerank=args.skip_fact_rerank,
        fact_similarity_threshold=args.fact_similarity_threshold,
        use_raw_threshold_filter=args.use_raw_threshold_filter,
        retrieval_max_workers=args.retrieval_max_workers,
        qa_top_k=args.qa_top_k,
        qa_max_workers=args.qa_max_workers,
        max_qa_steps=args.max_qa_steps,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
