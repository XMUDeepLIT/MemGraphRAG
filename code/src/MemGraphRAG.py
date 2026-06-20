import json
import os
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
from collections import deque
from concurrent.futures import as_completed
from pathlib import Path
from string import Template

from .Memory import ThreeLayerMemory

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import normalize_triple_entry
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig
from .utils.llm_utils import fix_broken_generated_json

from .prompts.prompt import PROMPT as MEMORY_PROMPTS

logger = logging.getLogger(__name__)

class MemGraphRAG:

    def __init__(self, global_config=None, save_dir=None, llm_model_name=None, embedding_model_name=None, llm_base_url=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific MemGraphRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed vLLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"MemGraphRAG init with config:\n  {_print_config}\n")

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            # Lazy import: vllm pulls transformers→torchvision; skip when using online OpenIE only.
            from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE

            self.openie = VLLMOfflineOpenIE(self.global_config)

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        # Entity to triples index for fast lookup
        # Key: processed entity string, Value: List of dicts with 'chunk_id', 'triple', 'passage'
        self.entity_to_triples: Dict[str, List[Dict[str, Any]]] = {}

        # openie results
        self.openie_results = None
        # memory
        self.memory = None
        self.memory_path = os.path.join(self.global_config.save_dir, "initial_memory.json")    


    def initialize_graph(self):
        """
        Initializes a graph using a GraphML file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a GraphML file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graphml_xml_file = os.path.join(
            self.working_dir, f"graph.graphml"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graphml_xml_file):
                preloaded_graph = ig.Graph.Read_GraphML(self._graphml_xml_file)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        self.openie_results = all_openie_info
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the MemGraphRAG framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunks = self.chunk_embedding_store.get_text_for_all_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        self.openie_results = all_openie_info
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunks) == len(ner_results_dict) == len(triple_results_dict)

        # prepare data_store
        chunk_ids = list(chunks.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings(list(facts.values()))

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_num_chunk = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

    def triple_extraction(self, docs: List[str]):
        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunks = self.chunk_embedding_store.get_text_for_all_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            httpx_logger = logging.getLogger("httpx")
            previous_httpx_level = httpx_logger.level
            try:
                httpx_logger.setLevel(logging.WARNING)
                new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            finally:
                httpx_logger.setLevel(previous_httpx_level)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)
        self.openie_results = all_openie_info
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)


    def build_memory(self):
        # 先从指定路径加载openie结果
        with open(self.openie_results_path, "r", encoding="utf-8") as f:
            openie_results = json.load(f)
        memory = ThreeLayerMemory()
        memory.build_from_raw_openie_results(openie_results)
        memory.save(self.memory_path)
        self.memory = memory

    @staticmethod
    def _memory_json(response: str) -> Dict[str, Any]:
        """Parse JSON returned by a memory-pipeline prompt."""
        cleaned = str(response).strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            candidate = match.group(0) if match else cleaned
            return json.loads(fix_broken_generated_json(candidate))

    @staticmethod
    def _memory_infer_result(result: Any) -> Tuple[str, Dict[str, Any]]:
        if isinstance(result, (tuple, list)):
            text = str(result[0]) if result else ""
            metadata = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
            return text, metadata
        return str(result), {}

    @staticmethod
    def _normalize_memory_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    @classmethod
    def _normalize_memory_triple(cls, triple: Any) -> Tuple[str, str, str]:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            return "", "", ""
        return tuple(cls._normalize_memory_text(x) for x in triple)

    @staticmethod
    def _memory_to_dict(memory: Union[ThreeLayerMemory, Dict[str, Any]]) -> Dict[str, Any]:
        return memory.to_dict() if isinstance(memory, ThreeLayerMemory) else json.loads(json.dumps(memory))

    @staticmethod
    def _write_json(path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def extract_memory_schema(self, memory: Union[ThreeLayerMemory, Dict[str, Any]]) -> ThreeLayerMemory:
        """Attach one ontology/schema triple to every fact using passage context."""
        data = self._memory_to_dict(memory)
        facts = data.get("fact_layer", [])
        passages = data.get("passage_layer", [])
        schemas = data.get("schema_layer", [])
        schema_to_idx = {
            tuple(map(str, row.get("content", []))): int(row["idx"])
            for row in schemas if len(row.get("content", [])) == 3
        }

        def extract_one(fact: Dict[str, Any]) -> Tuple[int, Optional[Tuple[str, str, str]]]:
            triple = fact.get("content", [])
            contexts = []
            for passage_idx in fact.get("passage_indices", [])[:3]:
                if isinstance(passage_idx, int) and 0 <= passage_idx < len(passages):
                    text = str(passages[passage_idx].get("content", "")).strip()
                    if text:
                        contexts.append(text)
            formatted = "\n".join(
                f"- ({t[0]}) [{t[1]}] ({t[2]})" for t in [triple]
                if isinstance(t, list) and len(t) == 3
            )
            prompt = MEMORY_PROMPTS["ontology_extraction"]
            user = prompt["user"].replace("${passage}", "\n\n".join(contexts) or "No passage context provided.")
            user = user.replace("${triples}", formatted)
            result = self.llm_model.infer(
                [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}],
                max_completion_tokens=self.global_config.schema_extraction_max_new_tokens,
                temperature=self.global_config.schema_extraction_temperature,
            )
            response, _ = self._memory_infer_result(result)
            rows = self._memory_json(response).get("ontology_triples", [])
            if not rows:
                return int(fact["idx"]), None
            ontology = rows[0].get("ontology", [])
            if not isinstance(ontology, list) or len(ontology) != 3:
                return int(fact["idx"]), None
            normalized = []
            for position, value in enumerate(ontology):
                value = str(value).strip() or "Unknown"
                if position != 1 and not (value.startswith("<") and value.endswith(">")):
                    value = f"<{value}>"
                normalized.append(value)
            return int(fact["idx"]), tuple(normalized)

        pending = [
            fact for fact in facts
            if self.global_config.schema_extraction_force_reextract
            or int(fact.get("schema_idx", -1)) < 0
        ]
        sample = self.global_config.schema_extraction_sample_facts
        if sample > 0:
            pending = pending[:sample]
        extracted: Dict[int, Tuple[str, str, str]] = {}
        workers = max(1, self.global_config.memory_max_workers)
        if workers == 1:
            for fact in tqdm(pending, desc="Extracting schema"):
                fact_idx, ontology = extract_one(fact)
                if ontology is not None:
                    extracted[fact_idx] = ontology
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(extract_one, fact) for fact in pending]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting schema"):
                    fact_idx, ontology = future.result()
                    if ontology is not None:
                        extracted[fact_idx] = ontology

        for fact in facts:
            ontology = extracted.get(int(fact.get("idx", -1)))
            if ontology is None:
                continue
            if ontology not in schema_to_idx:
                schema_idx = len(schemas)
                schema_to_idx[ontology] = schema_idx
                schemas.append({"idx": schema_idx, "content": list(ontology), "frequency": 0,
                                "embedding": None, "fact_indices": []})
            fact["schema_idx"] = schema_to_idx[ontology]

        by_schema: Dict[int, List[int]] = defaultdict(list)
        fact_frequency = {int(fact["idx"]): int(fact.get("frequency", 0) or 0) for fact in facts}
        for fact in facts:
            schema_idx = int(fact.get("schema_idx", -1))
            if schema_idx >= 0:
                by_schema[schema_idx].append(int(fact["idx"]))
        for schema in schemas:
            fact_indices = sorted(set(by_schema.get(int(schema["idx"]), [])))
            schema["fact_indices"] = fact_indices
            schema["frequency"] = sum(fact_frequency.get(fact_idx, 0) for fact_idx in fact_indices)
        data["schema_layer"] = schemas
        data["stats"] = {"num_schemas": len(schemas), "num_facts": len(facts), "num_passages": len(passages)}
        return ThreeLayerMemory.from_dict(data)

    def filter_memory_ontology(self, memory: Union[ThreeLayerMemory, Dict[str, Any]]) -> ThreeLayerMemory:
        """Remove low-frequency schemas and rebuild all inter-layer indices."""
        source = ThreeLayerMemory.from_dict(self._memory_to_dict(memory))
        mode = self.global_config.ontology_filter_mode
        if mode == "absolute":
            removed = {s.idx for s in source.schema_layer if s.frequency < self.global_config.ontology_filter_min_frequency}
        elif mode == "percentile":
            percent = min(100.0, max(0.0, self.global_config.ontology_filter_low_percent))
            count = int(len(source.schema_layer) * percent / 100.0)
            removed = {s.idx for s in sorted(source.schema_layer, key=lambda s: (s.frequency, s.idx))[:count]}
        else:
            raise ValueError(f"Unsupported ontology_filter_mode: {mode}")

        data = source.to_dict()
        kept_schemas = [s for s in data["schema_layer"] if int(s["idx"]) not in removed]
        schema_map = {int(row["idx"]): idx for idx, row in enumerate(kept_schemas)}
        kept_facts = [f for f in data["fact_layer"] if int(f.get("schema_idx", -1)) in schema_map]
        fact_map = {int(row["idx"]): idx for idx, row in enumerate(kept_facts)}
        for idx, row in enumerate(kept_facts):
            row["idx"] = idx
            row["schema_idx"] = schema_map[int(row["schema_idx"])]
            row["passage_indices"] = list(dict.fromkeys(row.get("passage_indices", [])))
            row["frequency"] = len(row["passage_indices"])
        for idx, row in enumerate(kept_schemas):
            old_idx = int(row["idx"])
            row["idx"] = idx
            row["fact_indices"] = [fact_map[f] for f in row.get("fact_indices", []) if f in fact_map]
            row["frequency"] = len(row["fact_indices"])
        for row in data["passage_layer"]:
            row["fact_indices"] = [fact_map[f] for f in row.get("fact_indices", []) if f in fact_map]
        result = {"schema_layer": kept_schemas, "fact_layer": kept_facts, "passage_layer": data["passage_layer"],
                  "stats": {"num_schemas": len(kept_schemas), "num_facts": len(kept_facts),
                            "num_passages": len(data["passage_layer"])}}
        return ThreeLayerMemory.from_dict(result)

    def _conflict_maps(self, memory: ThreeLayerMemory) -> Tuple[Dict[int, List[str]], Dict[int, List[str]], Dict[int, int]]:
        facts, evidence, order = {}, {}, {}
        limit = self.global_config.conflict_passage_evidence_per_fact
        for position, fact in enumerate(memory.fact_layer):
            facts[fact.idx] = list(fact.content)
            order[fact.idx] = position
            evidence[fact.idx] = [str(p.content)[:1200] for p in memory.get_passages_by_fact(fact.idx)[:limit]]
        return facts, evidence, order

    def _conflict_candidates(self, facts: Dict[int, List[str]]) -> Dict[int, List[int]]:
        by_subject_relation: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        duplicates: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        for fact_idx, triple in facts.items():
            normalized = self._normalize_memory_triple(triple)
            by_subject_relation[normalized[:2]].append(fact_idx)
            duplicates[normalized].append(fact_idx)
        duplicate_peers: Dict[int, Set[int]] = defaultdict(set)
        for ids in duplicates.values():
            for fact_idx in ids:
                duplicate_peers[fact_idx].update(set(ids) - {fact_idx})
        candidates: Dict[int, Set[int]] = {fact_idx: set() for fact_idx in facts}
        for ids in by_subject_relation.values():
            for fact_idx in ids:
                own = self._normalize_memory_triple(facts[fact_idx])
                candidates[fact_idx].update(
                    other for other in ids
                    if other != fact_idx and self._normalize_memory_triple(facts[other])[2] != own[2]
                )
        if self.global_config.conflict_enable_reverse_relation_check:
            allowed = {self._normalize_memory_text(x) for x in self.global_config.conflict_reverse_relations}
            for fact_idx, triple in facts.items():
                head, relation, tail = self._normalize_memory_triple(triple)
                if relation not in allowed:
                    continue
                for other, other_triple in facts.items():
                    other_head, other_relation, other_tail = self._normalize_memory_triple(other_triple)
                    if other != fact_idx and relation == other_relation and head == other_tail and tail == other_head:
                        candidates[fact_idx].add(other)
        limit = self.global_config.conflict_max_related_per_target
        return {key: [x for x in sorted(value) if x not in duplicate_peers[key]][:limit]
                for key, value in candidates.items()}

    def detect_memory_conflicts(self, memory: Union[ThreeLayerMemory, Dict[str, Any]]) -> Dict[str, Any]:
        """Detect hard conflicts among candidate facts."""
        memory_obj = ThreeLayerMemory.from_dict(self._memory_to_dict(memory))
        facts, evidence, order = self._conflict_maps(memory_obj)
        candidates = self._conflict_candidates(facts)

        def detect_one(target_id: int, related_ids: List[int]) -> Dict[str, Any]:
            related_lines = []
            if self.global_config.conflict_include_passage_evidence:
                related_lines.extend(f"Target evidence {i}: {text}" for i, text in enumerate(evidence[target_id], 1))
            for rank, related_id in enumerate(related_ids, 1):
                related_lines.append(f"- Triple {rank} | ID: {related_id} | triple: {facts[related_id]}")
                if self.global_config.conflict_include_passage_evidence:
                    related_lines.extend(f"  Evidence {i}: {text}" for i, text in enumerate(evidence[related_id], 1))
            prompt = MEMORY_PROMPTS["conflict_detection"]
            user = Template(prompt["user"]).substitute(
                target_triple=str(facts[target_id]), related_triples="\n".join(related_lines))
            raw = self.llm_model.infer(
                [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}],
                max_completion_tokens=self.global_config.conflict_detection_max_new_tokens,
                temperature=0.0,
            )
            response, metadata = self._memory_infer_result(raw)
            parsed = self._memory_json(response)
            hard = []
            matched_ids: Set[int] = set()
            for value in parsed.get("conflicting_triple_ids", []):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
                if value in related_ids:
                    matched_ids.add(value)
            for entry in parsed.get("conflicts", []):
                if not isinstance(entry, dict):
                    continue
                conflict_type = str(entry.get("conflict_type", "uncertain")).lower()
                confidence = float(entry.get("confidence", 0.0) or 0.0)
                if (not entry.get("is_hard_conflict") and not entry.get("needs_resolution")) \
                        or confidence < self.global_config.conflict_min_confidence \
                        or conflict_type in {"duplicate", "none", "uncertain"}:
                    continue
                other_id = None
                for field in ("triple2", "other_triple"):
                    normalized = self._normalize_memory_triple(entry.get(field))
                    for candidate_id in related_ids:
                        if normalized == self._normalize_memory_triple(facts[candidate_id]):
                            other_id = candidate_id
                            break
                if other_id is not None:
                    entry["_matched_other_id"] = other_id
                    matched_ids.add(other_id)
                hard.append(entry)
            return {"target_id": target_id, "related_ids": related_ids, "hard_conflicts": hard,
                    "conflicting_related_ids": sorted(matched_ids),
                    "token_usage": {"prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
                                    "completion_tokens": int(metadata.get("completion_tokens", 0) or 0)}}

        tasks = []
        for fact_idx, related in candidates.items():
            if self.global_config.conflict_streaming_only_previous:
                related = [other for other in related if order[other] < order[fact_idx]]
            if related:
                tasks.append((fact_idx, related))
        results = []
        workers = max(1, self.global_config.memory_max_workers)
        if workers == 1:
            results = [detect_one(*task) for task in tqdm(tasks, desc="Detecting conflicts")]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(detect_one, *task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Detecting conflicts"):
                    results.append(future.result())

        items, seen, conflicting = [], set(), set()
        for result in results:
            target_id = int(result["target_id"])
            detailed = {int(x.get("_matched_other_id")): x for x in result["hard_conflicts"]
                        if x.get("_matched_other_id") is not None}
            for other_id in result["conflicting_related_ids"]:
                pair = tuple(sorted((target_id, int(other_id))))
                if pair in seen:
                    continue
                seen.add(pair)
                entry = detailed.get(int(other_id), {})
                items.append({"target_fact_id": target_id, "other_fact_id": int(other_id),
                              "target_triple": facts[target_id], "other_triple": facts[int(other_id)],
                              "conflict_type": entry.get("conflict_type", "unspecified"),
                              "confidence": float(entry.get("confidence", 0.6) or 0.6),
                              "reason": entry.get("conflict_reason", "Returned by model."),
                              "needs_resolution": True,
                              "evidence_passages": {"target": evidence[target_id], "other": evidence[int(other_id)]}})
                conflicting.update(pair)
        prompt_tokens = sum(r["token_usage"]["prompt_tokens"] for r in results)
        completion_tokens = sum(r["token_usage"]["completion_tokens"] for r in results)
        return {"meta": {"created_at": datetime.now().isoformat(timespec="seconds"),
                         "llm_name": self.global_config.llm_name,
                         "max_workers": workers},
                "summary": {"num_facts": len(facts), "num_detection_tasks": len(tasks),
                            "num_candidate_edges": sum(map(len, candidates.values())),
                            "num_conflict_items": len(items), "num_conflicting_facts": len(conflicting),
                            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens},
                "conflict_items": items}

    def _apply_memory_actions(self, memory: ThreeLayerMemory, actions: Dict[int, Dict[str, Any]]) -> ThreeLayerMemory:
        """Apply conflict actions and rebuild every inter-layer index."""
        source = memory.to_dict()
        facts = []
        old_to_new: Dict[int, int] = {}
        for fact in source["fact_layer"]:
            old_idx = int(fact["idx"])
            action = actions.get(old_idx, {})
            if action.get("action") == "discarded":
                continue
            new_fact = dict(fact)
            resolved = action.get("resolved_triple")
            if action.get("action") == "modified" and isinstance(resolved, (list, tuple)) and len(resolved) == 3:
                new_fact["content"] = list(map(str, resolved))
                if self.global_config.conflict_resolution_set_schema_negative_on_modified:
                    new_fact["schema_idx"] = -1
            new_idx = len(facts)
            old_to_new[old_idx] = new_idx
            new_fact["idx"] = new_idx
            new_fact["passage_indices"] = sorted({int(x) for x in new_fact.get("passage_indices", [])
                                                  if 0 <= int(x) < len(source["passage_layer"])})
            new_fact["frequency"] = len(new_fact["passage_indices"])
            facts.append(new_fact)
        schemas = [dict(row) for row in source["schema_layer"]]
        by_schema: Dict[int, List[int]] = defaultdict(list)
        for fact in facts:
            schema_idx = int(fact.get("schema_idx", -1))
            if 0 <= schema_idx < len(schemas):
                by_schema[schema_idx].append(int(fact["idx"]))
        for schema in schemas:
            schema_idx = int(schema["idx"])
            schema["fact_indices"] = sorted(by_schema.get(schema_idx, []))
            schema["frequency"] = len(schema["fact_indices"])
        passages = []
        for passage in source["passage_layer"]:
            row = dict(passage)
            row["fact_indices"] = sorted({old_to_new[x] for x in passage.get("fact_indices", []) if x in old_to_new})
            passages.append(row)
        data = {"schema_layer": schemas, "fact_layer": facts, "passage_layer": passages,
                "stats": {"num_schemas": len(schemas), "num_facts": len(facts), "num_passages": len(passages)}}
        return ThreeLayerMemory.from_dict(data)

    def resolve_memory_conflicts(
        self, memory: Union[ThreeLayerMemory, Dict[str, Any]], conflict_data: Dict[str, Any]
    ) -> Tuple[ThreeLayerMemory, Dict[str, Any]]:
        """Resolve connected conflict components and return rebuilt memory plus audit data."""
        memory_obj = ThreeLayerMemory.from_dict(self._memory_to_dict(memory))
        facts, evidence, _ = self._conflict_maps(memory_obj)
        edges = []
        for item in conflict_data.get("conflict_items", []):
            try:
                left, right = int(item["target_fact_id"]), int(item["other_fact_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if item.get("needs_resolution") and left in facts and right in facts and left != right:
                edges.append((left, right, item))
        adjacency: Dict[int, Set[int]] = defaultdict(set)
        for left, right, _ in edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        components, visited = [], set()
        for start in adjacency:
            if start in visited:
                continue
            queue, component = deque([start]), []
            visited.add(start)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(sorted(component))

        def resolve_component(component: List[int]) -> Dict[str, Any]:
            if len(component) > self.global_config.conflict_resolution_max_component_size:
                return {"component_fact_ids": component, "resolved_triples": [],
                        "unresolved_conflicts": [{"triple_ids": component, "reason": "component too large"}],
                        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0}}
            rows = []
            component_set = set(component)
            for left, right, item in edges:
                if left in component_set and right in component_set:
                    rows.append({"triple_id_1": left, "triple_1": facts[left],
                                 "triple_id_2": right, "triple_2": facts[right],
                                 "detected_conflict_type": item.get("conflict_type", "unspecified"),
                                 "detected_reason": item.get("reason", ""),
                                 "evidence_1": evidence[left], "evidence_2": evidence[right]})
            prompt = MEMORY_PROMPTS["conflict_resolution"]
            payloads = [(prompt["system"], rows)]
            if self.global_config.conflict_resolution_enable_content_filter_fallback:
                minimal = [{key: value for key, value in row.items()
                            if not key.startswith("evidence") and key != "detected_reason"} for row in rows]
                payloads.append(("Resolve knowledge graph conflicts conservatively. Output JSON only.", minimal))
            last_error = None
            for attempt, (system, payload) in enumerate(payloads, 1):
                try:
                    user = Template(prompt["user"]).substitute(
                        conflicting_triples_with_sources=json.dumps(
                            {"component_fact_ids": component, "conflicts": payload}, ensure_ascii=False, indent=2))
                    raw = self.llm_model.infer(
                        [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        max_completion_tokens=self.global_config.conflict_resolution_max_new_tokens,
                        temperature=0.0,
                    )
                    response, metadata = self._memory_infer_result(raw)
                    parsed = self._memory_json(response)
                    parsed["component_fact_ids"] = component
                    parsed["fallback_attempt"] = attempt
                    parsed["token_usage"] = {"prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
                                             "completion_tokens": int(metadata.get("completion_tokens", 0) or 0)}
                    return parsed
                except Exception as exc:
                    last_error = exc
                    message = str(exc).lower()
                    if "content_filter" not in message and "content management policy" not in message:
                        raise
            return {"component_fact_ids": component, "resolved_triples": [],
                    "unresolved_conflicts": [{"triple_ids": component, "reason": str(last_error)}],
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0}}

        component_results = [resolve_component(component) for component in tqdm(components, desc="Resolving conflicts")]
        rank = {"kept": 1, "modified": 2, "discarded": 3}
        actions: Dict[int, Dict[str, Any]] = {}
        by_normalized = defaultdict(list)
        for fact_idx, triple in facts.items():
            by_normalized[self._normalize_memory_triple(triple)].append(fact_idx)
        for result in component_results:
            for row in result.get("resolved_triples", []):
                action = str(row.get("resolution", "")).lower()
                if action not in rank:
                    continue
                fact_idx = None
                try:
                    candidate = int(row.get("triple_id"))
                    fact_idx = candidate if candidate in facts else None
                except (TypeError, ValueError):
                    pass
                if fact_idx is None:
                    matches = by_normalized[self._normalize_memory_triple(row.get("original_triple"))]
                    fact_idx = matches[0] if len(matches) == 1 else None
                if fact_idx is None:
                    continue
                resolved = row.get("resolved_triple")
                info = {"action": action,
                        "resolved_triple": list(map(str, resolved)) if isinstance(resolved, (list, tuple)) and len(resolved) == 3 else None,
                        "reason": str(row.get("reason", "")), "conflict_type": row.get("conflict_type", "unspecified")}
                if fact_idx not in actions or rank[action] > rank[actions[fact_idx]["action"]]:
                    actions[fact_idx] = info
        updated = self._apply_memory_actions(memory_obj, actions)
        counts = {name: sum(1 for item in actions.values() if item["action"] == name) for name in rank}
        prompt_tokens = sum(x.get("token_usage", {}).get("prompt_tokens", 0) for x in component_results)
        completion_tokens = sum(x.get("token_usage", {}).get("completion_tokens", 0) for x in component_results)
        audit = {"meta": {"created_at": datetime.now().isoformat(timespec="seconds"),
                          "llm_name": self.global_config.llm_name,
                          "max_component_size": self.global_config.conflict_resolution_max_component_size},
                 "summary": {"num_conflict_items_input": len(conflict_data.get("conflict_items", [])),
                             "num_resolvable_edges": len(edges), "num_components": len(components),
                             "actions_applied": counts, "facts_before": len(facts),
                             "facts_after": len(updated.fact_layer), "prompt_tokens": prompt_tokens,
                             "completion_tokens": completion_tokens,
                             "total_tokens": prompt_tokens + completion_tokens},
                 "component_resolutions": component_results,
                 "applied_actions": [{"fact_id": fact_idx, **info} for fact_idx, info in sorted(actions.items())]}
        return updated, audit

    def build_memory_graph(self, memory: Union[ThreeLayerMemory, Dict[str, Any]]) -> Dict[str, Any]:
        """Build the final type/entity/passage graph from conflict-resolved memory."""
        data = self._memory_to_dict(memory)
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        def add_node(node_id: str, **attrs: Any) -> None:
            nodes.setdefault(node_id, {"id": node_id, **attrs})

        def add_edge(source: str, target: str, edge_type: str, weight: float = 1.0,
                     accumulate: bool = False, **attrs: Any) -> None:
            if source == target:
                return
            key = (source, target, edge_type)
            if key not in edges:
                edges[key] = {"source": source, "target": target, "type": edge_type,
                              "weight": float(weight), **attrs}
            elif accumulate:
                edges[key]["weight"] += float(weight)

        schemas = {int(row["idx"]): row for row in data.get("schema_layer", [])}
        passages = {int(row["idx"]): row for row in data.get("passage_layer", [])}
        final_entities: Set[str] = set()
        final_facts = []
        entity_types: Dict[str, Set[str]] = defaultdict(set)

        # Keep every indexed passage node, including chunks without a surviving fact.
        for chunk_id, row in self.chunk_embedding_store.get_text_for_all_rows().items():
            content = str(row.get("content", ""))
            add_node(chunk_id, layer="passage", content=content, label=content[:120])

        for fact in data.get("fact_layer", []):
            triple = fact.get("content", [])
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                continue
            head, _, tail = map(str, triple)
            final_entities.update((head, tail))
            final_facts.append(str((head, str(triple[1]), tail)))
            head_id = compute_mdhash_id(head, prefix="entity-")
            tail_id = compute_mdhash_id(tail, prefix="entity-")
            add_node(head_id, layer="entity", content=head, label=head)
            add_node(tail_id, layer="entity", content=tail, label=tail)
            # Preserve HippoRAG entity-to-entity weighting: one count per surviving fact.
            add_edge(head_id, tail_id, "entity_relation", accumulate=True)
            add_edge(tail_id, head_id, "entity_relation", accumulate=True)

            for passage_idx in fact.get("passage_indices", []):
                passage = passages.get(int(passage_idx))
                if passage is None:
                    continue
                passage_id = str(passage.get("chunk_id", ""))
                if passage_id in nodes:
                    add_edge(passage_id, head_id, "passage_entity")
                    add_edge(passage_id, tail_id, "passage_entity")

            schema = schemas.get(int(fact.get("schema_idx", -1)))
            if schema is not None and len(schema.get("content", [])) == 3:
                head_type, _, tail_type = map(str, schema["content"])
                entity_types[head_type].add(head)
                entity_types[tail_type].add(tail)

        self.entity_embedding_store.insert_strings(sorted(final_entities))
        self.fact_embedding_store.insert_strings(final_facts)

        # Type nodes and entity-type edges. Each type distributes total weight 1 over its entities.
        for type_name, entities in entity_types.items():
            type_id = compute_mdhash_id(type_name, prefix="type-")
            add_node(type_id, layer="type", content=type_name, label=type_name,
                     entity_count=len(entities))
            weight = 1.0 / len(entities) if entities else 0.0
            for entity in entities:
                entity_id = compute_mdhash_id(entity, prefix="entity-")
                add_edge(entity_id, type_id, "entity_to_type", weight=weight)
                if self.global_config.is_directed_graph:
                    add_edge(type_id, entity_id, "type_to_entity", weight=weight)

        # Type-to-type weight is the sum of frequencies of schemas joining the same type pair.
        for schema in data.get("schema_layer", []):
            content = schema.get("content", [])
            frequency = float(schema.get("frequency", 0) or 0)
            if len(content) != 3 or frequency <= 0:
                continue
            head_type, _, tail_type = map(str, content)
            head_type_id = compute_mdhash_id(head_type, prefix="type-")
            tail_type_id = compute_mdhash_id(tail_type, prefix="type-")
            if head_type_id in nodes and tail_type_id in nodes:
                add_edge(head_type_id, tail_type_id, "type_relation", frequency, accumulate=True)
                if self.global_config.is_directed_graph:
                    add_edge(tail_type_id, head_type_id, "type_relation", frequency, accumulate=True)

        # Add synonymy edges only now, using entities that survived conflict resolution.
        entity_ids = [compute_mdhash_id(entity, prefix="entity-") for entity in sorted(final_entities)]
        if len(entity_ids) > 1:
            embeddings = self.entity_embedding_store.get_embeddings(entity_ids)
            neighbors = retrieve_knn(
                query_ids=entity_ids, key_ids=entity_ids,
                query_vecs=embeddings, key_vecs=embeddings,
                k=min(len(entity_ids), self.global_config.synonymy_edge_topk + 1),
                query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                key_batch_size=self.global_config.synonymy_edge_key_batch_size,
            )
            for source_id, (neighbor_ids, scores) in neighbors.items():
                entity = nodes[source_id]["content"]
                if len(re.sub("[^A-Za-z0-9]", "", entity)) <= 2:
                    continue
                for target_id, score in zip(neighbor_ids, scores):
                    if source_id == target_id or float(score) < self.global_config.synonymy_edge_sim_threshold:
                        continue
                    add_edge(source_id, target_id, "entity_similarity",
                             weight=float(score), similarity=float(score))

        stats = {
            "num_type_nodes": sum(n.get("layer") == "type" for n in nodes.values()),
            "num_passage_nodes": sum(n.get("layer") == "passage" for n in nodes.values()),
            "num_entity_nodes": sum(n.get("layer") == "entity" for n in nodes.values()),
            "num_edges": len(edges),
            "num_entity_relation_edges": sum(e["type"] == "entity_relation" for e in edges.values()),
            "num_passage_entity_edges": sum(e["type"] == "passage_entity" for e in edges.values()),
            "num_entity_type_edges": sum(e["type"] in {"entity_to_type", "type_to_entity"} for e in edges.values()),
            "num_type_relation_edges": sum(e["type"] == "type_relation" for e in edges.values()),
            "num_entity_similarity_edges": sum(e["type"] == "entity_similarity" for e in edges.values()),
        }
        return {"nodes": list(nodes.values()), "edges": list(edges.values()), "stats": stats}

    def install_memory_graph(self, graph: Dict[str, Any]) -> None:
        """Install the final three-node-type graph as the graph used by retrieval."""
        result = ig.Graph(directed=self.global_config.is_directed_graph)
        node_ids = [node["id"] for node in graph["nodes"]]
        result.add_vertices(len(node_ids))
        result.vs["name"] = node_ids
        result.vs["layer"] = [node.get("layer", "") for node in graph["nodes"]]
        result.vs["content"] = [str(node.get("content", "")) for node in graph["nodes"]]
        valid = set(node_ids)
        graph_edges = [edge for edge in graph["edges"]
                       if edge["source"] in valid and edge["target"] in valid]
        result.add_edges([(edge["source"], edge["target"]) for edge in graph_edges])
        result.es["weight"] = [float(edge.get("weight", 1.0)) for edge in graph_edges]
        result.es["type"] = [edge.get("type", "") for edge in graph_edges]
        self.graph = result
        self.node_to_node_stats = {
            (edge["source"], edge["target"]): float(edge.get("weight", 1.0))
            for edge in graph_edges
        }
        entity_passages: Dict[str, Set[str]] = defaultdict(set)
        for edge in graph_edges:
            if edge.get("type") != "passage_entity":
                continue
            entity_passages[edge["target"]].add(edge["source"])
        self.ent_node_to_num_chunk = {
            node["id"]: len(entity_passages.get(node["id"], set()))
            for node in graph["nodes"] if node.get("layer") == "entity"
        }
        self.save_igraph()

    def save_memory_graph(self, graph: Dict[str, Any]) -> Dict[str, Optional[str]]:
        output_dir = os.path.join(self.global_config.save_dir, "graph_from_memory")
        stem = self.global_config.memory_graph_output_name
        json_path = os.path.join(output_dir, f"{stem}.json")
        graphml_path = os.path.join(output_dir, f"{stem}.graphml")
        self._write_json(json_path, graph)
        os.makedirs(output_dir, exist_ok=True)
        self.graph.write_graphml(graphml_path)
        return {"json": json_path, "graphml": graphml_path}

    def index_with_memory(self, docs: List[str]) -> Dict[str, Any]:
        """Run OpenIE, memory construction, conflict resolution, and final graph construction."""
        if not docs:
            raise ValueError("index_with_memory requires at least one document")
        missing_prompts = {"ontology_extraction", "conflict_detection", "conflict_resolution"} - set(MEMORY_PROMPTS)
        if missing_prompts:
            raise RuntimeError(f"Missing memory prompts: {sorted(missing_prompts)}")
        detailed = self.global_config.memory_artifact_mode == "detailed"
        if self.global_config.memory_artifact_mode not in {"default", "detailed"}:
            raise ValueError("memory_artifact_mode must be 'default' or 'detailed'")
        artifacts: Dict[str, Any] = {"openie": self.openie_results_path}

        stage = "OpenIE"
        try:
            # OpenIE is intentionally retained in both artifact modes.
            self.triple_extraction(docs)
            openie_payload = {"docs": self.openie_results or []}
            initial = ThreeLayerMemory()
            initial.build_from_raw_openie_results(openie_payload)
            if detailed:
                path = os.path.join(self.global_config.save_dir, "initial_memory.json")
                initial.save(path)
                artifacts["initial_memory"] = path
                backup_path = os.path.join(self.global_config.save_dir, "backup", "initial_memory.backup.json")
                self._write_json(backup_path, initial.to_dict())
                artifacts["initial_memory_backup"] = backup_path

            stage = "schema extraction"
            with_schema = self.extract_memory_schema(initial)
            schema_path = os.path.join(self.global_config.save_dir, "initial_memory_with_schema.json")
            with_schema.save(schema_path)
            artifacts["schema_memory"] = schema_path

            stage = "ontology filtering"
            filtered = self.filter_memory_ontology(with_schema)
            if detailed:
                path = os.path.join(self.global_config.save_dir, "memory_after_ontology_filtering.json")
                filtered.save(path)
                artifacts["filtered_memory"] = path

            stage = "conflict detection"
            conflicts = self.detect_memory_conflicts(filtered)
            if detailed:
                path = os.path.join(self.global_config.save_dir, "conflict_detection_results.json")
                self._write_json(path, conflicts)
                artifacts["conflict_detection"] = path

            stage = "conflict resolution"
            final_memory, resolution = self.resolve_memory_conflicts(filtered, conflicts)
            if detailed:
                path = os.path.join(self.global_config.save_dir, "conflict_resolution_results.json")
                self._write_json(path, resolution)
                artifacts["conflict_resolution"] = path

            final_path = os.path.join(self.global_config.save_dir, "memory.json")
            final_memory.save(final_path)
            artifacts["memory"] = final_path
            self.memory = final_memory
            self.memory_path = final_path

            stage = "memory graph construction"
            memory_graph = self.build_memory_graph(final_memory)
            self.install_memory_graph(memory_graph)
            artifacts["memory_graph"] = self.save_memory_graph(memory_graph)
            artifacts["retrieval_graph"] = self._graphml_xml_file
            return {"memory": final_memory, "graph": memory_graph, "stats": memory_graph["stats"],
                    "conflict_summary": conflicts["summary"],
                    "resolution_summary": resolution["summary"], "artifacts": artifacts}
        except Exception as exc:
            raise RuntimeError(f"index_with_memory failed during {stage}: {exc}") from exc



    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the MemGraphRAG framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        # 统计计数器（线程安全）
        import threading
        fact_stats = {'no_facts': 0, 'has_facts': 0, 'total_facts_before': 0, 'total_facts_after': 0}
        fact_stats_lock = threading.Lock()

        def retrieve_single_query(query):
            """检索单个查询的辅助函数"""
            # 始终使用归一化分数用于后续的图搜索
            query_fact_scores = self.get_fact_scores(query, normalize=True)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)

            # 统计三元组过滤情况
            facts_before = rerank_log.get('facts_before_rerank', [])
            if isinstance(facts_before, int):
                facts_before = facts_before  # 已经是总数
            else:
                facts_before = len(facts_before)
            facts_after = len(rerank_log.get('facts_after_rerank', []))

            if len(top_k_facts) == 0:
                with fact_stats_lock:
                    fact_stats['no_facts'] += 1
                    fact_stats['total_facts_before'] += facts_before
                    fact_stats['total_facts_after'] += facts_after
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                with fact_stats_lock:
                    fact_stats['has_facts'] += 1
                    fact_stats['total_facts_before'] += facts_before
                    fact_stats['total_facts_after'] += facts_after
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]
            return QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve])

        # 并行或串行执行检索
        retrieval_max_workers = self.global_config.retrieval_max_workers
        if retrieval_max_workers > 1:
            # 并行执行
            from concurrent.futures import ThreadPoolExecutor, as_completed
            retrieval_results = [None] * len(queries)
            with ThreadPoolExecutor(max_workers=retrieval_max_workers) as executor:
                future_to_idx = {executor.submit(retrieve_single_query, query): idx 
                                for idx, query in enumerate(queries)}
                for future in tqdm(as_completed(future_to_idx), total=len(queries), desc="Retrieving (parallel)"):
                    idx = future_to_idx[future]
                    retrieval_results[idx] = future.result()
        else:
            # 串行执行
            retrieval_results = []
            for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
                retrieval_results.append(retrieve_single_query(query))

        # 输出 fact 统计日志
        total_queries = fact_stats['no_facts'] + fact_stats['has_facts']
        total_facts_before = fact_stats['total_facts_before']
        total_facts_after = fact_stats['total_facts_after']
        total_facts_filtered = total_facts_before - total_facts_after
        filter_rate = (total_facts_filtered / total_facts_before * 100) if total_facts_before > 0 else 0
        logger.info(f"Fact reranking stats: total_queries={total_queries}, "
                   f"has_facts={fact_stats['has_facts']} ({fact_stats['has_facts']/total_queries*100:.1f}%), "
                   f"no_facts={fact_stats['no_facts']} ({fact_stats['no_facts']/total_queries*100:.1f}%)")
        logger.info(f"Fact filtering stats: total_before={total_facts_before}, total_after={total_facts_after}, "
                   f"filtered={total_facts_filtered} ({filter_rate:.1f}%), "
                   f"avg_per_query_before={total_facts_before/total_queries:.1f}, avg_per_query_after={total_facts_after/total_queries:.1f}")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the MemGraphRAG framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None
        retrieval_time = 0
        
        if not isinstance(queries[0], QuerySolution):
            retrieval_start = time.time()
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)
            retrieval_time = time.time() - retrieval_start
            logger.info(f"Retrieval time: {retrieval_time:.2f}s")

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results, retrieval_time
        else:
            return queries_solutions, all_response_message, all_metadata, retrieval_time

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        # 保存前5个问题的 prompt
        sample_prompts_path = os.path.join(self.working_dir, "sample_qa_prompts.json")
        sample_prompts = []
        for i, qa_messages in enumerate(all_qa_messages[:5]):
            sample_prompts.append({
                "index": i,
                "question": queries[i].question,
                "messages": qa_messages
            })
        with open(sample_prompts_path, 'w', encoding='utf-8') as f:
            json.dump(sample_prompts, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(sample_prompts)} sample QA prompts to {sample_prompts_path}")

        # 并行或串行调用 LLM 进行 QA
        qa_max_workers = self.global_config.qa_max_workers
        if qa_max_workers > 1:
            # 并行执行
            from concurrent.futures import ThreadPoolExecutor, as_completed
            all_qa_results = [None] * len(all_qa_messages)
            with ThreadPoolExecutor(max_workers=qa_max_workers) as executor:
                future_to_idx = {executor.submit(self.llm_model.infer, qa_messages): idx 
                                for idx, qa_messages in enumerate(all_qa_messages)}
                for future in tqdm(as_completed(future_to_idx), total=len(all_qa_messages), desc="QA Reading (parallel)"):
                    idx = future_to_idx[future]
                    all_qa_results[idx] = future.result()
        else:
            # 串行执行
            all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_num_chunk[node] = self.ent_node_to_num_chunk.get(node, 0) + 1

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_text_for_all_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            triple_records = []
            seen_fact_ids = set()
            for triple in triple_results_dict[chunk_key].triples:
                record = normalize_triple_entry(triple)
                if record is None:
                    continue
                fact_id = record['fact_id']
                if fact_id in seen_fact_ids:
                    continue
                seen_fact_ids.add(fact_id)
                triple_records.append({
                    'fact_id': fact_id,
                    'processed_triple': record['triple'],
                    'triple_str': record['triple_str'],
                    'raw_triple': triple
                })

            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_records}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            openie_dict = {'docs': all_openie_info, 'avg_ent_chars': round(sum_phrase_chars / num_phrases, 4),
                           'avg_ent_words': round(sum_phrase_words / num_phrases, 4)}
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_nodes = self.entity_embedding_store.get_text_for_all_rows()
        passage_nodes = self.chunk_embedding_store.get_text_for_all_rows()

        nodes = entity_nodes
        nodes.update(passage_nodes)

        new_nodes = {}
        for node_id, node in nodes.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_graphml(self._graphml_xml_file)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def get_triples_by_entity(self, entity: str) -> List[Dict[str, Any]]:
        """
        Retrieves all triples containing the specified entity and their corresponding chunk IDs.
        
        This method uses the pre-built entity-to-triples index for fast O(1) lookup. If the index
        has not been built yet, it will be built automatically on first call.

        Parameters:
            entity (str): The entity string to search for. Can be in any format as it will
                be processed using text_processing() to match the stored format.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                - 'chunk_id' (str): The hash ID of the chunk containing the triple
                - 'triple' (List[str]): The triple as [subject, predicate, object]
                - 'fact_id' (str): Stable identifier of the triple
                - 'passage' (str, optional): The original passage text if available

        Example:
            >>> results = MemGraphRAG.get_triples_by_entity("barack obama")
            >>> # Returns:
            >>> # [
            >>> #     {
            >>> #         'chunk_id': 'chunk-abc123...',
            >>> #         'triple': ['barack obama', 'was born in', 'hawaii'],
            >>> #         'passage': 'Barack Obama was born in Hawaii...'
            >>> #     },
            >>> #     ...
            >>> # ]
        """
        # Build index if not already built
        if not self.entity_to_triples:
            self.build_entity_to_triples_index()
        
        # Process the input entity to match the stored format
        processed_entity = text_processing(entity)
        
        # Fast lookup from index
        results = self.entity_to_triples.get(processed_entity, [])
        
        logger.info(f"Found {len(results)} triples containing entity '{entity}'")
        return results

    def build_entity_to_triples_index(self):
        """
        Builds an index dictionary mapping entities to their associated triples and passages.
        This index allows for fast O(1) lookup of all triples containing a specific entity.
        
        The index is stored in self.entity_to_triples, where:
        - Key: processed entity string (normalized using text_processing)
        - Value: List of dictionaries, each containing:
            - 'chunk_id': The hash ID of the chunk containing the triple
            - 'triple': The triple as [subject, predicate, object]
            - 'passage': The original passage text
        
        This method should be called after indexing is complete or when OpenIE results are loaded.
        """
        logger.info("Building entity to triples index for fast lookup...")
        
        # Clear existing index
        self.entity_to_triples = {}
        
        # Load OpenIE results
        if not os.path.isfile(self.openie_results_path):
            logger.warning(f"OpenIE results file not found: {self.openie_results_path}")
            return
        
        with open(self.openie_results_path, 'r') as f:
            openie_results = json.load(f)
        
        all_openie_info = openie_results.get('docs', [])
        if not all_openie_info:
            logger.warning("No OpenIE results found in the file")
            return
        
        # Standardize indices (same as in load_openie_results)
        renamed_openie_info = []
        for openie_info in all_openie_info:
            openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
            renamed_openie_info.append(openie_info)
        
        all_openie_info = renamed_openie_info
        
        # Build index by iterating through all triples
        total_triples = 0
        for chunk_info in all_openie_info:
            chunk_id = chunk_info['idx']
            passage = chunk_info.get('passage', '')
            triple_entries = chunk_info.get('extracted_triples', [])

            triple_records = []
            seen_triples = set()
            for entry in triple_entries:
                record = normalize_triple_entry(entry)
                if record is None:
                    continue
                triple_tuple = tuple(record['triple'])
                if triple_tuple in seen_triples:
                    continue
                seen_triples.add(triple_tuple)
                triple_records.append(record)

            for record in triple_records:
                triple = record['triple']
                fact_id = record['fact_id']
                processed_subject = triple[0]
                processed_object = triple[2]

                triple_info = {
                    'chunk_id': chunk_id,
                    'triple': triple,
                    'passage': passage,
                    'fact_id': fact_id
                }

                total_triples += 1

                if processed_subject:
                    if processed_subject not in self.entity_to_triples:
                        self.entity_to_triples[processed_subject] = []
                    self.entity_to_triples[processed_subject].append(triple_info)

                if processed_object and processed_object != processed_subject:
                    if processed_object not in self.entity_to_triples:
                        self.entity_to_triples[processed_object] = []
                    self.entity_to_triples[processed_object].append(triple_info)
        
        logger.info(f"Entity index built: {len(self.entity_to_triples)} unique entities, {total_triples} total triples indexed")

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        graph_node_names = set(self.graph.vs["name"])
        self.entity_node_keys: List = [key for key in self.entity_embedding_store.get_all_ids()
                                       if key in graph_node_names]
        self.passage_node_keys: List = [key for key in self.chunk_embedding_store.get_all_ids()
                                        if key in graph_node_names]
        self.type_node_keys: List = [node["name"] for node in self.graph.vs
                                     if node.attributes().get("layer") == "type"]
        all_fact_node_keys = set(self.fact_embedding_store.get_all_ids())
        final_memory_path = os.path.join(self.global_config.save_dir, "memory.json")
        if os.path.isfile(final_memory_path):
            final_memory = ThreeLayerMemory.load(final_memory_path)
            final_fact_keys = [
                compute_mdhash_id(str(tuple(map(str, fact.content))), prefix="fact-")
                for fact in final_memory.fact_layer
            ]
            self.fact_node_keys = [key for key in final_fact_keys if key in all_fact_node_keys]
        else:
            self.fact_node_keys = list(all_fact_node_keys)

        assert len(self.entity_node_keys) + len(self.passage_node_keys) + len(self.type_node_keys) == self.graph.vcount()

        igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
        self.node_name_to_vertex_idx = igraph_name_to_idx
        self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
        self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index

        entity_keys = set(self.entity_node_keys)
        passage_keys = set(self.passage_node_keys)
        entity_passages: Dict[str, Set[str]] = defaultdict(set)
        self.node_to_node_stats = {}
        for edge in self.graph.es:
            source = self.graph.vs[edge.source]["name"]
            target = self.graph.vs[edge.target]["name"]
            weight = float(edge["weight"]) if "weight" in edge.attributes() else 1.0
            self.node_to_node_stats[(source, target)] = weight
            if source in passage_keys and target in entity_keys:
                entity_passages[target].add(source)
            elif target in passage_keys and source in entity_keys:
                entity_passages[source].add(target)
        self.ent_node_to_num_chunk = {
            entity: len(entity_passages.get(entity, set())) for entity in entity_keys
        }

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        # Build entity to triples index for fast lookup
        self.build_entity_to_triples_index()

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Retrieves and computes similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.
        normalize : bool
            Whether to apply min-max normalization to the scores. Default is True.

        Returns:
        numpy.ndarray
            An array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
        query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
        
        if normalize:
            query_fact_scores = min_max_normalize(query_fact_scores)

        return query_fact_scores

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """
        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    phrase_weights[phrase_id] = fact_score

                    if self.ent_node_to_num_chunk[phrase_key] != 0:
                        phrase_weights[phrase_id] /= self.ent_node_to_num_chunk[phrase_key]

                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        #Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        #Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores


    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k
        
        # 如果 use_raw_threshold_filter=True，跳过 top-k 选择，直接对所有 facts 进行阈值过滤
        if self.global_config.skip_fact_rerank and self.global_config.use_raw_threshold_filter:
            threshold = self.global_config.fact_similarity_threshold
            
            # 获取原始分数用于阈值过滤
            raw_fact_scores = self.get_fact_scores(query, normalize=False)
            
            # 对所有 facts 进行阈值过滤
            all_fact_indices = list(range(len(query_fact_scores)))
            all_fact_ids = self.fact_node_keys
            fact_row_dict = self.fact_embedding_store.get_rows(all_fact_ids)
            
            filtered_indices = []  # 存储归一化后的索引
            filtered_facts = []
            candidate_scores = []
            
            for idx in all_fact_indices:
                raw_score = raw_fact_scores[idx]
                candidate_scores.append(raw_score)
                if raw_score >= threshold:
                    filtered_indices.append(idx)
                    fact_content = fact_row_dict[self.fact_node_keys[idx]]['content']
                    filtered_facts.append(eval(fact_content))
            
            # 打印调试信息
            # logger.info(f"Fact filtering (raw scores, skip top-k): threshold={threshold}, "
            #            f"total={len(all_fact_indices)}, passed={len(filtered_indices)}, "
            #            f"max_score={max(candidate_scores):.4f}, min_score={min(candidate_scores):.4f}")
            
            top_k_fact_indices = filtered_indices
            top_k_facts = filtered_facts
            rerank_log = {
                'facts_before_rerank': len(all_fact_indices),
                'facts_after_rerank': top_k_facts,
                'rerank_skipped': True,
                'use_raw_threshold_filter': True,
                'similarity_threshold': threshold,
                'filtered_count': len(top_k_facts),
                'max_candidate_score': float(max(candidate_scores)),
                'min_candidate_score': float(min(candidate_scores))
            }
            return top_k_fact_indices, top_k_facts, rerank_log

        # 原始逻辑：先选 top-k，再过滤
        candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][
                                 ::-1].tolist()  # list of ranked link_top_k fact relative indices
        real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in
                                   candidate_fact_indices]  # list of ranked link_top_k fact keys
        fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
        candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]  # list of link_top_k facts (each fact is a relation triple in tuple data type)

        # 如果 skip_fact_rerank 为 True，根据相似度阈值过滤，而不是使用LLM rerank
        if self.global_config.skip_fact_rerank:
            threshold = self.global_config.fact_similarity_threshold
            
            # 获取原始分数（不归一化）用于阈值过滤
            raw_fact_scores = self.get_fact_scores(query, normalize=False)
            
            # 过滤出相似度 >= 阈值的三元组（使用原始分数）
            filtered_indices = []
            filtered_facts = []
            candidate_scores = []  # 记录候选三元组的原始分数用于调试
            for idx, fact in zip(candidate_fact_indices, candidate_facts):
                raw_score = raw_fact_scores[idx]
                candidate_scores.append(raw_score)
                if raw_score >= threshold:
                    filtered_indices.append(idx)
                    filtered_facts.append(fact)
            
            # 打印调试信息
            # logger.info(f"Fact filtering (raw scores): threshold={threshold}, candidates={len(candidate_facts)}, "
            #            f"passed={len(filtered_facts)}, max_score={max(candidate_scores):.4f}, "
            #            f"min_score={min(candidate_scores):.4f}")
            
            top_k_fact_indices = filtered_indices
            top_k_facts = filtered_facts
            rerank_log = {
                'facts_before_rerank': candidate_facts, 
                'facts_after_rerank': top_k_facts, 
                'rerank_skipped': True,
                'similarity_threshold': threshold,
                'filtered_count': len(top_k_facts),
                'max_candidate_score': float(max(candidate_scores)),
                'min_candidate_score': float(min(candidate_scores))
            }
            return top_k_fact_indices, top_k_facts, rerank_log

        top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                             candidate_facts,
                                                                             candidate_fact_indices,
                                                                             len_after_rerank=link_top_k)

        rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}

        return top_k_fact_indices, top_k_facts, rerank_log
    
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
