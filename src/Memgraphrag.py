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

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
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
from .memory import ThreeLayerMemory
from .schema_fact_extract import RelationExtractor, TripleWithSchema, ChunkWithEntities
from .resolve_conflict import ConflictResolver
from .llm_client import LLMClient, LLMConfig
import threading
import queue

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

        # Three-layer Memory structure for streaming pipeline
        self.memory: ThreeLayerMemory = ThreeLayerMemory()

        # Conflict resolution settings
        self.conflict_resolution_threshold: int = 1  # schema频次>1时触发冲突检测


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

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the MemGraphRAG 2 framework which generates an OpenIE knowledge graph
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

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the MemGraphRAG 2 framework, which consists of several steps:
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
        Performs retrieval-augmented generation enhanced QA using the MemGraphRAG 2 framework.

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

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        assert len(self.entity_node_keys) + len(self.passage_node_keys) == self.graph.vcount()

        igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
        self.node_name_to_vertex_idx = igraph_name_to_idx
        self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
        self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index

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

    def construct_graph(
        self,
        docs: List[str],
        chunk_size: int = 512,
        max_workers: int = 4,
        conflict_max_workers: int = 8
    ):
        """
        Parallel streaming pipeline for entity extraction, relation extraction, and conflict resolution.

        Processing flow:
        1. Chunk documents and extract entities in parallel
        2. For each chunk, call schema_fact_extract to extract relations and schemas in parallel
        3. Store results in ThreeLayerMemory
        4. When schema frequency > 1, perform conflict detection and resolution in parallel
        5. Write conflict resolution results to Memory

        Parameters:
            docs: List of documents
            chunk_size: Number of tokens per chunk
            max_workers: Number of parallel worker threads for entity and relation extraction
            conflict_max_workers: Number of parallel worker threads for conflict detection
        """
        logger.info(f"Starting construct_graph with {len(docs)} documents")

        # Initialize LLM client
        llm_config = LLMConfig(
            model_name=self.global_config.llm_name,
            base_url=self.global_config.llm_base_url,
            temperature=0.0,
            max_workers=max_workers
        )
        llm_client = LLMClient(llm_config)

        # Initialize conflict resolver
        conflict_resolver = ConflictResolver(
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            schema_frequency_threshold=self.conflict_resolution_threshold
        )

        # Initialize relation extractor
        relation_extractor = RelationExtractor(
            llm_config=llm_config,
            max_entities_per_chunk=50
        )

        # Step 1: Chunk documents and extract entities
        logger.info("Step 1: Chunking documents and extracting entities...")
        chunks_with_entities = self._extract_entities_parallel(docs, chunk_size, max_workers)

        # Initialize queue for producer-consumer pattern
        chunk_queue = queue.Queue()
        for chunk_data in chunks_with_entities:
            chunk_queue.put(chunk_data)

        # Add sentinel values to signal worker termination
        for _ in range(max_workers):
            chunk_queue.put(None)

        # Thread-safe locks for Memory operations
        memory_lock = threading.Lock()
        schema_count_lock = threading.Lock()
        schema_frequencies: Dict[Tuple[str, str, str], int] = {}

        # Conflict resolution results collector
        conflict_results: List[Dict[str, Any]] = []
        conflict_lock = threading.Lock()

        def relation_extraction_worker(worker_id: int):
            """Worker thread for relation extraction"""
            while True:
                chunk_data = chunk_queue.get()
                if chunk_data is None:
                    chunk_queue.task_done()
                    break

                chunk_id, chunk_text, entities = chunk_data

                # Call schema_fact_extract to extract relations and schemas
                triples_with_schema = relation_extractor.extract_from_chunk_streaming(
                    chunk_text=chunk_text,
                    entities=entities,
                    chunk_id=chunk_id
                )

                if not triples_with_schema:
                    chunk_queue.task_done()
                    continue

                # Write to Memory and check for conflicts
                for tws in triples_with_schema:
                    triple_tuple = tws.triple
                    schema_tuple = tws.schema

                    with memory_lock:
                        # Get or create passage node
                        passage_idx = self.memory._get_or_create_passage(chunk_id, chunk_text)

                        # Get or create schema node
                        schema_idx = self.memory._get_or_create_schema(schema_tuple)

                        # Get or create fact node
                        fact_idx = self.memory._get_or_create_fact(triple_tuple, schema_idx)

                        # Update frequency statistics
                        with schema_count_lock:
                            current_freq = len(self.memory.schema_layer[schema_idx].fact_indices)
                            schema_frequencies[schema_tuple] = current_freq + 1

                        # Establish inter-layer relationships
                        schema_node = self.memory.schema_layer[schema_idx]
                        if fact_idx not in schema_node.fact_indices:
                            schema_node.fact_indices.append(fact_idx)

                        fact_node = self.memory.fact_layer[fact_idx]
                        if passage_idx not in fact_node.passage_indices:
                            fact_node.passage_indices.append(passage_idx)

                        passage_node = self.memory.passage_layer[passage_idx]
                        if fact_idx not in passage_node.fact_indices:
                            passage_node.fact_indices.append(fact_idx)

                        # Check if conflict detection is needed
                        new_freq = len(schema_node.fact_indices)
                        if new_freq > self.conflict_resolution_threshold:
                            # Add triple to conflict detector
                            conflict_resolver.add_triple_for_conflict_check(
                                triple=triple_tuple,
                                triple_id=f"fact_{fact_idx}",
                                source_passage=chunk_text,
                                schema=schema_tuple
                            )

                chunk_queue.task_done()

        def conflict_resolution_worker(worker_id: int):
            """Worker thread for conflict detection and resolution"""
            while True:
                # Get schemas that need processing
                with schema_count_lock:
                    schemas_to_process = [
                        schema for schema, freq in schema_frequencies.items()
                        if freq > self.conflict_resolution_threshold
                        and conflict_resolver.should_process_schema(schema)
                    ]

                if not schemas_to_process:
                    time.sleep(0.1)
                    continue

                # Process each schema
                for schema in schemas_to_process:
                    result = conflict_resolver.check_and_resolve_schema_conflicts(schema)
                    if result:
                        with conflict_lock:
                            conflict_results.append(result)
                        logger.info(f"Resolved conflict for schema: {schema}")

        # Start relation extraction worker threads
        relation_workers = []
        for i in range(max_workers):
            t = threading.Thread(target=relation_extraction_worker, args=(i,))
            t.start()
            relation_workers.append(t)

        # Start conflict resolution worker threads
        conflict_workers = []
        for i in range(conflict_max_workers):
            t = threading.Thread(target=conflict_resolution_worker, args=(i,))
            t.start()
            conflict_workers.append(t)

        # Wait for relation extraction to complete
        for t in relation_workers:
            t.join()

        # Wait for conflict resolution to complete (with timeout)
        logger.info("Waiting for conflict resolution to complete...")
        max_wait_time = 60
        wait_interval = 2
        elapsed = 0
        while elapsed < max_wait_time:
            pending = conflict_resolver.get_pending_count()
            if pending == 0:
                break
            time.sleep(wait_interval)
            elapsed += wait_interval
            logger.info(f"Pending conflicts: {pending}")

        # Force terminate conflict resolution threads
        for t in conflict_workers:
            t.join(timeout=5)

        # Output statistics
        logger.info("=" * 60)
        logger.info("construct_graph completed!")
        logger.info(f"  Schema layer: {len(self.memory.schema_layer)} unique schemas")
        logger.info(f"  Fact layer: {len(self.memory.fact_layer)} unique triples")
        logger.info(f"  Passage layer: {len(self.memory.passage_layer)} unique chunks")
        logger.info(f"  Conflicts resolved: {len(conflict_results)}")
        logger.info("=" * 60)

        return {
            "num_schemas": len(self.memory.schema_layer),
            "num_facts": len(self.memory.fact_layer),
            "num_passages": len(self.memory.passage_layer),
            "conflicts_resolved": len(conflict_results)
        }

    def _extract_entities_parallel(
        self,
        docs: List[str],
        chunk_size: int,
        max_workers: int
    ) -> List[Tuple[str, str, List[Dict]]]:
        """
        Extract entities in parallel.

        Parameters:
            docs: List of documents
            chunk_size: Number of tokens per chunk
            max_workers: Number of parallel worker threads

        Returns:
            List of (chunk_id, chunk_text, entities) tuples
        """
        import spacy
        try:
            nlp = spacy.load("en_core_web_trf")
        except OSError:
            logger.warning("en_core_web_trf not found, trying en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

        def process_doc(doc_text: str) -> List[Tuple[str, str, List[Dict]]]:
            """Process a single document, return all its chunks and entities"""
            results = []
            doc = nlp(doc_text)
            chunks = []

            # Chunk the document
            current_tokens = []
            current_start = 0
            for i, token in enumerate(doc):
                current_tokens.append(token)
                if len(current_tokens) >= chunk_size or i == len(doc) - 1:
                    if current_tokens:
                        chunk_start = current_tokens[0].idx
                        chunk_end = current_tokens[-1].idx + len(current_tokens[-1].text)
                        chunk_text = doc_text[chunk_start:chunk_end]
                        chunk_id = compute_mdhash_id(chunk_text, prefix="chunk-")
                        chunks.append((chunk_id, chunk_text))
                        current_tokens = []

            # Extract entities from each chunk
            for chunk_id, chunk_text in chunks:
                chunk_doc = nlp(chunk_text)
                entities = []
                for ent in chunk_doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start_char': ent.start_char,
                        'end_char': ent.end_char
                    })
                results.append((chunk_id, chunk_text, entities))

            return results

        # Process documents in parallel
        all_chunks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_doc, doc): doc for doc in docs}
            for future in tqdm(futures, desc="Extracting entities"):
                results = future.result()
                all_chunks.extend(results)

        logger.info(f"Extracted {len(all_chunks)} chunks with entities")
        return all_chunks