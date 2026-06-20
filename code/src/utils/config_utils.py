import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional,
    Tuple,
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""
    # LLM specific attributes 
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )
    
    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."}
    )

    # Storage specific attributes 
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )
    
    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "Name of the text-based preprocessor to use in preprocessing."}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')
    
    
    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."}
    )
    
    
    # Embedding specific attributes
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    
    
    
    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    triple_duplicate_sim_threshold: float = field(
        default=0.9,
        metadata={"help": "Cosine similarity threshold for treating two facts with identical subject/object as duplicates."}
    )
    entity_equivalence_sim_threshold: float = field(
        default=0.9,
        metadata={"help": "Cosine similarity threshold for considering two identical-surface-form entities as the same node."}
    )
    enable_triple_conflict_resolution: bool = field(
        default=True,
        metadata={"help": "Enable sequential triple conflict detection before graph construction."}
    )
    triple_conflict_resolution_template_name: str = field(
        default="triple_conflict_resolution",
        metadata={"help": "Prompt template used for triple conflict detection and resolution."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )

    # Memory indexing pipeline
    memory_artifact_mode: Literal["default", "detailed"] = field(
        default="default",
        metadata={"help": "Whether to save only final memory artifacts or every pipeline stage."},
    )
    memory_max_workers: int = field(
        default=20,
        metadata={"help": "Shared worker count for parallel stages in the memory indexing pipeline."},
    )
    schema_extraction_sample_facts: int = 0
    schema_extraction_force_reextract: bool = False
    schema_extraction_max_new_tokens: int = 1024
    schema_extraction_temperature: float = 0.0
    ontology_filter_mode: Literal["absolute", "percentile"] = "percentile"
    ontology_filter_min_frequency: int = 2
    ontology_filter_low_percent: float = 20.0
    conflict_max_related_per_target: int = 20
    conflict_include_passage_evidence: bool = True
    conflict_passage_evidence_per_fact: int = 2
    conflict_min_confidence: float = 0.85
    conflict_streaming_only_previous: bool = True
    conflict_enable_reverse_relation_check: bool = True
    conflict_reverse_relations: Tuple[str, ...] = (
        "father of", "mother of", "parent of", "child of",
        "husband of", "wife of", "spouse of",
    )
    conflict_detection_max_new_tokens: int = 1200
    conflict_resolution_max_component_size: int = 20
    conflict_resolution_max_new_tokens: int = 1400
    conflict_resolution_enable_content_filter_fallback: bool = True
    conflict_resolution_set_schema_negative_on_modified: bool = False
    memory_graph_output_name: str = "memory_graph"
    
    
    
    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    skip_fact_rerank: bool = field(
        default=False,
        metadata={"help": "If True, skip the fact reranking step and use facts filtered by similarity threshold."}
    )
    fact_similarity_threshold: float = field(
        default=0.5,
        metadata={"help": "When skip_fact_rerank=True, only facts with similarity score >= this threshold will be returned."}
    )
    use_raw_threshold_filter: bool = field(
        default=False,
        metadata={"help": "When skip_fact_rerank=True, if True skip top-k selection and directly filter all facts by threshold (using raw scores)."}
    )
    retrieval_max_workers: int = field(
        default=1,
        metadata={"help": "Max workers for parallel retrieval. Set to 1 for serial execution."}
    )
    
    
    # QA specific attributes
    qa_max_workers: int = field(
        default=1,
        metadata={"help": "Max workers for parallel QA inference. Set to 1 for serial execution."}
    )
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    
    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )
    
    
    
    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        'dpr_only', 
        'entity', 
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation', 
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )
    
    
    def __post_init__(self):
        if self.save_dir is None: # If save_dir not given
            if self.dataset is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # customize your dataset's output dir here
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")
