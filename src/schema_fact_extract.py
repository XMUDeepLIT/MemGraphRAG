#!/usr/bin/env python3
"""
Relation Extraction Script
Features:
1. Read chunks and entities from entity extraction result JSON file
2. Extract relations between entities using LLM
3. Output triples (entity, relation, entity) and corresponding schemas (type, relation, type)
4. Save results to JSON file
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
import queue

from llm_client import LLMClient, LLMConfig
from prompt_builder import build_relation_extraction_messages
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OPENAI_API_KEY"] = ""
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ChunkWithEntities:
    """Data class for chunk with extracted entities (used for streaming pipeline)"""
    chunk_id: str
    chunk_text: str
    entities: List[Dict]  # [{'text': ..., 'label': ...}, ...]


@dataclass
class TripleWithSchema:
    """Data class for extracted triple with schema (used for streaming pipeline)"""
    triple: Tuple[str, str, str]  # (head, relation, tail)
    schema: Tuple[str, str, str]   # (head_type, relation, tail_type)
    chunk_id: str
    passage: str


@dataclass
class RelationTriple:
    """Relation triple"""
    head: str           # Head entity text
    relation: str       # Relation type
    tail: str           # Tail entity text
    head_type: str      # Head entity type
    tail_type: str      # Tail entity type
    chunk_id: int       # Source chunk ID
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @property
    def triple(self) -> tuple:
        """Return triple (head, relation, tail)"""
        return (self.head, self.relation, self.tail)
    
    @property
    def schema(self) -> tuple:
        """Return schema (head_type, relation, tail_type)"""
        return (self.head_type, self.relation, self.tail_type)


class RelationExtractor:
    """Relation extractor"""
    
    def __init__(
        self, 
        llm_config: Optional[LLMConfig] = None,
        max_entities_per_chunk: int = 50
    ):
        """
        Initialize relation extractor
        
        Args:
            llm_config: LLM configuration
            max_entities_per_chunk: Maximum number of entities to process per chunk
        """
        self.llm_config = llm_config or LLMConfig()
        self.llm_client = LLMClient(self.llm_config)
        self.max_entities_per_chunk = max_entities_per_chunk
        
        logger.info(f"Relation extractor initialized: model={self.llm_config.model_name}")
    
    def load_entities_file(self, input_path: str) -> dict:
        """
        Load entity extraction result file
        
        Args:
            input_path: Input JSON file path
            
        Returns:
            Parsed JSON data
        """
        logger.info(f"Loading entity file: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"File loaded: {data['metadata']['total_chunks']} chunks, "
                   f"{data['metadata']['total_entities']} entities")
        
        return data
    
    def filter_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Filter entities, remove entity types not suitable for relation extraction
        
        Args:
            entities: Original entity list
            
        Returns:
            Filtered entity list
        """
        # Exclude entity types less suitable for relation extraction
        excluded_labels = {"CARDINAL", "ORDINAL", "MONEY", "PERCENT", "QUANTITY"}
        
        filtered = [
            ent for ent in entities 
            if ent.get("label") not in excluded_labels
        ]
        
        # Deduplicate (based on text and label)
        seen = set()
        unique_entities = []
        for ent in filtered:
            key = (ent["text"], ent["label"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)
        
        return unique_entities[:self.max_entities_per_chunk]
    
    def extract_relations_from_chunk(
        self, 
        chunk_text: str, 
        entities: List[Dict],
        chunk_id: int
    ) -> List[RelationTriple]:
        """
        Extract relations from a single chunk
        
        Args:
            chunk_text: Chunk text
            entities: Entity list in the chunk
            chunk_id: Chunk ID
            
        Returns:
            List of relation triples
        """
        # Filter entities
        filtered_entities = self.filter_entities(entities)
        
        if len(filtered_entities) < 2:
            logger.debug(f"Chunk {chunk_id}: Insufficient entities, skipping")
            return []
        
        # Build prompt
        messages = build_relation_extraction_messages(chunk_text, filtered_entities)
        
        try:
            # Call LLM
            response, metadata, cache_hit = self.llm_client.chat_json(messages)
            
            cache_status = "Cache hit" if cache_hit else "API call"
            logger.debug(f"Chunk {chunk_id}: {cache_status}, tokens={metadata.get('prompt_tokens', 0)}+{metadata.get('completion_tokens', 0)}")
            
            # Parse results
            triples = []
            if isinstance(response, list):
                for item in response:
                    if self._validate_relation(item):
                        triple = RelationTriple(
                            head=item["head"],
                            relation=item["relation"],
                            tail=item["tail"],
                            head_type=item["head_type"],
                            tail_type=item["tail_type"],
                            chunk_id=chunk_id
                        )
                        triples.append(triple)
            
            return triples
            
        except Exception as e:
            logger.error(f"Chunk {chunk_id} relation extraction failed: {e}")
            return []
    
    def extract_from_chunk_streaming(
        self,
        chunk_text: str,
        entities: List[Dict],
        chunk_id: str
    ) -> List[TripleWithSchema]:
        """
        Extract relations from a single chunk for streaming pipeline.
        Returns TripleWithSchema objects instead of RelationTriple.
        
        Args:
            chunk_text: Chunk text
            entities: Entity list in the chunk
            chunk_id: Chunk ID (string)
            
        Returns:
            List of TripleWithSchema objects
        """
        filtered_entities = self.filter_entities(entities)
        
        if len(filtered_entities) < 2:
            logger.debug(f"Chunk {chunk_id}: Insufficient entities, skipping")
            return []
        
        messages = build_relation_extraction_messages(chunk_text, filtered_entities)
        
        try:
            response, metadata, cache_hit = self.llm_client.chat_json(messages)
            
            triples_with_schema = []
            if isinstance(response, list):
                for item in response:
                    if self._validate_relation(item):
                        triple_tuple = (item["head"], item["relation"], item["tail"])
                        schema_tuple = (item["head_type"], item["relation"], item["tail_type"])
                        triples_with_schema.append(TripleWithSchema(
                            triple=triple_tuple,
                            schema=schema_tuple,
                            chunk_id=chunk_id,
                            passage=chunk_text
                        ))
            
            return triples_with_schema
            
        except Exception as e:
            logger.error(f"Chunk {chunk_id} streaming extraction failed: {e}")
            return []
    
    def _validate_relation(self, item: dict) -> bool:
        """Validate if relation item is valid"""
        required_fields = ["head", "relation", "tail", "head_type", "tail_type"]
        return all(field in item and item[field] for field in required_fields)
    
    def extract_relations_batch(
        self, 
        chunks: List[Dict]
    ) -> List[RelationTriple]:
        """
        Batch extract relations (parallel processing)
        
        Args:
            chunks: Chunk list
            
        Returns:
            All relation triples
        """
        # Prepare batch messages
        batch_data = []  # (chunk_id, messages, entities)
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["chunk_text"]
            entities = chunk.get("entities", [])
            
            filtered_entities = self.filter_entities(entities)
            
            if len(filtered_entities) < 2:
                continue
            
            messages = build_relation_extraction_messages(chunk_text, filtered_entities)
            batch_data.append((chunk_id, messages, filtered_entities))
        
        if not batch_data:
            logger.warning("No valid chunks to process")
            return []
        
        logger.info(f"Starting batch processing of {len(batch_data)} chunks...")
        
        # Batch call LLM
        batch_messages = [item[1] for item in batch_data]
        results = self.llm_client.batch_chat_json(
            batch_messages, 
            use_cache=True,
            show_progress=True
        )
        
        # Parse results
        all_triples = []
        for (chunk_id, _, _), (response, metadata, cache_hit) in zip(batch_data, results):
            if response is None:
                logger.warning(f"Chunk {chunk_id} returned empty result")
                continue
            
            if isinstance(response, list):
                for item in response:
                    if self._validate_relation(item):
                        triple = RelationTriple(
                            head=item["head"],
                            relation=item["relation"],
                            tail=item["tail"],
                            head_type=item["head_type"],
                            tail_type=item["tail_type"],
                            chunk_id=chunk_id
                        )
                        all_triples.append(triple)
        
        logger.info(f"Relation extraction completed: extracted {len(all_triples)} relations")
        return all_triples
    
    def process_file(
        self, 
        input_path: str, 
        output_path: str,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Process entity file and extract relations
        
        Args:
            input_path: Input JSON file path (entity extraction results)
            output_path: Output JSON file path
            parallel: Whether to process in parallel
            
        Returns:
            Processing result summary
        """
        # Load data
        data = self.load_entities_file(input_path)
        chunks = data.get("chunks", [])
        
        # Extract relations
        if parallel:
            all_triples = self.extract_relations_batch(chunks)
        else:
            all_triples = []
            for chunk in chunks:
                triples = self.extract_relations_from_chunk(
                    chunk["chunk_text"],
                    chunk.get("entities", []),
                    chunk["chunk_id"]
                )
                all_triples.extend(triples)
                logger.info(f"Chunk {chunk['chunk_id']}: Extracted {len(triples)} relations")
        
        # Build output data (pass chunks for organizing output by chunk)
        output_data = self._build_output(all_triples, chunks, data["metadata"])
        
        # Save results
        self._save_results(output_data, output_path)
        
        return output_data["metadata"]
    
    def _build_output(
        self, 
        triples: List[RelationTriple], 
        chunks: List[Dict],
        source_metadata: dict
    ) -> Dict[str, Any]:
        """
        Build output data structure, organized by chunk
        
        Args:
            triples: List of relation triples
            chunks: Original chunk list
            source_metadata: Source file metadata
            
        Returns:
            Output data dictionary
        """
        # Group triples by chunk_id
        triples_by_chunk = {}
        for t in triples:
            if t.chunk_id not in triples_by_chunk:
                triples_by_chunk[t.chunk_id] = []
            triples_by_chunk[t.chunk_id].append(t)
        
        # Build chunk-organized output
        chunks_output = []
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_triples = triples_by_chunk.get(chunk_id, [])
            
            # Triple to schema mapping (current chunk)
            triple_to_schema = {}
            for t in chunk_triples:
                triple_key = f"({t.head}, {t.relation}, {t.tail})"
                schema_value = f"({t.head_type}, {t.relation}, {t.tail_type})"
                triple_to_schema[triple_key] = schema_value
            
            chunk_output = {
                "chunk_id": chunk_id,
                "chunk_text": chunk["chunk_text"],
                "entities": chunk.get("entities", []),
                "num_entities": chunk.get("num_entities", len(chunk.get("entities", []))),
                "triples": [t.to_dict() for t in chunk_triples],
                "num_triples": len(chunk_triples),
                "triple_to_schema": triple_to_schema
            }
            chunks_output.append(chunk_output)
        
        # Global statistics
        # Triple to schema mapping (global)
        all_triple_to_schema = {}
        for t in triples:
            triple_key = f"({t.head}, {t.relation}, {t.tail})"
            schema_value = f"({t.head_type}, {t.relation}, {t.tail_type})"
            all_triple_to_schema[triple_key] = schema_value
        
        # Count schema distribution
        schema_counts = {}
        for t in triples:
            schema_key = f"({t.head_type}, {t.relation}, {t.tail_type})"
            schema_counts[schema_key] = schema_counts.get(schema_key, 0) + 1
        
        # Count relation type distribution
        relation_counts = {}
        for t in triples:
            relation_counts[t.relation] = relation_counts.get(t.relation, 0) + 1
        
        output_data = {
            "metadata": {
                "source_file": source_metadata,
                "total_chunks": len(chunks),
                "total_relations": len(triples),
                "unique_schemas": len(schema_counts),
                "relation_types_count": relation_counts,
                "schema_count": schema_counts,
                "model": self.llm_config.model_name
            },
            "chunks": chunks_output,
            "global_triple_to_schema": all_triple_to_schema
        }
        
        return output_data
    
    def _save_results(self, output_data: dict, output_path: str):
        """Save results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"  - Total chunks: {output_data['metadata']['total_chunks']}")
        logger.info(f"  - Total relations: {output_data['metadata']['total_relations']}")
        logger.info(f"  - Unique schemas: {output_data['metadata']['unique_schemas']}")
        
        # Display relations per chunk
        for chunk in output_data.get("chunks", []):
            logger.info(f"  - Chunk {chunk['chunk_id']}: {chunk['num_triples']} relations, {chunk['num_entities']} entities")


def main():
    """Main function"""
    
    # ==================== Configuration Parameters ====================
    
    # LLM configuration
    LLM_MODEL = "gpt-4o-mini"  # Model name
    LLM_BASE_URL = 'https:'  # API base URL, None means default OpenAI
    LLM_TEMPERATURE = 0.0  # Temperature parameter
    LLM_MAX_TOKENS = 16384  # Maximum generation tokens
    LLM_MAX_WORKERS = 10  # Parallel worker threads
    
    # Input/output configuration
    INPUT_FILE = "output/output_entities.json"  # Entity extraction result file
    OUTPUT_FILE = "output/output_relations.json"  # Relation extraction output file
    
    # Processing configuration
    PARALLEL = True  # Whether to process in parallel
    MAX_ENTITIES_PER_CHUNK = 100  # Maximum entities per chunk
    
    # ================================================================
    
    try:
        # Create LLM configuration
        llm_config = LLMConfig(
            model_name=LLM_MODEL,
            base_url=LLM_BASE_URL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            max_workers=LLM_MAX_WORKERS,
            cache_dir="./cache"
        )
        
        # Create relation extractor
        extractor = RelationExtractor(
            llm_config=llm_config,
            max_entities_per_chunk=MAX_ENTITIES_PER_CHUNK
        )
        
        # Process file
        result = extractor.process_file(
            INPUT_FILE, 
            OUTPUT_FILE,
            parallel=PARALLEL
        )
        
        logger.info("=" * 50)
        logger.info("Relation extraction completed!")
        logger.info("=" * 50)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


def demo():
    """Demo function"""
    
    # Example data
    demo_chunk = {
        "chunk_id": 0,
        "chunk_text": "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.",
        "entities": [
            {"text": "Apple Inc.", "label": "ORG"},
            {"text": "Cupertino", "label": "GPE"},
            {"text": "California", "label": "GPE"},
            {"text": "Steve Jobs", "label": "PERSON"},
            {"text": "Steve Wozniak", "label": "PERSON"},
            {"text": "Ronald Wayne", "label": "PERSON"},
            {"text": "April 1976", "label": "DATE"}
        ]
    }
    
    logger.info("Running demo mode...")
    
    # Create configuration
    llm_config = LLMConfig(
        model_name="gpt-4o-mini",
        temperature=0.0,
        cache_dir="./cache"
    )
    
    # Create extractor
    extractor = RelationExtractor(llm_config=llm_config)
    
    # Extract relations
    triples = extractor.extract_relations_from_chunk(
        demo_chunk["chunk_text"],
        demo_chunk["entities"],
        demo_chunk["chunk_id"]
    )
    
    # Print results
    logger.info(f"\nExtracted {len(triples)} relations:")
    for t in triples:
        logger.info(f"  Triple: {t.triple}")
        logger.info(f"  Schema: {t.schema}")
        logger.info("")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()