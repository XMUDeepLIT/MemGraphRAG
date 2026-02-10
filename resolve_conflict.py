import json
import os
from typing import Dict, List, Tuple, Optional, Any
from string import Template

import numpy as np
import pandas as pd
from src.hipporag.prompts.linking import get_query_instruction
from src.hipporag.utils.misc_utils import normalize_triple_entry
from src.hipporag.utils.llm_utils import TextChatMessage
from src.hipporag.llm import _get_llm_class
from src.hipporag.embedding_model import _get_embedding_model_class
from src.hipporag.utils.config_utils import BaseConfig
from prompt import PROMPT
# Optional environment variables
os.environ["OPENAI_API_KEY"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

DEFAULT_VDB_FACT_PATH = (
    ""
    ""
)

DEFAULT_OPENIE_RESULTS_PATH = (
    ""
    ""
)

# Global cache to avoid repeated parquet reading
_FACT_DF_CACHE: pd.DataFrame | None = None


def _load_fact_dataframe(parquet_path: str = DEFAULT_VDB_FACT_PATH) -> pd.DataFrame:
    """
    Read vdb_fact.parquet and cache in global variable for reuse in subsequent calls.
    """
    global _FACT_DF_CACHE

    if _FACT_DF_CACHE is not None:
        return _FACT_DF_CACHE

    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"vdb_fact.parquet not found at: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    required_cols = {"hash_id", "content", "embedding"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"vdb_fact.parquet is missing required columns: {missing}")

    _FACT_DF_CACHE = df
    return df


def load_fact_id_to_fact_dict(
    parquet_path: str = DEFAULT_VDB_FACT_PATH,
) -> Dict[str, str]:
    """
    Read the vdb_fact.parquet file from the specified path,
    return a dictionary {fact_id(hash_id): fact_string(content)}.
    """
    # Construct dictionary from cached DataFrame; if need to read from other path, temporarily bypass cache
    df = _load_fact_dataframe(parquet_path)

    fact_id_to_fact: Dict[str, str] = dict(
        zip(df["hash_id"].tolist(), df["content"].tolist())
    )
    return fact_id_to_fact


def encode_new_triple(
    triple_text: str,
    embedding_model,
) -> np.ndarray:
    """
    Encode a "new triple string" using the same method as HippoRAG to get its vector representation.

    Refer to HippoRAG.get_fact_scores approach:
    - Use embedding_model.batch_encode
    - Instruction: get_query_instruction('query_to_fact')
    - norm=True

    Parameters
    ----------
    triple_text : str
        Triple string to encode (e.g., "('barack obama', 'was born in', 'hawaii')").
    embedding_model :
        embedding_model instance used in HippoRAG (i.e., subclass of BaseEmbeddingModel).

    Returns
    -------
    np.ndarray
        Encoded vector, shape (dim,).
    """
    # batch_encode accepts List[str] or str, returns numpy array
    # According to HippoRAG implementation, can directly pass string or list
    emb_result = embedding_model.batch_encode(
        triple_text,  # Can pass string directly, batch_encode will handle
        instruction=get_query_instruction("query_to_fact"),
        norm=True,
    )
    
    # batch_encode returns numpy array, shape might be (1, dim) or (dim,)
    if emb_result is None:
        raise ValueError("embedding_model.batch_encode returned None, cannot get vector")
    
    emb = np.array(emb_result, dtype=np.float32)
    
    # If 2D array, take first row
    if emb.ndim == 2:
        if emb.shape[0] == 1:
            emb = emb[0]
        else:
            raise ValueError(f"Unexpected embedding shape: {emb.shape}")
    
    # Ensure 1D array
    if emb.ndim != 1:
        emb = emb.reshape(-1)
    
    return emb


def load_chunk_triples_and_triple_to_chunks(
    openie_results_path: str = DEFAULT_OPENIE_RESULTS_PATH,
) -> Tuple[List[List[List[str]]], Dict[str, List[str]]]:
    """
    Read all chunk triples from the specified openie_results_ner_*.json:
    - Build chunk_triples list: List[List[triple]], each element corresponds to a chunk's triple list;
    - Simultaneously build triple_to_chunks dictionary: fact_id -> [chunk_passage1, chunk_passage2, ...].

    For compatibility with current/old formats, call normalize_triple_entry for each extracted_triples element:
    - If it's a simple ["s","p","o"] list, will do text_processing normalization and generate fact_id/triple_str;
    - If it's already a dict with fact_id / processed_triple, will also be uniformly converted.
    """
    if not os.path.isfile(openie_results_path):
        raise FileNotFoundError(f"OpenIE results file not found: {openie_results_path}")

    with open(openie_results_path, "r", encoding="utf-8") as f:
        openie_results = json.load(f)

    all_openie_info = openie_results.get("docs", [])
    if not all_openie_info:
        # Allow empty, but return empty structure
        return [], {}

    chunk_triples: List[List[List[str]]] = []
    triple_to_chunks: Dict[str, List[str]] = {}

    for chunk_info in all_openie_info:
        passage = chunk_info.get("passage", "")
        triple_entries = chunk_info.get("extracted_triples", [])

        this_chunk_triples: List[List[str]] = []
        seen_triples_in_chunk = set()

        for entry in triple_entries:
            record = normalize_triple_entry(entry)
            if record is None:
                continue

            triple = record["triple"]  # [s, p, o], already text_processed
            fact_id = record.get("fact_id")
            triple_tuple = tuple(triple)
            if triple_tuple in seen_triples_in_chunk:
                continue
            seen_triples_in_chunk.add(triple_tuple)

            triple_str = record.get("triple_str") or str(triple_tuple)

            this_chunk_triples.append(triple)

            # Record triple -> chunk_passage
            if fact_id is None:
                # If no fact_id, fallback to triple_str as key
                fact_id = triple_str
            if fact_id not in triple_to_chunks:
                triple_to_chunks[fact_id] = []
            triple_to_chunks[fact_id].append(passage)

        chunk_triples.append(this_chunk_triples)

    return chunk_triples, triple_to_chunks


def find_triples_sharing_head_or_tail(
    new_triple: List[str],
    old_triples: List[List[str]],
    triple_to_chunks: Optional[Dict[str, List[str]]] = None,
    openie_results_path: str = DEFAULT_OPENIE_RESULTS_PATH,
) -> Dict[str, List[str]]:
    """
    Given a "new triple" and a batch of "existing triples", return all old triples
    that share head or tail entity with the new triple,
    along with the chunk content where these old triples appear.
    Returned dictionary uses fact_id as key for easy access via id.

    - Entity comparison is based on normalize_triple_entry + text_processing processed subject/object;
    - Old triple chunk content is looked up via triple_to_chunks dictionary;
      if not provided, will automatically build from openie_results_*.json.

    Parameters
    ----------
    new_triple : List[str]
        New triple [subject, predicate, object], can be original case form.
    old_triples : List[List[str]]
        List of old triples, each also [subject, predicate, object].
    triple_to_chunks : Optional[Dict[str, List[str]]]
        Optional, mapping fact_id -> [chunk_passage1, ...] (if missing fact_id, fallback to triple_str).
        If None, will automatically call load_chunk_triples_and_triple_to_chunks(openie_results_path) to build.
    openie_results_path : str
        When triple_to_chunks is None, used to build triple_to_chunks OpenIE result file path.

    Returns
    -------
    Dict[str, List[str]]
        Key: old triple's fact_id (or triple_str if missing),
        Value: list of passage texts for that triple across all chunks.
    """
    # Ensure triple_to_chunks exists
    if triple_to_chunks is None:
        _, triple_to_chunks = load_chunk_triples_and_triple_to_chunks(
            openie_results_path=openie_results_path
        )

    # Normalize new triple to get standardized head/tail entities
    new_record = normalize_triple_entry(new_triple)
    if new_record is None:
        raise ValueError(f"Invalid new_triple: {new_triple}")
    new_s, _, new_o = new_record["triple"]

    related: Dict[str, List[str]] = {}

    for t in old_triples:
        record = normalize_triple_entry(t)
        if record is None:
            continue

        s, _, o = record["triple"]
        fact_id = record.get("fact_id")

        # Check if shares head or tail entity
        if not (s == new_s or o == new_s or s == new_o or o == new_o):
            continue

        triple_tuple = tuple(record["triple"])
        triple_str = record.get("triple_str") or str(triple_tuple)
        fact_key = fact_id if fact_id is not None else triple_str

        chunks = triple_to_chunks.get(fact_key, [])
        if not chunks:
            # Cannot find corresponding chunk in OpenIE results, skip
            continue

        # Accumulate all chunk content for this old triple
        existing = related.get(fact_key, [])
        # Deduplicate and append
        for p in chunks:
            if p not in existing:
                existing.append(p)
        related[fact_key] = existing

    return related


def find_similar_facts(
    query_embedding: np.ndarray,
    parquet_path: str = DEFAULT_VDB_FACT_PATH,
    threshold: float = 0.8,
) -> List[Tuple[str, str, float]]:
    """
    Input a triple vector (recommended computed by encode_new_triple),
    find triples in vdb_fact.parquet with cosine similarity > threshold,
    return list [(fact_id, fact_content, similarity), ...] sorted by similarity descending.

    Parameters
    ----------
    query_embedding : np.ndarray
        Query triple vector, shape (dim,) or (1, dim).
    parquet_path : str
        Absolute path to vdb_fact.parquet.
    threshold : float
        Cosine similarity threshold, default 0.8.
    """
    df = _load_fact_dataframe(parquet_path)
    embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)
    fact_ids = df["hash_id"].tolist()
    contents = df["content"].tolist()

    # Normalize query_embedding shape and L2 normalize
    query_vec = np.array(query_embedding, dtype=np.float32)
    if query_vec.ndim == 2:
        query_vec = query_vec.squeeze(0)
    if query_vec.ndim != 1:
        raise ValueError(f"query_embedding should be 1D or (1, dim), got shape {query_embedding.shape}")

    # L2 normalize
    def l2_normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return x / norm

    embeddings_norm = l2_normalize(embeddings)
    query_vec_norm = l2_normalize(query_vec)

    # Cosine similarity = dot(normalized)
    sims = embeddings_norm @ query_vec_norm

    # Filter those greater than threshold
    mask = sims >= threshold
    if not np.any(mask):
        return []

    idxs = np.where(mask)[0]
    # Sort by similarity descending
    idxs_sorted = idxs[np.argsort(sims[idxs])[::-1]]

    results: List[Tuple[str, str, float]] = []
    for i in idxs_sorted:
        results.append((fact_ids[i], contents[i], float(sims[i])))
    return results


def detect_triple_conflicts(
    triple_list: List[List[str]],
    triple_ids: List[str],
    llm_model,
    embedding_model,
    fact_id_to_fact: Dict[str, str],
    triple_to_chunks: Optional[Dict[str, List[str]]] = None,
    openie_results_path: str = DEFAULT_OPENIE_RESULTS_PATH,
    vdb_fact_path: str = DEFAULT_VDB_FACT_PATH,
    similarity_threshold: float = 0.9,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Iterate through triple list, detect conflicts for each triple.

    For each triple, find:
    1. All triples sharing its head or tail entity
    2. Triples with similarity greater than threshold

    Send these triples to LLM to determine if conflicts exist.
    If conflicts exist, save relevant information.

    Parameters
    ----------
    triple_list : List[List[str]]
        Triple list, each triple is [head, relation, tail]
    triple_ids : List[str]
        Triple ID list, corresponds one-to-one with triple_list
    llm_model :
        LLM model instance for conflict detection
    embedding_model :
        Embedding model instance for similarity calculation
    fact_id_to_fact : Dict[str, str]
        Mapping from fact_id to fact string
    triple_to_chunks : Optional[Dict[str, List[str]]]
        Mapping from triple ID to chunk list
    openie_results_path : str
        OpenIE results file path
    vdb_fact_path : str
        vdb_fact.parquet file path
    similarity_threshold : float
        Similarity threshold, default 0.9
    output_path : Optional[str]
        Output file path, if None then not saved

    Returns
    -------
    Dict[str, Any]
        Dictionary containing conflict detection results
    """
    if len(triple_list) != len(triple_ids):
        raise ValueError("triple_list and triple_ids must have the same length")
    
    # Ensure triple_to_chunks exists
    if triple_to_chunks is None:
        _, triple_to_chunks = load_chunk_triples_and_triple_to_chunks(
            openie_results_path=openie_results_path
        )
    
    # Build mapping from fact_id to triple (from triple_list and triple_ids)
    fact_id_to_triple_map: Dict[str, List[str]] = {}
    for tid, t in zip(triple_ids, triple_list):
        fact_id_to_triple_map[tid] = t
    
    # Also build mapping from fact_id_to_fact (as supplement)
    import ast
    for fact_id, fact_content in fact_id_to_fact.items():
        if fact_id not in fact_id_to_triple_map:
            try:
                parsed_triple = ast.literal_eval(fact_content)
                if isinstance(parsed_triple, tuple) and len(parsed_triple) == 3:
                    fact_id_to_triple_map[fact_id] = list(parsed_triple)
            except:
                pass
    
    all_conflicts = []
    conflicting_triple_ids = set()
    
    print(f"Starting conflict detection for {len(triple_list)} triples...")
    
    for idx, (triple, triple_id) in enumerate(zip(triple_list, triple_ids)):
        if idx % 10 == 0:
            print(f"Processing progress: {idx}/{len(triple_list)}")
        
        # 1. Find triples sharing head or tail entity
        related_by_entity = find_triples_sharing_head_or_tail(
            new_triple=triple,
            old_triples=triple_list,
            triple_to_chunks=triple_to_chunks,
            openie_results_path=openie_results_path,
        )
        
        # 2. Find triples with similarity greater than threshold
        # First encode current triple
        triple_str = str(tuple(triple))
        try:
            triple_embedding = encode_new_triple(triple_str, embedding_model)
            similar_facts = find_similar_facts(
                query_embedding=triple_embedding,
                parquet_path=vdb_fact_path,
                threshold=similarity_threshold,
            )
        except Exception as e:
            print(f"Warning: Cannot encode triple {triple_id}: {e}")
            similar_facts = []
        
        # 3. Collect all related triples
        related_triples = []
        related_triple_ids = []
        
        # Add triples sharing entities
        for fact_id, chunks in related_by_entity.items():
            if fact_id != triple_id:  # Exclude self
                # Get triple from mapping
                related_triple = fact_id_to_triple_map.get(fact_id)
                if related_triple:
                    related_triples.append({
                        "triple": related_triple,
                        "id": fact_id,
                        "source": "shared_entity"
                    })
                    related_triple_ids.append(fact_id)
        
        # Add high similarity triples
        for fact_id, fact_content, similarity in similar_facts:
            if fact_id != triple_id:  # Exclude self
                if fact_id not in related_triple_ids:  # Avoid duplicates
                    # Prefer from mapping, otherwise try parsing fact_content
                    related_triple = fact_id_to_triple_map.get(fact_id)
                    if not related_triple:
                        try:
                            parsed_triple = ast.literal_eval(fact_content)
                            if isinstance(parsed_triple, tuple) and len(parsed_triple) == 3:
                                related_triple = list(parsed_triple)
                        except:
                            pass
                    
                    if related_triple:
                        related_triples.append({
                            "triple": related_triple,
                            "id": fact_id,
                            "source": "similarity",
                            "similarity": similarity
                        })
                        related_triple_ids.append(fact_id)
        
        # If no related triples, skip
        if not related_triples:
            continue
        
        # 4. Call LLM to detect conflicts
        try:
            conflict_result = check_conflict_with_llm(
                target_triple=triple,
                target_triple_id=triple_id,
                related_triples=related_triples,
                llm_model=llm_model,
            )
            
            if conflict_result.get("has_conflict", False):
                all_conflicts.append({
                    "target_triple": triple,
                    "target_triple_id": triple_id,
                    "conflicts": conflict_result.get("conflicts", []),
                    "conflicting_triple_ids": conflict_result.get("conflicting_triple_ids", []),
                })
                conflicting_triple_ids.add(triple_id)
                for cid in conflict_result.get("conflicting_triple_ids", []):
                    conflicting_triple_ids.add(cid)
                
                print(f"Conflict found: Triple {triple_id} conflicts with {len(conflict_result.get('conflicting_triple_ids', []))} triples")
        
        except Exception as e:
            print(f"Warning: Error detecting conflicts for triple {triple_id}: {e}")
            continue
    
    result = {
        "total_triples": len(triple_list),
        "conflicts_detected": len(all_conflicts),
        "conflicting_triple_ids": list(conflicting_triple_ids),
        "conflicts": all_conflicts,
    }
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Conflict detection results saved to: {output_path}")
    
    print(f"\nConflict detection completed:")
    print(f"  Total triples: {result['total_triples']}")
    print(f"  Conflict groups found: {result['conflicts_detected']}")
    print(f"  Triple IDs involved in conflicts: {len(result['conflicting_triple_ids'])}")
    
    return result


def check_conflict_with_llm(
    target_triple: List[str],
    target_triple_id: str,
    related_triples: List[Dict[str, Any]],
    llm_model,
) -> Dict[str, Any]:
    """
    Use LLM to detect conflicts between target triple and related triples.

    Parameters
    ----------
    target_triple : List[str]
        Target triple [head, relation, tail]
    target_triple_id : str
        Target triple ID
    related_triples : List[Dict[str, Any]]
        List of related triples, each containing triple, id, etc.
    llm_model :
        LLM model instance

    Returns
    -------
    Dict[str, Any]
        Dictionary containing has_conflict, conflicts, conflicting_triple_ids
    """
    # Prepare prompt
    prompt_template = PROMPT['conflict_detection']
    
    # Format related triples
    related_triples_str = ""
    for i, rt in enumerate(related_triples, 1):
        triple = rt.get("triple", [])
        rt_id = rt.get("id", "unknown")
        source = rt.get("source", "unknown")
        related_triples_str += f"Triple {i} (ID: {rt_id}, Source: {source}): {triple}\n"
    
    # Build user message
    user_prompt = Template(prompt_template["user"]).substitute(
        target_triple=str(target_triple),
        related_triples=related_triples_str.strip()
    )
    
    messages: List[TextChatMessage] = [
        {"role": "system", "content": prompt_template["system"]},
        {"role": "user", "content": user_prompt}
    ]
    
    # Call LLM
    try:
        response, metadata = llm_model.infer(messages)
        
        # Parse JSON response
        import re
        # Try to extract JSON part
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response
        
        # Fix potential JSON format issues
        from src.hipporag.utils.llm_utils import fix_broken_generated_json
        json_str = fix_broken_generated_json(json_str)
        
        result = json.loads(json_str)
        
        # Ensure correct return format
        if "has_conflict" not in result:
            result["has_conflict"] = False
        if "conflicts" not in result:
            result["conflicts"] = []
        if "conflicting_triple_ids" not in result:
            result["conflicting_triple_ids"] = []
        
        # If LLM returned conflicts but no conflicting_triple_ids, try to extract from conflicts
        if result.get("has_conflict") and result.get("conflicts") and not result.get("conflicting_triple_ids"):
            conflicting_ids = set()
            for conflict in result.get("conflicts", []):
                # Try to match triple1 and triple2 to IDs in related_triples
                triple1 = conflict.get("triple1", [])
                triple2 = conflict.get("triple2", [])
                
                # Match triple1 and triple2 with related_triples
                for rt in related_triples:
                    rt_triple = rt.get("triple", [])
                    rt_id = rt.get("id", "")
                    # Simple matching: compare triple content
                    if (isinstance(triple1, list) and len(triple1) == 3 and 
                        isinstance(rt_triple, list) and len(rt_triple) == 3):
                        if (triple1[0].lower() == rt_triple[0].lower() and
                            triple1[1].lower() == rt_triple[1].lower() and
                            triple1[2].lower() == rt_triple[2].lower()):
                            conflicting_ids.add(rt_id)
                    
                    if (isinstance(triple2, list) and len(triple2) == 3 and 
                        isinstance(rt_triple, list) and len(rt_triple) == 3):
                        if (triple2[0].lower() == rt_triple[0].lower() and
                            triple2[1].lower() == rt_triple[1].lower() and
                            triple2[2].lower() == rt_triple[2].lower()):
                            conflicting_ids.add(rt_id)
            
            result["conflicting_triple_ids"] = list(conflicting_ids)
        
        return result
    
    except Exception as e:
        print(f"LLM call or parsing failed: {e}")
        # Return default result
        return {
            "has_conflict": False,
            "conflicts": [],
            "conflicting_triple_ids": []
        }



def load_all_triples_with_ids(
    openie_results_path: str = DEFAULT_OPENIE_RESULTS_PATH,
) -> Tuple[List[List[str]], List[str]]:
    """
    Load all triples and their IDs from OpenIE results file.
    
    Parameters
    ----------
    openie_results_path : str
        OpenIE results file path
    
    Returns
    -------
    Tuple[List[List[str]], List[str]]
        (triple list, triple ID list)
    """
    if not os.path.isfile(openie_results_path):
        raise FileNotFoundError(f"OpenIE results file not found: {openie_results_path}")
    
    with open(openie_results_path, "r", encoding="utf-8") as f:
        openie_results = json.load(f)
    
    all_openie_info = openie_results.get("docs", [])
    if not all_openie_info:
        return [], []
    
    triple_list: List[List[str]] = []
    triple_ids: List[str] = []
    seen_triples = set()  # For deduplication
    
    for chunk_info in all_openie_info:
        triple_entries = chunk_info.get("extracted_triples", [])
        
        for entry in triple_entries:
            record = normalize_triple_entry(entry)
            if record is None:
                continue
            
            triple = record["triple"]  # [s, p, o]
            fact_id = record.get("fact_id")
            triple_tuple = tuple(triple)
            
            # Deduplication: if already processed this triple, skip
            if triple_tuple in seen_triples:
                continue
            seen_triples.add(triple_tuple)
            
            triple_str = record.get("triple_str") or str(triple_tuple)
            
            # Use fact_id as ID, if not available use triple_str
            triple_id = fact_id if fact_id is not None else triple_str
            
            triple_list.append(triple)
            triple_ids.append(triple_id)
    
    print(f"Loaded {len(triple_list)} unique triples from {openie_results_path}")
    return triple_list, triple_ids


def main():
    """
    Main function: load all triples and detect conflicts.
    """
    import sys
    from pathlib import Path
    
    # Add project path
    sys.path.insert(0, str(Path(__file__).parent))
    
    # ============ Configuration Parameters ============
    # OpenIE results file path
    OPENIE_RESULTS_PATH = (
        ""
        ""
    )
    
    # VDB fact file path
    VDB_FACT_PATH = (
        ""
        ""
        ""
    )
    
    # LLM configuration
    LLM_BASE_URL = "https:"  # Note: removed leading space
    LLM_NAME = "gpt-4o-mini"
    
    # Embedding model configuration
    EMBEDDING_NAME = ""
    
    # Similarity threshold
    SIMILARITY_THRESHOLD = 0.9
    
    # Output file path
    OUTPUT_PATH = (
        ""
        ""
    )
    
    # Optional: set API key (if needed)
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    # ============ End Configuration Parameters ============
    
    print("="*80)
    print("Triple Conflict Detection")
    print("="*80)
    
    try:
        # 1. Load all triples and IDs
        print("\n[Step 1/5] Loading triple data...")
        triple_list, triple_ids = load_all_triples_with_ids(
            openie_results_path=OPENIE_RESULTS_PATH
        )
        
        if len(triple_list) == 0:
            print("Error: No triples found")
            return
        
        print(f"✓ Successfully loaded {len(triple_list)} triples")
        
        # 2. Load fact_id to fact mapping
        print("\n[Step 2/5] Loading fact mapping...")
        fact_id_to_fact = load_fact_id_to_fact_dict(parquet_path=VDB_FACT_PATH)
        print(f"✓ Successfully loaded {len(fact_id_to_fact)} fact mappings")
        
        # 3. Initialize LLM model
        print("\n[Step 3/5] Initializing LLM model...")
        from src.hipporag.utils.config_utils import BaseConfig
        config = BaseConfig()
        config.llm_name = LLM_NAME
        config.llm_base_url = LLM_BASE_URL
        config.save_dir = ""
        
        llm_model = _get_llm_class(config)
        print(f"✓ LLM model initialized: {LLM_NAME}")
        
        # 4. Initialize Embedding model
        print("\n[Step 4/5] Initializing Embedding model...")
        from src.hipporag.embedding_model import _get_embedding_model_class
        EmbeddingModelClass = _get_embedding_model_class(EMBEDDING_NAME)
        embedding_model = EmbeddingModelClass(
            global_config=config,
            embedding_model_name=EMBEDDING_NAME
        )
        print(f"✓ Embedding model initialized: {EMBEDDING_NAME}")
        
        # 5. Execute conflict detection
        print("\n[Step 5/5] Starting conflict detection...")
        print(f"   - Total triples: {len(triple_list)}")
        print(f"   - Similarity threshold: {SIMILARITY_THRESHOLD}")
        print(f"   - Output file: {OUTPUT_PATH}")
        print()
        
        result = detect_triple_conflicts(
            triple_list=triple_list,
            triple_ids=triple_ids,
            llm_model=llm_model,
            embedding_model=embedding_model,
            fact_id_to_fact=fact_id_to_fact,
            openie_results_path=OPENIE_RESULTS_PATH,
            vdb_fact_path=VDB_FACT_PATH,
            similarity_threshold=SIMILARITY_THRESHOLD,
            output_path=OUTPUT_PATH,
        )
        
        # 6. Print result summary
        print("\n" + "="*80)
        print("Conflict detection completed!")
        print("="*80)
        print(f"Total triples: {result['total_triples']}")
        print(f"Conflict groups found: {result['conflicts_detected']}")
        print(f"Triple IDs involved in conflicts: {len(result['conflicting_triple_ids'])}")
        if result['conflicts_detected'] > 0:
            print(f"\nConflict details saved to: {OUTPUT_PATH}")
        print("="*80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()




if __name__ == "__main__":
    main()