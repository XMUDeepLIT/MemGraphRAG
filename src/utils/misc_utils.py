from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional, Set
import logging
import numpy as np
import re

from .typing import Triple
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]
    triple_records: Optional[List[Dict[str, Any]]] = None

@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None


    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]]  if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
        }

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):
    ner_output_dict = {}
    triple_output_dict = {}

    for chunk_item in corpus_openie_results:
        chunk_id = chunk_item['idx']
        ner_output_dict[chunk_id] = NerRawOutput(
            chunk_id=chunk_id,
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))
        )

        triple_records = []
        seen_triples: Set[Tuple[str, str, str]] = set()
        for entry in chunk_item.get('extracted_triples', []):
            record = normalize_triple_entry(entry)
            if record is None:
                continue
            triple_tuple = tuple(record['triple'])
            if triple_tuple in seen_triples:
                continue
            seen_triples.add(triple_tuple)
            triple_records.append(record)

        triple_output_dict[chunk_id] = TripleRawOutput(
            chunk_id=chunk_id,
            response=None,
            metadata={},
            triples=[record['triple'] for record in triple_records],
            triple_records=triple_records
        )

    return ner_output_dict, triple_output_dict

def extract_entity_nodes(chunk_triples: List[Triple]) -> (List[str], List[List[str]]):
    chunk_triple_entities = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[Triple]) -> Dict[str, str]:
    """
    Flattens chunk-level triples into a global fact registry.

    Returns:
        Dict[str, str]: Mapping from fact_id to canonical triple string representation.
    """
    fact_map: Dict[str, str] = {}
    for triples in chunk_triples:
        for triple in triples:
            if len(triple) != 3:
                continue
            triple_tuple = tuple(triple)
            triple_str = str(triple_tuple)
            fact_id = compute_mdhash_id(triple_str, prefix="fact-")
            fact_map[fact_id] = triple_str
    return fact_map


def normalize_triple_entry(entry: Union[List[str], Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Normalizes a triple entry (list or dict) into a canonical representation that includes
    the processed triple text, its string form, and a stable fact id.
    """
    if entry is None:
        return None

    raw_triple: Optional[List[str]] = None
    processed_candidate: Optional[Union[List[str], str]] = None
    triple_id: Optional[str] = None
    triple_str: Optional[str] = None

    if isinstance(entry, dict):
        raw_triple = entry.get('raw_triple')
        processed_candidate = entry.get('processed_triple') or entry.get('triple')
        triple_id = entry.get('fact_id') or entry.get('id')
        triple_str = entry.get('triple_str')
    else:
        raw_triple = entry
        processed_candidate = entry

    if processed_candidate is None:
        return None

    processed_triple = text_processing(processed_candidate)
    if not isinstance(processed_triple, list) or len(processed_triple) != 3:
        return None

    triple_tuple = tuple(processed_triple)
    if not triple_str:
        triple_str = str(triple_tuple)
    if not triple_id:
        triple_id = compute_mdhash_id(triple_str, prefix="fact-")

    record: Dict[str, Any] = {
        "fact_id": triple_id,
        "triple": processed_triple,
        "triple_str": triple_str
    }
    if raw_triple is not None:
        record["raw_triple"] = raw_triple

    return record

def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
