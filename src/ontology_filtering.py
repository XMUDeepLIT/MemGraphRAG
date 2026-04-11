import json
import os
import ast
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any


# Default input file: results containing extracted_triple_ontology
DEFAULT_OPENIE_RESULTS_WITH_ONTOLOGY_PATH = (
    ""
    ""
)
THRESHOLD = 1

def load_openie_results(path: str) -> Dict[str, Any]:
    """Load OpenIE results JSON with ontology."""
    print(f"Loading OpenIE results with ontology from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "docs" not in data or not isinstance(data["docs"], list):
        raise ValueError("Input JSON must contain key 'docs' as a list.")
    print(f"Loaded {len(data['docs'])} docs")
    return data


def build_ontology_stats(
    docs: List[Dict[str, Any]]
) -> Tuple[
    Counter,
    Dict[Tuple[str, str, str], List[Tuple[int, str]]],
    Dict[str, Tuple[str, str, str]],
]:
    """
    Traverse all docs, build statistics in two steps:

    1) For each triple_key (string like "('h', 'r', 't')"), keep only its "last occurrence" ontology,
       get triple_latest_ontology: triple_key -> (head_type, relation, tail_type).
    2) Then assign all occurrences of that triple (across all docs) to this final ontology, counting:
       - ontology_counter: (head_type, relation, tail_type) -> occurrence count
       - ontology_to_triples: ontology -> [(doc_idx, triple_key), ...]
    """
    # Step 1: Record the last ontology for each triple_key, and in which docs it appears
    triple_latest_ontology: Dict[str, Tuple[str, str, str]] = {}
    triple_occurrences: Dict[str, List[int]] = defaultdict(list)

    for doc_idx, doc in enumerate(docs):
        triple_ont_map = doc.get("extracted_triple_ontology") or {}
        if not isinstance(triple_ont_map, dict):
            continue

        for triple_key, ontology in triple_ont_map.items():
            # ontology expected to be [head_type, relation, tail_type]
            if not (isinstance(ontology, list) and len(ontology) == 3 and all(isinstance(x, str) for x in ontology)):
                continue
            ontology_tuple = tuple(ontology)  # type: ignore[assignment]
            # Keep only the "last occurrence" ontology (later overwrites earlier)
            triple_latest_ontology[triple_key] = ontology_tuple
            triple_occurrences[triple_key].append(doc_idx)

    # Step 2: Count frequency by "final ontology", and build ontology -> (doc_idx, triple_key) mapping
    ontology_counter: Counter = Counter()
    ontology_to_triples: Dict[Tuple[str, str, str], List[Tuple[int, str]]] = defaultdict(list)

    for triple_key, ontology_tuple in triple_latest_ontology.items():
        doc_indices = triple_occurrences.get(triple_key, [])
        for doc_idx in doc_indices:
            ontology_counter[ontology_tuple] += 1
            ontology_to_triples[ontology_tuple].append((doc_idx, triple_key))

    print(f"Collected stats for {len(ontology_counter)} unique ontology triples")
    return ontology_counter, ontology_to_triples, triple_latest_ontology


def _remove_triple_from_doc(doc: Dict[str, Any], triple_key: str) -> bool:
    """
    Remove the triple corresponding to triple_key from a single doc:
    - Delete the key from extracted_triple_ontology
    - Try to delete the corresponding [h, r, t] from extracted_triples list

    Return whether successfully deleted a triple from extracted_triples.
    """
    removed_from_triples = False

    # 1) Delete mapping from ontology dictionary
    triple_ont_map = doc.get("extracted_triple_ontology") or {}
    if isinstance(triple_ont_map, dict) and triple_key in triple_ont_map:
        del triple_ont_map[triple_key]
        doc["extracted_triple_ontology"] = triple_ont_map

    # 2) Delete corresponding triple from extracted_triples
    triples = doc.get("extracted_triples")
    if not isinstance(triples, list):
        return removed_from_triples

    try:
        # triple_key is a Python tuple string representation, e.g., "('h', 'r', 't')"
        triple_tuple = ast.literal_eval(triple_key)
        if isinstance(triple_tuple, tuple) and len(triple_tuple) == 3:
            triple_list = list(triple_tuple)
        else:
            triple_list = None
    except Exception:
        triple_list = None

    if triple_list is None:
        return removed_from_triples

    # Delete the first matching triple
    for i, triple in enumerate(triples):
        if (
            isinstance(triple, list)
            and len(triple) == 3
            and all(isinstance(x, str) for x in triple)
            and triple == triple_list
        ):
            del triples[i]
            removed_from_triples = True
            break

    doc["extracted_triples"] = triples
    return removed_from_triples


def _add_chunk_extra_fields(doc: Dict[str, Any]) -> None:
    """
    Add two extra fields for a single doc(chunk):
    1. unique_ontologies: List of unique ontologies in this chunk
    2. entity_mapping: List of type-entity correspondence

    Directly modify the passed doc dictionary.
    """
    triple_ont_map = doc.get("extracted_triple_ontology") or {}
    if not isinstance(triple_ont_map, dict):
        doc["unique_ontologies"] = []
        doc["entity_mapping"] = []
        return

    # 1. Collect unique ontologies
    unique_ontos = set()
    # Collect type-entity pair list
    entity_mapping: List[Dict[str, str]] = []

    for triple_key, ontology in triple_ont_map.items():
        # ontology format: [head_type, relation, tail_type]
        if not (isinstance(ontology, list) and len(ontology) == 3):
            continue
        onto_tuple = tuple(ontology)
        unique_ontos.add(onto_tuple)

        # Parse triple
        try:
            triple_tuple = ast.literal_eval(triple_key)
            if not (isinstance(triple_tuple, tuple) and len(triple_tuple) == 3):
                continue
        except Exception:
            continue

        head_type, _, tail_type = onto_tuple
        head_entity, _, tail_entity = triple_tuple

        # Add head type-entity pair
        entity_mapping.append({
            "type": head_type,
            "entity": head_entity
        })
        # Add tail type-entity pair
        entity_mapping.append({
            "type": tail_type,
            "entity": tail_entity
        })

    # Write to doc
    doc["unique_ontologies"] = [list(o) for o in unique_ontos]
    doc["entity_mapping"] = entity_mapping


def add_chunk_extra_fields(docs: List[Dict[str, Any]]) -> None:
    """
    Add extra fields (unique_ontologies and entity_mapping) for all doc(chunks).
    entity_mapping format: [{"type": "xxx", "entity": "xxx"}, ...]
    """
    print("Adding chunk extra fields (unique_ontologies & entity_mapping)...")
    for i, doc in enumerate(docs):
        _add_chunk_extra_fields(doc)
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(docs)} chunks...")

    # Statistics
    onto_counts = [len(doc.get("unique_ontologies", [])) for doc in docs]
    print(f"Done. Unique ontologies per chunk: min={min(onto_counts)}, max={max(onto_counts)}, avg={sum(onto_counts)/len(onto_counts):.2f}")


def filter_low_frequency_ontology(
    data: Dict[str, Any],
    min_count: int = 10,
) -> Dict[str, Any]:
    """
    Count statistics for all ontologies, delete ontologies with occurrence count below min_count,
    and delete all corresponding triples.

    Also print statistics of deleted ontologies and triples in console.
    """
    docs = data.get("docs", [])
    if not isinstance(docs, list):
        raise ValueError("'docs' must be a list.")

    # First build statistics, get "final ontology" for each triple_key
    ontology_counter, ontology_to_triples, triple_latest_ontology = build_ontology_stats(docs)

    # Unify extracted_triple_ontology in all docs to "final ontology"
    for doc in docs:
        triple_ont_map = doc.get("extracted_triple_ontology") or {}
        if not isinstance(triple_ont_map, dict):
            continue
        for triple_key in list(triple_ont_map.keys()):
            ontology_tuple = triple_latest_ontology.get(triple_key)
            if ontology_tuple is not None:
                triple_ont_map[triple_key] = list(ontology_tuple)
        doc["extracted_triple_ontology"] = triple_ont_map
    total_ontology_unique = len(ontology_counter)
    total_triple_instances = sum(ontology_counter.values())
    print(
        f"\nOriginal stats -> unique ontology: {total_ontology_unique}, "
        f"triple instances: {total_triple_instances}"
    )

    # Find ontologies to delete
    to_remove = {ont for ont, cnt in ontology_counter.items() if cnt < min_count}
    print(f"\nOntology with frequency < {min_count}: {len(to_remove)} will be removed.")

    total_deleted_triple_instances = 0
    deleted_ontology_info: List[Tuple[Tuple[str, str, str], int, int, List[str]]] = []

    for ontology in sorted(to_remove, key=lambda x: ontology_counter[x]):
        count = ontology_counter[ontology]
        triple_refs = ontology_to_triples.get(ontology, [])

        deleted_for_this_ontology = 0
        deleted_triple_keys: List[str] = []
        for doc_idx, triple_key in triple_refs:
            if 0 <= doc_idx < len(docs):
                doc = docs[doc_idx]
                if _remove_triple_from_doc(doc, triple_key):
                    deleted_for_this_ontology += 1
                    deleted_triple_keys.append(triple_key)

        total_deleted_triple_instances += deleted_for_this_ontology
        deleted_ontology_info.append((ontology, count, deleted_for_this_ontology, deleted_triple_keys))

    # Print detailed information
    print("\n===== Deleted Ontology with Triples =====\n")
    for ontology, freq, deleted_triples, triple_keys in deleted_ontology_info:
        head_type, relation, tail_type = ontology
        print(f"Ontology: ({head_type}) -[{relation}]-> ({tail_type})")
        print(f"  Original frequency: {freq}, Deleted triples ({deleted_triples}):")
        for tk in triple_keys:
            print(f"    - {tk}")
        print()
    print("\n===== Deleted Ontology Summary =====")
    
    print(f'threshold:{THRESHOLD}')
    print(
        f"Unique ontology removed: {len(deleted_ontology_info)} / {total_ontology_unique}, "
        f"remaining: {total_ontology_unique - len(deleted_ontology_info)}"
    )
    print(
        f"Total triple instances removed: {total_deleted_triple_instances} / {total_triple_instances}, "
        f"remaining: {total_triple_instances - total_deleted_triple_instances}\n"
    )

    # Add extra fields for each chunk
    add_chunk_extra_fields(docs)

    return data


def save_filtered_results(
    data: Dict[str, Any],
    input_path: str,
    output_path: str = None,
) -> str:
    """
    Save filtered results to a new file.
    If output_path is not explicitly specified, add suffix _filtered to original filename.
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_filtered{ext}"

    print(f"\nSaving filtered results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def main():
    input_path = DEFAULT_OPENIE_RESULTS_WITH_ONTOLOGY_PATH
    min_count = THRESHOLD+1

    data = load_openie_results(input_path)
    filtered_data = filter_low_frequency_ontology(data, min_count=min_count)
    save_filtered_results(filtered_data, input_path=input_path)


if __name__ == "__main__":
    main()