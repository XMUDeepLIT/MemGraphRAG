"""
Build a three-layer Memory structure with inter-layer connections:
1. Schema layer: Stores ontology (type triples), frequency, vectors, lower-level fact indices
2. Fact layer: Stores triples, vectors, upper-level schema indices, lower-level passage indices
3. Passage layer: Stores chunks, vectors, upper-level fact indices

Inter-layer relationships:
    Schema --1:N--> Fact --N:M--> Passage

Raw OpenIE outputs (``extracted_triples`` only) can be loaded with
``ThreeLayerMemory.build_from_raw_openie_results`` so ``schema_layer`` stays empty.
"""

import os
import json
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import numpy as np


# Default input path
DEFAULT_FILTERED_PATH = (
    ""
    ""
)

# Default output path
DEFAULT_MEMORY_OUTPUT_PATH = (
    ""
    ""
)


@dataclass
class SchemaNode:
    """Schema layer node: stores ontology information"""
    idx: int                                    # Unique index
    content: Tuple[str, str, str]               # ontology: (head_type, relation, tail_type)
    frequency: int = 0                          # Frequency of this ontology
    embedding: Optional[List[float]] = None     # Vector representation
    fact_indices: List[int] = field(default_factory=list)  # Lower-level fact index list


@dataclass
class FactNode:
    """Fact layer node: stores triple information"""
    idx: int                                    # Unique index
    content: Tuple[str, str, str]               # Triple: (head, relation, tail)
    frequency: int = 0                         # Distinct chunks (passages) this fact appears in
    embedding: Optional[List[float]] = None     # Vector representation
    schema_idx: int = -1                        # Upper-level schema index (unique)
    passage_indices: List[int] = field(default_factory=list)  # Lower-level passage index list


@dataclass
class PassageNode:
    """Passage layer node: stores chunk information"""
    idx: int                                    # Unique index
    chunk_id: str                               # Original chunk id
    content: str                                # Passage text
    embedding: Optional[List[float]] = None     # Vector representation
    fact_indices: List[int] = field(default_factory=list)  # Upper-level fact index list


class ThreeLayerMemory:
    """Three-layer Memory structure"""

    def __init__(self):
        # Three layer lists
        self.schema_layer: List[SchemaNode] = []
        self.fact_layer: List[FactNode] = []
        self.passage_layer: List[PassageNode] = []

        # Mappings for deduplication and fast lookup
        self._schema_to_idx: Dict[Tuple[str, str, str], int] = {}
        self._fact_to_idx: Dict[Tuple[str, str, str], int] = {}
        self._chunk_id_to_idx: Dict[str, int] = {}

    def _get_or_create_schema(self, ontology: Tuple[str, str, str]) -> int:
        """Get or create schema node, return index"""
        if ontology in self._schema_to_idx:
            return self._schema_to_idx[ontology]

        idx = len(self.schema_layer)
        node = SchemaNode(idx=idx, content=ontology)
        self.schema_layer.append(node)
        self._schema_to_idx[ontology] = idx
        return idx

    def _get_or_create_fact(
        self, triple: Tuple[str, str, str], schema_idx: int
    ) -> int:
        """Get or create fact node, return index"""
        if triple in self._fact_to_idx:
            return self._fact_to_idx[triple]

        idx = len(self.fact_layer)
        node = FactNode(idx=idx, content=triple, schema_idx=schema_idx)
        self.fact_layer.append(node)
        self._fact_to_idx[triple] = idx
        return idx

    def _get_or_create_passage(self, chunk_id: str, passage_text: str) -> int:
        """Get or create passage node, return index"""
        if chunk_id in self._chunk_id_to_idx:
            return self._chunk_id_to_idx[chunk_id]

        idx = len(self.passage_layer)
        node = PassageNode(idx=idx, chunk_id=chunk_id, content=passage_text)
        self.passage_layer.append(node)
        self._chunk_id_to_idx[chunk_id] = idx
        return idx

    @staticmethod
    def _triple_tuple_from_openie_item(item: Any) -> Optional[Tuple[str, str, str]]:
        """
        Normalize one element from extracted_triples to (h, r, t).

        Matches common OpenIE payloads: processed_triple, raw_triple, triple_str,
        or a bare length-3 list/tuple.
        """
        if item is None:
            return None
        if isinstance(item, (list, tuple)) and len(item) == 3:
            try:
                return (str(item[0]), str(item[1]), str(item[2]))
            except Exception:
                return None
        if isinstance(item, dict):
            for key in ("processed_triple", "triple", "raw_triple"):
                triple = item.get(key)
                if isinstance(triple, (list, tuple)) and len(triple) == 3:
                    try:
                        return (str(triple[0]), str(triple[1]), str(triple[2]))
                    except Exception:
                        pass
            ts = item.get("triple_str")
            if isinstance(ts, str):
                try:
                    t = ast.literal_eval(ts.strip())
                    if isinstance(t, (list, tuple)) and len(t) == 3:
                        return (str(t[0]), str(t[1]), str(t[2]))
                except Exception:
                    pass
        return None

    def build_from_raw_openie_results(self, data: Dict[str, Any]) -> None:
        """
        Build fact + passage layers from raw OpenIE JSON (no schema / ontology).

        Expects the same ``docs`` list shape as ner OpenIE outputs: each doc has
        ``idx``, ``passage``, and ``extracted_triples`` (list of dicts or 3-tuples).

        Does not populate ``schema_layer``. Every ``FactNode`` has ``schema_idx == -1``.

        Prefer a freshly constructed ``ThreeLayerMemory`` (``schema_layer`` is not cleared).
        """
        docs = data.get("docs", [])
        print(f"Building two-layer memory (no schema) from {len(docs)} docs...")

        for doc in docs:
            chunk_id = doc.get("idx", "")
            passage_text = doc.get("passage", "")
            triple_entries = doc.get("extracted_triples") or []

            if not isinstance(triple_entries, list) or not triple_entries:
                continue

            passage_idx = self._get_or_create_passage(chunk_id, passage_text)

            for item in triple_entries:
                triple_tuple = self._triple_tuple_from_openie_item(item)
                if triple_tuple is None:
                    continue

                fact_idx = self._get_or_create_fact(triple_tuple, schema_idx=-1)

                if passage_idx not in self.fact_layer[fact_idx].passage_indices:
                    self.fact_layer[fact_idx].passage_indices.append(passage_idx)
                if fact_idx not in self.passage_layer[passage_idx].fact_indices:
                    self.passage_layer[passage_idx].fact_indices.append(fact_idx)

        for fact_node in self.fact_layer:
            fact_node.frequency = len(fact_node.passage_indices)

        print(f"Built memory structure:")
        print(f"  Schema layer: {len(self.schema_layer)} unique ontologies (expected 0)")
        print(f"  Fact layer:   {len(self.fact_layer)} unique triples")
        print(f"  Passage layer: {len(self.passage_layer)} unique chunks")

    def build_from_openie_results(self, data: Dict[str, Any]) -> None:
        """
        Build three-layer structure from openie_results_with_ontology_filtered.json
        """
        docs = data.get("docs", [])
        print(f"Building memory from {len(docs)} docs...")

        for doc in docs:
            chunk_id = doc.get("idx", "")
            passage_text = doc.get("passage", "")
            triple_ont_map = doc.get("extracted_triple_ontology") or {}

            if not isinstance(triple_ont_map, dict) or not triple_ont_map:
                continue

            # 1. First create passage node
            passage_idx = self._get_or_create_passage(chunk_id, passage_text)

            # 2. Traverse all (triple_key, ontology) in this doc
            for triple_key, ontology in triple_ont_map.items():
                # Parse triple_key -> tuple
                try:
                    triple_tuple = ast.literal_eval(triple_key)
                    if not (isinstance(triple_tuple, tuple) and len(triple_tuple) == 3):
                        continue
                except Exception:
                    continue

                # Parse ontology -> tuple
                if not (isinstance(ontology, list) and len(ontology) == 3):
                    continue
                ontology_tuple = tuple(ontology)

                # 3. Create or get schema node
                schema_idx = self._get_or_create_schema(ontology_tuple)

                # 4. Create or get fact node
                fact_idx = self._get_or_create_fact(triple_tuple, schema_idx)

                # 5. Establish inter-layer index relationships
                # Schema -> Fact (lower-level index)
                if fact_idx not in self.schema_layer[schema_idx].fact_indices:
                    self.schema_layer[schema_idx].fact_indices.append(fact_idx)

                # Fact -> Passage (lower-level index)
                if passage_idx not in self.fact_layer[fact_idx].passage_indices:
                    self.fact_layer[fact_idx].passage_indices.append(passage_idx)

                # Passage -> Fact (upper-level index)
                if fact_idx not in self.passage_layer[passage_idx].fact_indices:
                    self.passage_layer[passage_idx].fact_indices.append(fact_idx)

        # 6. Count schema frequency (number of facts for each schema)
        for schema_node in self.schema_layer:
            schema_node.frequency = len(schema_node.fact_indices)

        # 7. Fact frequency: distinct chunks this triple was extracted from
        for fact_node in self.fact_layer:
            fact_node.frequency = len(fact_node.passage_indices)

        print(f"Built memory structure:")
        print(f"  Schema layer: {len(self.schema_layer)} unique ontologies")
        print(f"  Fact layer:   {len(self.fact_layer)} unique triples")
        print(f"  Passage layer: {len(self.passage_layer)} unique chunks")

    def get_schema_by_idx(self, idx: int) -> Optional[SchemaNode]:
        """Get schema node by index"""
        if 0 <= idx < len(self.schema_layer):
            return self.schema_layer[idx]
        return None

    def get_fact_by_idx(self, idx: int) -> Optional[FactNode]:
        """Get fact node by index"""
        if 0 <= idx < len(self.fact_layer):
            return self.fact_layer[idx]
        return None

    def get_passage_by_idx(self, idx: int) -> Optional[PassageNode]:
        """Get passage node by index"""
        if 0 <= idx < len(self.passage_layer):
            return self.passage_layer[idx]
        return None

    def get_facts_by_schema(self, schema_idx: int) -> List[FactNode]:
        """Get all facts under a schema"""
        schema = self.get_schema_by_idx(schema_idx)
        if schema is None:
            return []
        return [self.fact_layer[i] for i in schema.fact_indices]

    def get_passages_by_fact(self, fact_idx: int) -> List[PassageNode]:
        """Get all passages from which a fact was extracted"""
        fact = self.get_fact_by_idx(fact_idx)
        if fact is None:
            return []
        return [self.passage_layer[i] for i in fact.passage_indices]

    def get_facts_by_passage(self, passage_idx: int) -> List[FactNode]:
        """Get all facts extracted from a passage"""
        passage = self.get_passage_by_idx(passage_idx)
        if passage is None:
            return []
        return [self.fact_layer[i] for i in passage.fact_indices]

    def get_schema_by_fact(self, fact_idx: int) -> Optional[SchemaNode]:
        """Get the schema corresponding to a fact"""
        fact = self.get_fact_by_idx(fact_idx)
        if fact is None:
            return None
        return self.get_schema_by_idx(fact.schema_idx)

    def to_dict(self) -> Dict[str, Any]:
        """Export to serializable dictionary"""
        return {
            "schema_layer": [
                {
                    "idx": node.idx,
                    "content": list(node.content),
                    "frequency": node.frequency,
                    "embedding": node.embedding,
                    "fact_indices": node.fact_indices,
                }
                for node in self.schema_layer
            ],
            "fact_layer": [
                {
                    "idx": node.idx,
                    "content": list(node.content),
                    "frequency": node.frequency,
                    "embedding": node.embedding,
                    "schema_idx": node.schema_idx,
                    "passage_indices": node.passage_indices,
                }
                for node in self.fact_layer
            ],
            "passage_layer": [
                {
                    "idx": node.idx,
                    "chunk_id": node.chunk_id,
                    "content": node.content,
                    "embedding": node.embedding,
                    "fact_indices": node.fact_indices,
                }
                for node in self.passage_layer
            ],
            "stats": {
                "num_schemas": len(self.schema_layer),
                "num_facts": len(self.fact_layer),
                "num_passages": len(self.passage_layer),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreeLayerMemory":
        """Restore Memory structure from dictionary"""
        memory = cls()

        # Restore schema layer
        for item in data.get("schema_layer", []):
            node = SchemaNode(
                idx=item["idx"],
                content=tuple(item["content"]),
                frequency=item.get("frequency", 0),
                embedding=item.get("embedding"),
                fact_indices=item.get("fact_indices", []),
            )
            memory.schema_layer.append(node)
            memory._schema_to_idx[node.content] = node.idx

        # Restore fact layer
        for item in data.get("fact_layer", []):
            passage_indices = item.get("passage_indices", [])
            node = FactNode(
                idx=item["idx"],
                content=tuple(item["content"]),
                frequency=item["frequency"]
                if "frequency" in item
                else len(passage_indices),
                embedding=item.get("embedding"),
                schema_idx=item.get("schema_idx", -1),
                passage_indices=passage_indices,
            )
            memory.fact_layer.append(node)
            memory._fact_to_idx[node.content] = node.idx

        # Restore passage layer
        for item in data.get("passage_layer", []):
            node = PassageNode(
                idx=item["idx"],
                chunk_id=item["chunk_id"],
                content=item["content"],
                embedding=item.get("embedding"),
                fact_indices=item.get("fact_indices", []),
            )
            memory.passage_layer.append(node)
            memory._chunk_id_to_idx[node.chunk_id] = node.idx

        return memory

    def save(self, path: str) -> None:
        """Save to JSON file"""
        print(f"Saving memory structure to: {path}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print("Saved.")

    @classmethod
    def load(cls, path: str) -> "ThreeLayerMemory":
        """Load from JSON file"""
        print(f"Loading memory structure from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        memory = cls.from_dict(data)
        print(f"Loaded: {len(memory.schema_layer)} schemas, "
              f"{len(memory.fact_layer)} facts, {len(memory.passage_layer)} passages")
        return memory

    def print_summary(self) -> None:
        """Print statistical summary"""
        print("\n===== Three-Layer Memory Summary =====")
        print(f"Schema layer (ontologies): {len(self.schema_layer)}")
        print(f"Fact layer (triples):      {len(self.fact_layer)}")
        print(f"Passage layer (chunks):    {len(self.passage_layer)}")

        # Schema frequency distribution
        if self.schema_layer:
            freqs = [s.frequency for s in self.schema_layer]
            print(f"\nSchema frequency stats:")
            print(f"  Min: {min(freqs)}, Max: {max(freqs)}, "
                  f"Avg: {sum(freqs)/len(freqs):.2f}")

        # Fact to passage count distribution
        if self.fact_layer:
            passage_counts = [f.frequency for f in self.fact_layer]
            print(f"\nFact -> Passage count stats:")
            print(f"  Min: {min(passage_counts)}, Max: {max(passage_counts)}, "
                  f"Avg: {sum(passage_counts)/len(passage_counts):.2f}")

        # Fact count in passages distribution
        if self.passage_layer:
            fact_counts = [len(p.fact_indices) for p in self.passage_layer]
            print(f"\nPassage -> Fact count stats:")
            print(f"  Min: {min(fact_counts)}, Max: {max(fact_counts)}, "
                  f"Avg: {sum(fact_counts)/len(fact_counts):.2f}")

    def print_sample(self, n: int = 3) -> None:
        """Print sample data"""
        print(f"\n===== Sample Data (top {n}) =====")

        print("\n--- Schema Layer ---")
        for schema in self.schema_layer[:n]:
            print(f"  [{schema.idx}] {schema.content}, freq={schema.frequency}, "
                  f"facts={schema.fact_indices[:5]}{'...' if len(schema.fact_indices) > 5 else ''}")

        print("\n--- Fact Layer ---")
        for fact in self.fact_layer[:n]:
            print(f"  [{fact.idx}] {fact.content}, freq={fact.frequency}, schema={fact.schema_idx}, "
                  f"passages={fact.passage_indices[:5]}{'...' if len(fact.passage_indices) > 5 else ''}")

        print("\n--- Passage Layer ---")
        for passage in self.passage_layer[:n]:
            content_preview = passage.content[:80] + "..." if len(passage.content) > 80 else passage.content
            print(f"  [{passage.idx}] {passage.chunk_id}")
            print(f"       content: {content_preview}")
            print(f"       facts={passage.fact_indices[:5]}{'...' if len(passage.fact_indices) > 5 else ''}")


def load_openie_results(path: str) -> Dict[str, Any]:
    """Load OpenIE results JSON"""
    print(f"Loading OpenIE results from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data.get('docs', []))} docs")
    return data


def build_two_layer_memory_from_raw_openie(data: Dict[str, Any]) -> ThreeLayerMemory:
    """
    Build a ``ThreeLayerMemory`` with empty ``schema_layer`` from raw OpenIE ``data``
    (``docs`` with ``extracted_triples``, no ontology map).
    """
    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(data)
    return memory


def main():
    input_path = DEFAULT_FILTERED_PATH
    output_path = DEFAULT_MEMORY_OUTPUT_PATH

    # 1. Load data
    data = load_openie_results(input_path)

    # 2. Build three-layer Memory
    memory = ThreeLayerMemory()
    memory.build_from_openie_results(data)

    # 3. Print summary and sample
    memory.print_summary()
    memory.print_sample(n=3)

    # 4. Save structure
    memory.save(output_path)

    # 5. Verify loading
    print("\n--- Verification: reload and check ---")
    loaded_memory = ThreeLayerMemory.load(output_path)
    loaded_memory.print_summary()


if __name__ == "__main__":
    main()