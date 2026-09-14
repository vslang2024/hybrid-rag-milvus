"""
Quick utility to open an existing Milvus Lite database and look at what's
inside it — collection schema, row count, and sample rows. Handy for
debugging ingestion without re-running the whole RAG pipeline.

Install:
    pip install "pymilvus[milvus_lite]"

Usage:
    python inspect_db.py                       # uses ./hybrid_rag.db
    python inspect_db.py --db path/to/other.db
    python inspect_db.py --db http://localhost:19530   # Milvus Standalone (or set MILVUS_ADDRESS)
    python inspect_db.py --limit 20            # show more sample rows
    python inspect_db.py --full                # full text, not a 70-char preview
    python inspect_db.py --vectors             # also dump the index data: dense + sparse vectors
    python inspect_db.py --doc doc5 --vectors  # one row in full detail

Note: Milvus Lite allows one process per .db file — stop server.py first.
(Not an issue against Milvus Standalone.)
"""

import argparse
import collections
import math
import os
import re

from pymilvus import MilvusClient, DataType, FunctionType


def local_tokens(text: str) -> collections.Counter:
    """
    Reproduces Milvus's default "standard" analyzer (lowercase + split on
    non-alphanumerics) so we can label the sparse vector's hashed term ids.
    Milvus Lite doesn't implement run_analyzer(), so we can't ask it directly;
    the token count and term-frequency distribution match exactly on this data.
    """
    return collections.Counter(re.findall(r"[a-z0-9]+", text.lower()))


def print_vectors(row: dict):
    dense = row.get("dense") or []
    if dense:
        norm = math.sqrt(sum(x * x for x in dense))
        head = ", ".join(f"{x:+.4f}" for x in dense[:8])
        print(f"      dense : dim={len(dense)}  L2-norm={norm:.3f}  [{head}, ...]")
    sparse = row.get("sparse") or {}
    if sparse:
        # Milvus stores {hashed_term_id: term_frequency}; IDF is applied at query time.
        by_tf = sorted(sparse.items(), key=lambda kv: (-kv[1], kv[0]))
        pairs = ", ".join(f"{k}:{int(v)}" for k, v in by_tf[:6])
        print(f"      sparse: {len(sparse)} terms  {{{pairs}{', ...' if len(sparse) > 6 else ''}}}")
        tf = local_tokens(row.get("text", ""))
        top = ", ".join(f"{t}×{n}" for t, n in tf.most_common(8))
        print(f"      tokens (local re-analysis, {len(tf)} unique): {top}{', ...' if len(tf) > 8 else ''}")


def inspect(db_path: str | None, limit: int, full: bool = False, vectors: bool = False, doc: str | None = None):
    uri = db_path or os.environ.get("MILVUS_ADDRESS") or "hybrid_rag.db"
    token = os.environ.get("MILVUS_TOKEN", "")
    client = MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)
    db_path = uri

    collections = client.list_collections()
    print(f"Database: {db_path}")
    print(f"Collections: {collections}\n")

    for name in collections:
        print("=" * 80)
        print(f"Collection: {name}")

        # Schema — field names, types, dims
        desc = client.describe_collection(name)
        print("\nFields:")
        for field in desc["fields"]:
            type_name = DataType(field["type"]).name
            dim = field.get("params", {}).get("dim")
            print(f"  - {field['name']}: {type_name}" + (f" (dim={dim})" if dim else ""))

        # Functions — e.g. the built-in BM25 function that fills "sparse" from "text"
        functions = desc.get("functions", [])
        if functions:
            print("\nFunctions (computed inside Milvus):")
            for fn in functions:
                fn_type = FunctionType(fn["type"]).name if isinstance(fn["type"], int) else fn["type"]
                print(f"  - {fn['name']}: {fn_type}  "
                      f"{fn['input_field_names']} -> {fn['output_field_names']}")

        # Indexes — one per vector field: type + metric
        print("\nIndexes:")
        for index_name in client.list_indexes(name):
            info = client.describe_index(name, index_name)
            print(f"  - {index_name} on field '{info.get('field_name')}': "
                  f"type={info.get('index_type')} metric={info.get('metric_type')}")

        # Row count
        stats = client.get_collection_stats(name)
        print(f"\nRow count: {stats['row_count']}")

        # Milvus Lite collections load lazily — make sure it's loaded before querying
        client.load_collection(name)

        # Sample rows — query() needs a filter; this pulls everything back
        # (fine for small demo collections; add a real filter for big ones)
        output_fields = [f["name"] for f in desc["fields"]]
        if not vectors:
            output_fields = [f for f in output_fields if f not in ("dense", "sparse")]
        flt = f"doc_id == '{doc}'" if doc else "id >= 0"
        rows = client.query(name, filter=flt, output_fields=output_fields, limit=limit)
        print(f"\nRows (showing up to {limit}{', full text' if full else ''}{', with vectors' if vectors else ''}):")
        for row in rows:
            # Collapse newlines/extra spaces: image/video descriptions from
            # Gemini are multi-line, which would otherwise break the one-row-per-line layout.
            flat = " ".join(row.get("text", "").split())
            text_out = flat if full else ((flat[:70] + "...") if flat else "")
            modality = row.get("modality", "")
            source = row.get("source", "")
            print(f"  [{row.get('doc_id', row.get('id'))}] (id={row.get('id')} {modality}:{source})")
            print(f"      text  : {text_out}")
            if vectors:
                print_vectors(row)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None,
                        help="Milvus Lite .db path or server URI (default: $MILVUS_ADDRESS, else ./hybrid_rag.db)")
    parser.add_argument("--limit", type=int, default=10, help="Max sample rows to print per collection")
    parser.add_argument("--full", action="store_true", help="Print the full text field instead of a preview")
    parser.add_argument("--vectors", action="store_true", help="Also print dense + sparse vectors (the index data)")
    parser.add_argument("--doc", help="Only show the row with this doc_id, e.g. --doc doc5")
    args = parser.parse_args()
    inspect(args.db, args.limit, full=args.full, vectors=args.vectors, doc=args.doc)
