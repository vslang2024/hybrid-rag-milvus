"""
Quick utility to open an existing Milvus Lite database and look at what's
inside it — collection schema, row count, and sample rows. Handy for
debugging ingestion without re-running the whole RAG pipeline.

Install:
    pip install "pymilvus[milvus_lite]"

Usage:
    python inspect_db.py                       # uses ./hybrid_rag.db
    python inspect_db.py --db path/to/other.db
    python inspect_db.py --limit 20             # show more sample rows
"""

import argparse
from pymilvus import MilvusClient, DataType


def inspect(db_path: str, limit: int):
    client = MilvusClient(db_path)

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

        # Row count
        stats = client.get_collection_stats(name)
        print(f"\nRow count: {stats['row_count']}")

        # Milvus Lite collections load lazily — make sure it's loaded before querying
        client.load_collection(name)

        # Sample rows — query() needs a filter; this pulls everything back
        # (fine for small demo collections; add a real filter for big ones)
        output_fields = [f["name"] for f in desc["fields"] if f["name"] != "sparse"]
        rows = client.query(name, filter="id >= 0", output_fields=output_fields, limit=limit)
        print(f"\nSample rows (showing up to {limit}):")
        for row in rows:
            text_preview = (row.get("text", "")[:70] + "...") if row.get("text") else ""
            modality = row.get("modality", "")
            source = row.get("source", "")
            print(f"  [{row.get('doc_id', row.get('id'))}] ({modality}:{source}) {text_preview}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="hybrid_rag.db", help="Path to the Milvus Lite .db file")
    parser.add_argument("--limit", type=int, default=10, help="Max sample rows to print per collection")
    args = parser.parse_args()
    inspect(args.db, args.limit)
