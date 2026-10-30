"""Measure retrieval quality: hit-rate@k on sample_data/eval_queries.json.

Usage:  python scripts/eval_retrieval.py --backend tfidf --k 3
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examprep.ingestion import load_file  # noqa: E402
from examprep.retriever import Retriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="tfidf | sbert | auto")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    notes = ROOT / "sample_data" / "sample_notes.txt"
    retriever = Retriever.from_pages(load_file(notes.name, notes.read_bytes()), args.backend)
    queries = json.loads((ROOT / "sample_data" / "eval_queries.json").read_text(encoding="utf-8"))

    hits_found = 0
    for item in queries:
        hits = retriever.retrieve(item["query"], k=args.k)
        ok = any(item["expect"].lower() in h.chunk.text.lower() for h in hits)
        hits_found += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {item['query']!r} -> {[h.chunk.citation for h in hits]}")
    print(f"\nHit-rate@{args.k} ({retriever.embedder.name}): {hits_found}/{len(queries)} = {hits_found / len(queries):.0%}")


if __name__ == "__main__":
    main()
