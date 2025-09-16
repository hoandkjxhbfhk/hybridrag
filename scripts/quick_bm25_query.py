#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, Tuple

from whoosh import index as windex
from whoosh.qparser import QueryParser, OrGroup


def retrieve_bm25(index_dir: Path, query_text: str, topk: int) -> List[Tuple[str, float]]:
    idx = windex.open_dir(str(index_dir))
    with idx.searcher() as searcher:
        parser = QueryParser("text", schema=searcher.schema, group=OrGroup.factory(0.15))

        q_raw = (query_text or "").strip()
        if not q_raw:
            return []

        def _search(qs: str) -> List[Tuple[str, float]]:
            q = parser.parse(qs)
            results = searcher.search(q, limit=topk)
            return [(hit["doc_id"], float(hit.score)) for hit in results]

        try:
            pairs = _search(q_raw)
        except Exception:
            pairs = []

        if not pairs:
            import re
            tokens = [t for t in re.findall(r"[A-Za-z0-9]+", q_raw.lower()) if len(t) >= 2][:32]
            if tokens:
                try:
                    pairs = _search(" ".join(tokens))
                except Exception:
                    pairs = []
        return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick BM25 query against Whoosh index")
    parser.add_argument("--index", type=str, default="/home/hoan/hybridrag/indices/bm25_d")
    parser.add_argument("--query", type=str, default="t")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    index_dir = Path(args.index)
    pairs = retrieve_bm25(index_dir, args.query, args.topk)

    if not pairs:
        print("No hits.")
        return
    for rank, (docid, score) in enumerate(pairs, start=1):
        print(f"{rank:2d}. {docid}\t{score:.6f}")


if __name__ == "__main__":
    main()


