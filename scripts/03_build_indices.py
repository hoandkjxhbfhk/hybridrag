#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# BM25 (Whoosh)
from whoosh import index as windex
from whoosh.fields import ID, TEXT, Schema
from whoosh.qparser import QueryParser


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def split_by_granularity(records: Iterable[Dict]) -> Tuple[List[Dict], List[Dict]]:
    d_docs: List[Dict] = []
    p_docs: List[Dict] = []
    for rec in records:
        pid = rec.get("_id") or rec.get("id")
        if pid is None:
            continue
        # Quy ước: proposition có hậu tố "-p{k}" trong id
        if "-p" in pid:
            p_docs.append(rec)
        else:
            d_docs.append(rec)
    # Nếu không có proposition, coi tất cả là d
    if not p_docs and not d_docs:
        return [], []
    return d_docs, p_docs


def build_bm25_index(docs: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = Schema(
        doc_id=ID(unique=True, stored=True),
        title=TEXT(stored=True),
        text=TEXT(stored=True),
    )
    if not windex.exists_in(str(out_dir)):
        idx = windex.create_in(str(out_dir), schema)
    else:
        idx = windex.open_dir(str(out_dir))

    writer = idx.writer(limitmb=512)
    for rec in docs:
        writer.add_document(
            doc_id=str(rec.get("_id") or rec.get("id")),
            title=str(rec.get("title", "")),
            text=str(rec.get("text", "")),
        )
    writer.commit()

    # Lưu meta
    (out_dir / "meta.json").write_text(json.dumps({
        "retriever": "bm25",
        "docs": len(docs),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def build_dense_index(docs: List[Dict], out_dir: Path, model_name: str, batch_size: int = 64) -> None:
    # Lazy import để tránh yêu cầu khi không dùng dense
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer

    out_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)

    texts = [str(rec.get("text", "")) for rec in docs]
    ids = [str(rec.get("_id")) for rec in docs]

    embeddings: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
        embeddings.append(vecs.astype(np.float32))
    if embeddings:
        matrix = np.vstack(embeddings)
    else:
        matrix = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    matrix = l2_normalize(matrix)

    dim = matrix.shape[1] if matrix.size else model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    if matrix.size:
        index.add(matrix)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({
        "retriever": "dense",
        "model": model_name,
        "dim": dim,
        "docs": len(docs),
        "normalized": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_retrievers(s: str) -> List[str]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    # Hỗ trợ: bm25, simcse, contriever, dpr, ance, tas-b, mpnet, gtr
    return items


def get_model_name(alias: str) -> str:
    """Map shorthand alias to a SentenceTransformers-compatible model name.

    Notes:
    - We prefer well-known ST repos for stability.
    - Some families (e.g., DPR/ANCE) are approximated by strong MS MARCO dot-product models.
    - Users can always pass a full HF repo name to override.
    """
    a = alias.lower()
    # BM25 is handled separately
    if a in ("mpnet", "all-mpnet-base-v2", "st-mpnet"):
        return "sentence-transformers/all-mpnet-base-v2"
    if a in ("contriever", "facebook/contriever"):
        return "facebook/contriever"
    if a in ("gtr", "gtr-base", "gtr-t5-base"):
        return "sentence-transformers/gtr-t5-base"
    if a in ("gtr-large", "gtr-t5-large"):
        return "sentence-transformers/gtr-t5-large"
    if a in ("tas-b", "tasb", "msmarco-distilbert-base-tas-b"):
        return "sentence-transformers/msmarco-distilbert-base-tas-b"
    if a in ("dpr", "msmarco-dot", "msmarco-bert-base-dot-v5"):
        return "sentence-transformers/msmarco-bert-base-dot-v5"
    if a in ("ance", "msmarco-ance"):
        # If exact ANCE model is unavailable, fall back to a strong MS MARCO dot model
        return "sentence-transformers/msmarco-bert-base-dot-v5"
    if a in ("simcse", "simcse-bert-base", "unsup-simcse-bert-base"):
        # Prefer a supervised SimCSE for sentence-level retrieval
        return "princeton-nlp/sup-simcse-roberta-base"
    # Otherwise assume it's already a full HF repo id
    return alias


def main() -> None:
    parser = argparse.ArgumentParser(description="Build indices for BM25 (Whoosh) and Dense (FAISS)")
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--out", type=str, default="indices")
    parser.add_argument("--retrievers", type=str, default="bm25,mpnet,contriever")
    parser.add_argument("--gran", type=str, default="qd,qp,sqd,sqp", help="Used only to mirror CLI; indexing decides d/p based on ids")
    parser.add_argument("--dense-batch", type=int, default=64)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    out_base = Path(args.out)

    records = list(read_jsonl(corpus_path))
    d_docs, p_docs = split_by_granularity(records)

    retrievers = parse_retrievers(args.retrievers)

    for r in retrievers:
        if r.lower() == "bm25":
            if d_docs:
                build_bm25_index(d_docs, out_base / "bm25_d")
            if p_docs:
                build_bm25_index(p_docs, out_base / "bm25_p")
        else:
            # Dense retriever
            model_name = get_model_name(r)
            tag = r.lower().replace("/", "_")
            if d_docs:
                build_dense_index(d_docs, out_base / f"dense_{tag}_d", model_name, batch_size=args.dense_batch)
            if p_docs:
                build_dense_index(p_docs, out_base / f"dense_{tag}_p", model_name, batch_size=args.dense_batch)

    print("Finished building indices.")


if __name__ == "__main__":
    main()
