#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# BM25 (Whoosh)
from whoosh import index as windex
from whoosh.qparser import QueryParser, OrGroup


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def trec_write(path: Path, rows: List[Tuple[str, str, int, float, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for qid, docid, rank, score, runname in rows:
            f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {runname}\n")


def retrieve_bm25(index_dir: Path, query_text: str, topk: int) -> List[Tuple[str, float]]:
    idx = windex.open_dir(str(index_dir))
    with idx.searcher() as searcher:
        parser = QueryParser("text", schema=searcher.schema, group=OrGroup.factory(0.15))
        q = parser.parse(query_text)
        results = searcher.search(q, limit=topk)
        pairs: List[Tuple[str, float]] = []
        for hit in results:
            pairs.append((hit["doc_id"], float(hit.score)))
        return pairs


def retrieve_dense(index_dir: Path, model_alias: str, query_text: str, topk: int) -> List[Tuple[str, float]]:
    # Lazy imports để tránh lỗi khi không dùng dense
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(index_dir / "index.faiss"))
    ids = json.loads((index_dir / "ids.json").read_text(encoding="utf-8"))

    def get_model_name(alias: str) -> str:
        a = alias.lower()
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
            return "sentence-transformers/msmarco-roberta-base-ance-firstp"
        if a in ("simcse", "simcse-bert-base", "unsup-simcse-bert-base"):
            return "princeton-nlp/sup-simcse-roberta-base"
        return alias

    model_name = get_model_name(model_alias)

    model = SentenceTransformer(model_name)

    q_vec = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=False)
    q_vec = q_vec.astype(np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)

    scores, idxs = index.search(q_vec, topk)
    res: List[Tuple[str, float]] = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        res.append((ids[i], float(score)))
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve from built indices and write TREC runs")
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--indices", type=str, required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument(
        "--bm25-norm",
        type=str,
        default="none",
        choices=["none", "minmax", "zscore", "softmax", "max1"],
        help="Chuẩn hoá điểm BM25 theo truy vấn: none|minmax|zscore|softmax|max1",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries)
    idx_base = Path(args.indices)
    out_base = Path(args.out)

    available = [p for p in idx_base.iterdir() if p.is_dir()]

    # Xoá file run cũ để tránh cộng dồn qua các lần chạy
    for idx_dir in available:
        old_run = out_base / idx_dir.name / "test.txt"
        if old_run.exists():
            try:
                old_run.unlink()
            except Exception:
                pass

    for q in read_jsonl(queries_path):
        qid = q.get("qid") or q.get("id")
        qtext = q.get("text", "")
        if not qid or not qtext:
            continue

        for idx_dir in available:
            name = idx_dir.name
            runname = name
            if name.startswith("bm25_"):
                pairs = retrieve_bm25(idx_dir, qtext, args.topk)
                # Chuẩn hoá điểm theo truy vấn nếu được yêu cầu
                if args.bm25_norm and args.bm25_norm != "none":
                    scores = np.array([s for _, s in pairs], dtype=np.float32)
                    if scores.size:
                        method = args.bm25_norm
                        if method == "minmax":
                            s_min = float(scores.min())
                            s_max = float(scores.max())
                            if s_max > s_min:
                                norm = (scores - s_min) / (s_max - s_min)
                            else:
                                norm = np.ones_like(scores)
                        elif method == "zscore":
                            mean = float(scores.mean())
                            std = float(scores.std()) + 1e-12
                            norm = (scores - mean) / std
                        elif method == "softmax":
                            mx = float(scores.max())
                            expv = np.exp(scores - mx)
                            norm = expv / (float(expv.sum()) + 1e-12)
                        elif method == "max1":
                            m = float(scores.max())
                            if m > 0:
                                norm = scores / m
                            else:
                                norm = np.zeros_like(scores)
                        else:
                            norm = scores
                        pairs = [(docid, float(s)) for (docid, _), s in zip(pairs, norm)]
            elif name.startswith("dense_"):
                parts = name.split("_")
                if len(parts) < 3:
                    continue
                alias = parts[1]
                pairs = retrieve_dense(idx_dir, alias, qtext, args.topk)
            else:
                continue

            out_path = out_base / name / "test.txt"
            rows: List[Tuple[str, str, int, float, str]] = []
            for rank, (docid, score) in enumerate(pairs, start=1):
                rows.append((qid, docid, rank, score, runname))
            if rows:
                existing: List[Tuple[str, str, int, float, str]] = []
                if out_path.exists():
                    with out_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 6:
                                existing.append((parts[0], parts[2], int(parts[3]), float(parts[4]), parts[5]))
                rows = existing + rows
                trec_write(out_path, rows)

    print("Finished retrieval and wrote run files.")


if __name__ == "__main__":
    main()
