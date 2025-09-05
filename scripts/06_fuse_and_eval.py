#!/usr/bin/env python3

import argparse
import collections
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


RunRow = Tuple[str, str, int, float, str]  # qid, docid, rank, score, run


def read_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    rel: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            qid, docid, score = line.strip().split("\t")
            rel[qid][docid] = int(score)
    return rel


def read_run_file(path: Path) -> Dict[str, List[Tuple[str, float]]]:
    per_q: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            qid, _, docid, rank, score, _ = parts
            per_q[qid].append((docid, float(score)))
    # sort by score desc, keep best rank
    for qid in per_q:
        per_q[qid].sort(key=lambda x: x[1], reverse=True)
    return per_q


def collect_runs(run_root: Path) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    runs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    for sub in run_root.rglob("test.txt"):
        run_name = sub.parent.name
        runs[run_name] = read_run_file(sub)
    return runs


def normalized_sum_fusion(runs: Dict[str, Dict[str, List[Tuple[str, float]]]], topk: int) -> Dict[str, List[Tuple[str, float]]]:
    fused: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    for run_name, per_q in runs.items():
        for qid, pairs in per_q.items():
            if not pairs:
                continue
            scores = np.array([s for _, s in pairs], dtype=np.float32)
            if scores.size == 0:
                continue
            mn, mx = float(scores.min()), float(scores.max())
            denom = (mx - mn) if mx > mn else 1.0
            for docid, s in pairs:
                fused[qid][docid] += (float(s) - mn) / denom
    out: Dict[str, List[Tuple[str, float]]] = {}
    for qid, doc_scores in fused.items():
        items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        out[qid] = items
    return out


def rrf_fusion(runs: Dict[str, Dict[str, List[Tuple[str, float]]]], topk: int, k: int = 60) -> Dict[str, List[Tuple[str, float]]]:
    fused: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    for run_name, per_q in runs.items():
        for qid, pairs in per_q.items():
            for rank, (docid, _) in enumerate(pairs, start=1):
                fused[qid][docid] += 1.0 / (k + rank)
    out: Dict[str, List[Tuple[str, float]]] = {}
    for qid, doc_scores in fused.items():
        items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        out[qid] = items
    return out


def dcg(scores: List[int]) -> float:
    return sum((rel / np.log2(idx + 2)) for idx, rel in enumerate(scores))


def ndcg_at_k(true_rel: Dict[str, int], ranked_docs: List[str], k: int) -> float:
    gains = [true_rel.get(docid, 0) for docid in ranked_docs[:k]]
    ideal = sorted(true_rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(gains) / idcg) if idcg > 0 else 0.0


def evaluate_ndcg(qrels: Dict[str, Dict[str, int]], fused: Dict[str, List[Tuple[str, float]]], k_list=(5, 20)) -> Dict[int, float]:
    agg: Dict[int, List[float]] = {k: [] for k in k_list}
    for qid, pairs in fused.items():
        ranked = [docid for docid, _ in pairs]
        for k in k_list:
            agg[k].append(ndcg_at_k(qrels.get(qid, {}), ranked, k))
    return {k: (float(np.mean(v)) if v else 0.0) for k, v in agg.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse run files and evaluate NDCG@5/20")
    parser.add_argument("--qrels", type=str, required=True)
    parser.add_argument("--runs", type=str, required=True)
    parser.add_argument("--fusion", nargs="+", default=["normalized_sum", "rrf"])
    parser.add_argument("--weights", type=str, default="weights")
    parser.add_argument("--out", type=str, default="fusion")
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()

    qrels = read_qrels(Path(args.qrels))
    runs = collect_runs(Path(args.runs))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in args.fusion:
        if method == "normalized_sum":
            fused = normalized_sum_fusion(runs, topk=args.topk)
        elif method == "rrf":
            fused = rrf_fusion(runs, topk=args.topk)
        else:
            # placeholder cho weighted_sum_mor_pre/post
            fused = normalized_sum_fusion(runs, topk=args.topk)

        # ghi run hợp nhất
        run_rows: List[RunRow] = []
        for qid, pairs in fused.items():
            for rank, (docid, score) in enumerate(pairs, start=1):
                run_rows.append((qid, docid, rank, float(score), f"fuse_{method}"))
        trec_path = out_dir / f"fused_{method}.txt"
        with trec_path.open("w", encoding="utf-8") as f:
            for qid, docid, rank, score, runname in run_rows:
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {runname}\n")

        ndcgs = evaluate_ndcg(qrels, fused, k_list=(5, 20))
        print(f"{method}: Average NDCG@5={ndcgs[5]:.4f}, NDCG@20={ndcgs[20]:.4f}")


if __name__ == "__main__":
    main()
