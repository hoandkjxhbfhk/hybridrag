#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore


RunDict = Dict[str, Dict[str, List[Tuple[str, float]]]]  # run_name -> qid -> [(docid, score)]


def read_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def compute_kmeans(doc_matrix: np.ndarray, k: int, seed: int = 42) -> (np.ndarray, np.ndarray):
    k = int(max(1, min(k, doc_matrix.shape[0])))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(doc_matrix)
    centers = km.cluster_centers_.astype(np.float32)
    return centers, labels


def mor_pre_score(vec: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    # score = sum(size_prop / (eps + distance)) — càng gần cụm to càng cao
    k = centers.shape[0]
    size = np.bincount(labels, minlength=k).astype(np.float32)
    size_prop = size / (size.sum() + 1e-12)
    dists = np.linalg.norm(centers - vec[None, :], axis=1)
    scores = size_prop / (1e-6 + dists)
    return float(scores.sum())


def read_runs(run_root: Path) -> RunDict:
    runs: RunDict = {}
    for sub in run_root.rglob("test.txt"):
        run_name = sub.parent.name
        per_q: Dict[str, List[Tuple[str, float]]] = {}
        with sub.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                qid, _, docid, rank, score, _ = parts
                per_q.setdefault(qid, []).append((docid, float(score)))
        for qid in per_q:
            per_q[qid].sort(key=lambda x: x[1], reverse=True)
        runs[run_name] = per_q
    return runs


def morans_I(values: np.ndarray, W: np.ndarray) -> float:
    # values: shape (K,), W: adjacency shape (K,K), zero diagonal, non-negative
    n = values.shape[0]
    if n <= 1:
        return 0.0
    W_sum = float(W.sum())
    if W_sum <= 1e-12:
        return 0.0
    x = values.astype(np.float64)
    x_bar = float(x.mean())
    num = 0.0
    for i in range(n):
        for j in range(n):
            num += W[i, j] * (x[i] - x_bar) * (x[j] - x_bar)
    den = float(((x - x_bar) ** 2).sum()) + 1e-12
    I = (n / W_sum) * (num / den)
    return float(I)


def compute_mor_post_weights(
    runs: RunDict,
    qids: List[str],
    q_pre_scores: Dict[str, float],
    doc_matrix: np.ndarray,
    docid_to_idx: Dict[str, int],
    centers: np.ndarray,
    labels: np.ndarray,
    topk: int,
    a: float,
    b: float,
    c: float,
) -> Dict[str, Dict[str, float]]:
    # Tính s_run(q) = a*V_pre(q) + b*I_Moran(run,q) + c*V_post(run,q)
    # Sau đó chuẩn hoá theo từng qid: w_run(q) = s_run / sum_s
    run_names = list(runs.keys())
    weights: Dict[str, Dict[str, float]] = {rn: {} for rn in run_names}

    for qid in qids:
        scores_per_run: Dict[str, float] = {}
        for rn in run_names:
            pairs = runs[rn].get(qid, [])[:topk]
            if not pairs:
                scores_per_run[rn] = 0.0
                continue
            idxs = [docid_to_idx.get(docid, -1) for docid, _ in pairs]
            idxs = [i for i in idxs if i >= 0]
            if not idxs:
                scores_per_run[rn] = 0.0
                continue
            D = doc_matrix[idxs]  # (K, dim), đã L2 normalized
            # ma trận tương đồng cosine làm trọng số kề
            S = (D @ D.T).astype(np.float32)
            np.fill_diagonal(S, 0.0)
            # local density vector L_i = mean_j S_ij
            if D.shape[0] > 1:
                L = S.sum(axis=1) / float(D.shape[0] - 1)
            else:
                L = np.zeros((1,), dtype=np.float32)
            I_moran = morans_I(L, S)
            # V_post: trung bình V_pre(d_n)
            V_pres = [mor_pre_score(vec, centers, labels) for vec in D]
            V_post = float(np.mean(V_pres)) if V_pres else 0.0
            V_pre_q = float(q_pre_scores.get(qid, 0.0))
            s = a * V_pre_q + b * I_moran + c * V_post
            scores_per_run[rn] = max(0.0, s)
        total = sum(scores_per_run.values())
        if total <= 1e-12:
            # gán đều
            for rn in run_names:
                weights[rn][qid] = 1.0 / max(1, len(run_names))
        else:
            for rn in run_names:
                weights[rn][qid] = float(scores_per_run[rn] / total)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MoR-pre/post weights and save JSON")
    parser.add_argument("--queries", type=str, default="data/beir/queries.jsonl")
    parser.add_argument("--indices", type=str, default="indices")
    parser.add_argument("--runs", type=str, default="runs")
    parser.add_argument("--out", type=str, default="weights")
    parser.add_argument("--kmeans-k", type=int, default=64)
    parser.add_argument("--topk", type=int, default=20, help="Top-K docs per run for MoR-post")
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--b", type=float, default=0.3)
    parser.add_argument("--c", type=float, default=0.6)
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--mode", type=str, default="both", choices=["pre", "post", "both"])
    args = parser.parse_args()

    queries_path = Path(args.queries)
    idx_base = Path(args.indices)
    run_root = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dense index (mpnet, d)
    dense_dir = idx_base / "dense_mpnet_d"
    if not dense_dir.exists():
        raise SystemExit(f"Dense index not found at {dense_dir}")
    index = faiss.read_index(str(dense_dir / "index.faiss"))
    ntotal = index.ntotal
    dim = index.d
    # reconstruct all vectors
    doc_matrix = np.zeros((ntotal, dim), dtype=np.float32)
    for start in range(0, ntotal, 1024):
        end = min(ntotal, start + 1024)
        chunk = np.vstack([index.reconstruct(i) for i in range(start, end)])
        doc_matrix[start:end] = chunk
    doc_matrix = l2_normalize(doc_matrix)

    # id mapping
    ids = json.loads((dense_dir / "ids.json").read_text(encoding="utf-8"))
    docid_to_idx: Dict[str, int] = {docid: i for i, docid in enumerate(ids)}

    # KMeans clusters on corpus vectors
    centers, labels = compute_kmeans(doc_matrix, k=args.kmeans_k, seed=42)

    # Encode queries and compute V_pre(q)
    model = SentenceTransformer(args.model)
    q_items = read_jsonl(queries_path)
    q_texts = [q.get("text", "") for q in q_items]
    qids = [q.get("qid") or q.get("id") for q in q_items]
    q_matrix = model.encode(q_texts, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    q_matrix = l2_normalize(q_matrix)
    q_pre_scores: Dict[str, float] = {}
    for qid, qv in zip(qids, q_matrix):
        q_pre_scores[qid] = mor_pre_score(qv, centers, labels)

    if args.mode in ("pre", "both"):
        # Convert to weights per run (dense vs bm25) bằng đối ngẫu 1-w
        weights_pre: Dict[str, Dict[str, float]] = {"dense_mpnet_d": {}, "bm25_d": {}}
        # Normalize across queries với cặp 2 run
        scores = np.array([q_pre_scores[qid] for qid in qids], dtype=np.float32)
        mn, mx = float(scores.min()), float(scores.max())
        denom = (mx - mn) if mx > mn else 1.0
        norm_scores = (scores - mn) / denom
        for qid, s in zip(qids, norm_scores):
            wd = float(np.clip(s, 0.2, 0.8))  # clip để tránh cực trị
            wb = float(1.0 - wd)
            weights_pre["dense_mpnet_d"][qid] = wd
            weights_pre["bm25_d"][qid] = wb
        (out_dir / "mor_pre.json").write_text(json.dumps(weights_pre, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote MoR-pre weights to {out_dir / 'mor_pre.json'}")

    if args.mode in ("post", "both"):
        runs = read_runs(run_root)
        weights_post = compute_mor_post_weights(
            runs=runs,
            qids=qids,
            q_pre_scores=q_pre_scores,
            doc_matrix=doc_matrix,
            docid_to_idx=docid_to_idx,
            centers=centers,
            labels=labels,
            topk=args.topk,
            a=args.a,
            b=args.b,
            c=args.c,
        )
        (out_dir / "mor_post.json").write_text(json.dumps(weights_post, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote MoR-post weights to {out_dir / 'mor_post.json'}")


if __name__ == "__main__":
    main()
