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


def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Stable softmax với nhiệt độ tau cho vector 1D.

    Nếu tổng mũ bị underflow, trả về phân phối đều.
    """
    if x.ndim != 1:
        x = x.reshape(-1)
    t = float(max(tau, 1e-12))
    z = (x.astype(np.float64) / t)
    z = z - float(np.max(z))
    e = np.exp(z)
    s = float(np.sum(e))
    if s <= 1e-12:
        n = e.size if e.size > 0 else 1
        return np.ones((n,), dtype=np.float64) / float(n)
    return (e / s).astype(np.float64)


def compute_entropy_weights(
    runs: RunDict,
    qids: List[str],
    topk: int,
    tau: float,
) -> Dict[str, Dict[str, float]]:
    """Tính trọng số entropy cho mỗi run và mỗi truy vấn.

    - Với từng (run, qid): lấy top-K điểm số → p = softmax(scores / tau) → H = -Σ p log p.
    - Với mỗi qid: w_run(q) = softmax_theo_run(-H_run(q)) đảm bảo tổng = 1.
    """
    run_names = sorted(runs.keys())

    entropies: Dict[str, Dict[str, float]] = {rn: {} for rn in run_names}
    for rn in run_names:
        per_q = runs.get(rn, {})
        for qid in qids:
            pairs = per_q.get(qid, [])[:topk]
            if not pairs:
                entropies[rn][qid] = 1e6
                continue
            scores = np.array([float(s) for _, s in pairs], dtype=np.float64)
            probs = softmax(scores, tau=tau)
            H = float(-np.sum(probs * np.log(probs + 1e-12)))
            entropies[rn][qid] = H

    weights: Dict[str, Dict[str, float]] = {rn: {} for rn in run_names}
    for qid in qids:
        Hs = np.array([entropies[rn].get(qid, 1e6) for rn in run_names], dtype=np.float64)
        logits = -Hs
        m = float(np.max(logits))
        exps = np.exp(logits - m)
        denom = float(np.sum(exps))
        if denom <= 1e-12:
            w = np.ones_like(exps) / float(exps.size if exps.size else 1)
        else:
            w = exps / denom
        for rn, val in zip(run_names, w.tolist()):
            weights[rn][qid] = float(val)

    return weights


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


def compute_mor_pre_weights_all(
    runs: RunDict,
    qids: List[str],
    q_texts: List[str],
    indices_base: Path,
    kmeans_k: int,
) -> Dict[str, Dict[str, float]]:
    """Tính MoR-pre cho MỌI run dense theo không gian của CHÍNH run đó.

    - Với mỗi run dense: KMeans trên doc embedding; encode queries bằng model tương ứng; tính V_pre(q) → min-max per-run → weight [0.2, 0.8].
    - BM25: w_bm25(q) = 1 − mean(w_dense_runs(q)). Nếu nhiều BM25 runs → chia đều.
    """

    def is_bm25_run(name: str) -> bool:
        return "bm25" in name.lower()

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
            return "sentence-transformers/msmarco-bert-base-dot-v5"
        if a in ("simcse", "simcse-bert-base", "unsup-simcse-bert-base"):
            return "princeton-nlp/sup-simcse-roberta-base"
        return alias

    def dense_alias_from_run(run_name: str) -> str:
        if run_name.startswith("dense_"):
            part = run_name[len("dense_"):]
            if "_" in part:
                return part.rsplit("_", 1)[0]
            return part
        return run_name

    run_names = sorted(runs.keys())
    bm25_runs = [rn for rn in run_names if is_bm25_run(rn)]
    dense_runs = [rn for rn in run_names if not is_bm25_run(rn) and rn.startswith("dense_")]

    weights: Dict[str, Dict[str, float]] = {rn: {} for rn in run_names}

    # Với từng dense run: tính V_pre(q) dựa trên không gian riêng của run
    for rn in dense_runs:
        idx_dir = indices_base / rn
        if not idx_dir.exists():
            # Nếu không có index, coi như w=0
            for qid in qids:
                weights[rn][qid] = 0.0
            continue

        # reconstruct doc vectors
        index = faiss.read_index(str(idx_dir / "index.faiss"))
        ntotal = index.ntotal
        dim = index.d
        doc_matrix = np.zeros((ntotal, dim), dtype=np.float32)
        for start in range(0, ntotal, 1024):
            end = min(ntotal, start + 1024)
            chunk = np.vstack([index.reconstruct(i) for i in range(start, end)])
            doc_matrix[start:end] = chunk
        doc_matrix = l2_normalize(doc_matrix)

        # KMeans trong không gian của run
        centers, labels = compute_kmeans(doc_matrix, k=int(max(1, min(kmeans_k, ntotal))), seed=42)

        # Encode queries theo model của run
        alias = dense_alias_from_run(rn)
        model_name = get_model_name(alias)
        model = SentenceTransformer(model_name)
        q_matrix = model.encode(q_texts, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
        q_matrix = l2_normalize(q_matrix)

        # Tính điểm V_pre cho mọi qid, rồi min-max per-run → weight
        vals: List[float] = []
        per_q_score: Dict[str, float] = {}
        for qid, qv in zip(qids, q_matrix):
            sc = mor_pre_score(qv, centers, labels)
            per_q_score[qid] = float(sc)
            vals.append(float(sc))
        if vals:
            mn, mx = float(np.min(vals)), float(np.max(vals))
            denom = (mx - mn) if mx > mn else 1.0
            for qid in qids:
                norm = (per_q_score.get(qid, 0.0) - mn) / denom
                weights[rn][qid] = float(np.clip(norm, 0.001, 0.99))
        else:
            for qid in qids:
                weights[rn][qid] = 0.0

    # BM25: 1 - mean(weights của tất cả dense runs)
    for qid in qids:
        dense_vals = [weights[rn].get(qid, 0.0) for rn in dense_runs]
        if dense_vals:
            other_mean = float(np.mean(dense_vals))
            bm25_share = max(0.0, 1.0 - other_mean)
        else:
            bm25_share = 1.0
        if bm25_runs:
            per_bm25 = bm25_share / float(len(bm25_runs))
            for rn in bm25_runs:
                weights[rn][qid] = per_bm25

    return weights

def compute_mor_post_weights(
    runs: RunDict,
    qids: List[str],
    q_texts: List[str],
    indices_base: Path,
    kmeans_k: int,
    topk: int,
    a: float,
    b: float,
    c: float,
) -> Dict[str, Dict[str, float]]:
    """Tính MoR-post cho tất cả run dense theo không gian của CHÍNH run đó.

    - Với mỗi run dense (ví dụ: dense_mpnet_d, dense_contriever_d, ...):
      * Load index FAISS và ids.
      * KMeans trên doc embedding của run để có (centers, labels).
      * Encode queries bằng model tương ứng, tính V_pre(q) theo run.
      * Với mỗi qid: lấy top-K doc từ run, tạo ma trận S, tính I_moran và V_post.
      * s_run(q) = a*V_pre(q) + b*I_moran + c*V_post.
      * Chuẩn hoá min-max theo từng run → w_run(q) (clip [0.2, 0.8]).
    - BM25: w_bm25(q) = 1 - mean(w_others(q)). Nếu nhiều BM25 runs → chia đều.
    """

    def is_bm25_run(name: str) -> bool:
        n = name.lower()
        return "bm25" in n

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
            return "sentence-transformers/msmarco-bert-base-dot-v5"
        if a in ("simcse", "simcse-bert-base", "unsup-simcse-bert-base"):
            return "princeton-nlp/sup-simcse-roberta-base"
        return alias

    def dense_alias_from_run(run_name: str) -> str:
        # run_name ví dụ: dense_mpnet_d → alias: mpnet
        name = run_name
        if name.startswith("dense_"):
            part = name[len("dense_"):]
            if "_" in part:
                alias = part.rsplit("_", 1)[0]
            else:
                alias = part
            return alias
        return name

    run_names = sorted(runs.keys())
    bm25_runs = [rn for rn in run_names if is_bm25_run(rn)]
    dense_runs = [rn for rn in run_names if not is_bm25_run(rn) and rn.startswith("dense_")]

    # 1) Tính s_run(q) cho các run dense theo không gian của CHÍNH run đó
    scores_per_run: Dict[str, Dict[str, float]] = {rn: {} for rn in dense_runs}

    for rn in dense_runs:
        idx_dir = indices_base / rn
        if not idx_dir.exists():
            # không tìm thấy index tương ứng → cho s=0 cho mọi qid
            for qid in qids:
                scores_per_run[rn][qid] = 0.0
            continue

        # Load doc embeddings
        index = faiss.read_index(str(idx_dir / "index.faiss"))
        ntotal = index.ntotal
        dim = index.d
        doc_matrix = np.zeros((ntotal, dim), dtype=np.float32)
        for start in range(0, ntotal, 1024):
            end = min(ntotal, start + 1024)
            chunk = np.vstack([index.reconstruct(i) for i in range(start, end)])
            doc_matrix[start:end] = chunk
        doc_matrix = l2_normalize(doc_matrix)

        ids = json.loads((idx_dir / "ids.json").read_text(encoding="utf-8"))
        docid_to_idx: Dict[str, int] = {docid: i for i, docid in enumerate(ids)}

        # KMeans trên không gian của chính run
        centers, labels = compute_kmeans(doc_matrix, k=int(max(1, min(kmeans_k, ntotal))), seed=42)

        # Encode queries theo model của run để tính V_pre(q)
        alias = dense_alias_from_run(rn)
        model_name = get_model_name(alias)
        model = SentenceTransformer(model_name)
        q_matrix = model.encode(q_texts, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
        q_matrix = l2_normalize(q_matrix)
        q_pre_scores_rn: Dict[str, float] = {}
        for qid, qv in zip(qids, q_matrix):
            q_pre_scores_rn[qid] = mor_pre_score(qv, centers, labels)

        # Tính s_run(q) cho từng qid dùng top-K của RN
        for qid in qids:
            pairs = runs[rn].get(qid, [])[:topk]
            if not pairs:
                scores_per_run[rn][qid] = 0.0
                continue
            idxs = [docid_to_idx.get(docid, -1) for docid, _ in pairs]
            idxs = [i for i in idxs if i >= 0]
            if not idxs:
                scores_per_run[rn][qid] = 0.0
                continue
            D = doc_matrix[idxs]
            S = (D @ D.T).astype(np.float32)
            np.fill_diagonal(S, 0.0)
            if D.shape[0] > 1:
                L = S.sum(axis=1) / float(D.shape[0] - 1)
            else:
                L = np.zeros((1,), dtype=np.float32)
            I_moran = morans_I(L, S)
            V_pres = [mor_pre_score(vec, centers, labels) for vec in D]
            V_post = float(np.mean(V_pres)) if V_pres else 0.0
            V_pre_q = float(q_pre_scores_rn.get(qid, 0.0))
            s = a * V_pre_q + b * I_moran + c * V_post
            scores_per_run[rn][qid] = max(0.0, float(s))

    # 2) Min-max từng run dense → weight trong [0.2, 0.8]
    weights: Dict[str, Dict[str, float]] = {rn: {} for rn in run_names}
    for rn in dense_runs:
        vals = np.array([scores_per_run[rn].get(qid, 0.0) for qid in qids], dtype=np.float32)
        if vals.size == 0:
            for qid in qids:
                weights[rn][qid] = 0.0
            continue
        mn, mx = float(vals.min()), float(vals.max())
        denom = (mx - mn) if mx > mn else 1.0
        for qid in qids:
            norm = (scores_per_run[rn].get(qid, 0.0) - mn) / denom
            weights[rn][qid] = float(np.clip(norm, 0.01, 0.99))

    # 3) BM25 là phần bù: 1 - mean(weight các run dense)
    for qid in qids:
        other_vals = [weights[rn].get(qid, 0.0) for rn in dense_runs]
        if other_vals:
            other_mean = float(np.mean(other_vals))
            bm25_share = max(0.0, 1.0 - other_mean)
        else:
            bm25_share = 1.0
        if bm25_runs:
            per_bm25 = bm25_share / float(len(bm25_runs))
            for rn in bm25_runs:
                weights[rn][qid] = per_bm25

    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MoR-pre/post/entropy weights and save JSON")
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
    parser.add_argument("--mode", type=str, default="all", choices=["pre", "post", "entropy", "both", "all"])
    parser.add_argument("--entropy-topk", type=int, default=10, help="Top-K docs for entropy calculation")
    parser.add_argument("--entropy-tau", type=float, default=1.0, help="Temperature for softmax in entropy calculation")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    idx_base = Path(args.indices)
    run_root = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chuẩn bị queries
    q_items = read_jsonl(queries_path)
    q_texts = [q.get("text", "") for q in q_items]
    qids = [q.get("qid") or q.get("id") for q in q_items]

    run_in_all = args.mode == "all"

    if args.mode in ("pre", "both") or run_in_all:
        runs = read_runs(run_root)
        weights_pre = compute_mor_pre_weights_all(
            runs=runs,
            qids=qids,
            q_texts=q_texts,
            indices_base=idx_base,
            kmeans_k=args.kmeans_k,
        )
        (out_dir / "mor_pre.json").write_text(json.dumps(weights_pre, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote MoR-pre weights to {out_dir / 'mor_pre.json'}")

    if args.mode in ("post", "both") or run_in_all:
        runs = read_runs(run_root)
        weights_post = compute_mor_post_weights(
            runs=runs,
            qids=qids,
            q_texts=q_texts,
            indices_base=idx_base,
            kmeans_k=args.kmeans_k,
            topk=args.topk,
            a=args.a,
            b=args.b,
            c=args.c,
        )
        (out_dir / "mor_post.json").write_text(json.dumps(weights_post, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote MoR-post weights to {out_dir / 'mor_post.json'}")

    if args.mode in ("entropy",) or run_in_all:
        runs = read_runs(run_root)
        weights_entropy = compute_entropy_weights(
            runs=runs,
            qids=qids,
            topk=args.entropy_topk,
            tau=args.entropy_tau,
        )
        (out_dir / "entropy_weights.json").write_text(json.dumps(weights_entropy, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote Entropy weights to {out_dir / 'entropy_weights.json'}")


if __name__ == "__main__":
    main()
