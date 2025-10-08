#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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


def visualize_clusters_2d(
    doc_matrix: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    method: str,
    out_path: Path,
    title: str,
    max_points: int = 5000,
    seed: int = 42,
    dbscan: bool = False,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    dbscan_max_points_per_cluster: Optional[int] = None,
) -> None:
    """Vẽ trực quan các cụm trong 2D bằng PCA hoặc t-SNE và lưu hình.

    - doc_matrix: (N, D) đã chuẩn hoá L2 (không bắt buộc)
    - labels: (N,) nhãn cụm
    - centers: (K, D) tâm cụm
    - method: "pca" hoặc "tsne"
    - out_path: đường dẫn ảnh đầu ra (.png)
    - max_points: giới hạn số điểm để vẽ (lấy mẫu nếu N lớn)
    - seed: random state
    """
    N = int(doc_matrix.shape[0])
    K = int(centers.shape[0]) if centers is not None else 0
    if N <= 0 or K <= 0:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Lấy mẫu theo tỉ lệ cụm để đảm bảo mỗi cụm đều có điểm
    rng = np.random.default_rng(seed)
    max_points = int(max(100, max_points))
    unique_labels = np.arange(K)
    counts = np.bincount(labels, minlength=K).astype(int)

    if N > max_points:
        # Phân bổ quota cho mỗi cụm theo tỉ lệ kích thước, tối thiểu 5 điểm/cụm nếu có đủ
        quotas = np.maximum(1, np.floor(max_points * (counts / float(N))).astype(int))
        # Điều chỉnh tổng quota về đúng max_points
        diff = max_points - int(quotas.sum())
        if diff > 0:
            # phân phát phần dư cho các cụm lớn nhất
            order = np.argsort(-counts)
            for i in order:
                if diff == 0:
                    break
                quotas[i] += 1
                diff -= 1
        elif diff < 0:
            order = np.argsort(counts)
            for i in order:
                if diff == 0:
                    break
                if quotas[i] > 1:
                    quotas[i] -= 1
                    diff += 1

        sample_indices: List[int] = []
        for c in unique_labels:
            idxs_c = np.where(labels == c)[0]
            if idxs_c.size == 0:
                continue
            take = int(min(quotas[c], idxs_c.size))
            chosen = rng.choice(idxs_c, size=take, replace=False)
            sample_indices.extend(chosen.tolist())
        if not sample_indices:
            sample_indices = rng.choice(np.arange(N), size=min(max_points, N), replace=False).tolist()
        sample_indices = np.array(sample_indices, dtype=int)
    else:
        sample_indices = np.arange(N, dtype=int)

    X = doc_matrix[sample_indices]
    y = labels[sample_indices]

    # Import lười để tránh phụ thuộc khi không cần vẽ
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    if method.lower() == "pca":
        from sklearn.decomposition import PCA  # type: ignore
        pca = PCA(n_components=2, random_state=seed)
        X2 = pca.fit_transform(X)
        centers2 = pca.transform(centers) if centers is not None else None
    else:
        from sklearn.manifold import TSNE  # type: ignore
        concat = np.vstack([X, centers]) if centers is not None else X
        # perplexity < n_samples/3, chọn tự động dựa trên cỡ mẫu
        n_s = int(concat.shape[0])
        if n_s <= 5:
            perplexity = max(2, n_s - 1)
        else:
            perplexity = max(5, min(30, (n_s - 1) // 3))
        try:
            tsne = TSNE(
                n_components=2,
                perplexity=float(perplexity),
                learning_rate="auto",
                init="pca",
                n_iter=1000,
                random_state=seed,
                verbose=0,
            )
        except TypeError:
            # Tương thích phiên bản scikit-learn cũ (không hỗ trợ learning_rate="auto" hoặc n_iter trong __init__)
            tsne = TSNE(
                n_components=2,
                perplexity=float(perplexity),
                learning_rate=200.0,
                init="pca",
                random_state=seed,
                verbose=0,
            )
        Z = tsne.fit_transform(concat)
        if centers is not None:
            X2 = Z[: X.shape[0]]
            centers2 = Z[X.shape[0] :]
        else:
            X2 = Z
            centers2 = None

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    sc = None
    dbscan_num_plotted_clusters: Optional[int] = None
    if dbscan:
        # Phân cụm DBSCAN trên không gian 2D đã chiếu
        try:
            from sklearn.cluster import DBSCAN  # type: ignore
            db = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples))
            db_labels = db.fit_predict(X2)

            mask_noise = (db_labels == -1)
            sc_noise = None
            # Plot noise trước
            if np.any(mask_noise):
                only_noise = bool(np.all(mask_noise))
                noise_color = "#666666" if only_noise else "lightgray"
                noise_size = 10 if only_noise else 5
                noise_alpha = 0.8 if only_noise else 0.5
                sc_noise = ax.scatter(
                    X2[mask_noise, 0],
                    X2[mask_noise, 1],
                    c=noise_color,
                    s=noise_size,
                    alpha=noise_alpha,
                    linewidths=0,
                    label="noise",
                )
            # Plot các cụm DBSCAN (>=0)
            if np.any(~mask_noise):
                # Nếu có giới hạn số điểm mỗi cụm, tách phần dư thành các cụm phụ mới
                if dbscan_max_points_per_cluster is not None and dbscan_max_points_per_cluster > 0:
                    rng2 = np.random.default_rng(seed)
                    uniq = np.array(sorted([int(v) for v in np.unique(db_labels) if v >= 0]), dtype=int)
                    new_label_counter = 0
                    points_idx_list: List[int] = []
                    labels_plot_list: List[int] = []
                    cap = int(dbscan_max_points_per_cluster)
                    for cl in uniq:
                        idxs = np.where(db_labels == cl)[0]
                        if idxs.size == 0:
                            continue
                        if idxs.size <= cap:
                            points_idx_list.extend(idxs.tolist())
                            labels_plot_list.extend([new_label_counter] * int(idxs.size))
                            new_label_counter += 1
                        else:
                            perm = rng2.permutation(idxs)
                            for start in range(0, int(perm.size), cap):
                                chunk = perm[start : start + cap]
                                if chunk.size == 0:
                                    continue
                                points_idx_list.extend(chunk.tolist())
                                labels_plot_list.extend([new_label_counter] * int(chunk.size))
                                new_label_counter += 1
                    if points_idx_list:
                        pts = np.array(points_idx_list, dtype=int)
                        labels_plot = np.array(labels_plot_list, dtype=int)
                        dbscan_num_plotted_clusters = int(np.unique(labels_plot).size)
                        sc = ax.scatter(
                            X2[pts, 0],
                            X2[pts, 1],
                            c=labels_plot,
                            s=6,
                            cmap="tab20",
                            alpha=0.8,
                            linewidths=0,
                        )
                else:
                    sc = ax.scatter(
                        X2[~mask_noise, 0],
                        X2[~mask_noise, 1],
                        c=db_labels[~mask_noise],
                        s=6,
                        cmap="tab20",
                        alpha=0.8,
                        linewidths=0,
                    )
                    try:
                        dbscan_num_plotted_clusters = int(np.unique(db_labels[~mask_noise]).size)
                    except Exception:
                        dbscan_num_plotted_clusters = None
            # Nếu tất cả là noise, dùng scatter noise làm sc để đảm bảo có điểm hiển thị
            if sc is None and sc_noise is not None:
                sc = sc_noise
            # Không vẽ tâm/bán kính cụm KMeans khi đang hiển thị DBSCAN để tránh gây nhầm lẫn
            # (có thể bổ sung sau nếu cần tâm DBSCAN theo centroid)
        except Exception as e:
            print(f"[viz] Bỏ qua DBSCAN do lỗi: {e}")
            sc = ax.scatter(X2[:, 0], X2[:, 1], c=y, s=6, cmap="tab20", alpha=0.7, linewidths=0)
            if centers2 is not None:
                ax.scatter(centers2[:, 0], centers2[:, 1], c="black", s=40, marker="x", linewidths=1.5, alpha=0.9)
    else:
        # Mặc định tô màu theo nhãn KMeans và vẽ tâm + vòng tròn cụm
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=y, s=6, cmap="tab20", alpha=0.7, linewidths=0)
        if centers2 is not None:
            ax.scatter(centers2[:, 0], centers2[:, 1], c="black", s=40, marker="x", linewidths=1.5, alpha=0.9)
            # Vẽ vòng tròn cho mỗi cụm dựa trên bán kính phân vị khoảng cách tới tâm
            try:
                from matplotlib.patches import Circle  # type: ignore
                for c in range(K):
                    pts = X2[y == c]
                    if pts.shape[0] < 2:
                        continue
                    center_xy = centers2[c]
                    dists = np.linalg.norm(pts - center_xy[None, :], axis=1)
                    if dists.size == 0:
                        continue
                    # dùng phân vị 0.8 để bao phủ phần lớn điểm, tránh nhiễu outlier
                    r = float(np.quantile(dists, 0.8))
                    if r <= 0:
                        continue
                    circ = Circle(
                        (float(center_xy[0]), float(center_xy[1])),
                        r,
                        edgecolor="red",
                        facecolor="none",
                        linewidth=0.8,
                        alpha=0.7,
                    )
                    ax.add_patch(circ)
            except Exception as e:
                print(f"[viz] Bỏ qua vòng tròn cụm do lỗi: {e}")

    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    # Chỉ hiển thị colorbar nếu số cụm nhỏ để tránh rối
    if sc is not None:
        show_bar = True
        if dbscan:
            try:
                if dbscan_num_plotted_clusters is not None:
                    show_bar = dbscan_num_plotted_clusters <= 20
                else:
                    uniq = np.unique(db_labels[db_labels >= 0])
                    show_bar = uniq.size <= 20
            except Exception:
                show_bar = True
        else:
            show_bar = K <= 20
        if show_bar:
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Cluster")
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


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
    viz: bool = False,
    viz_method: str = "pca",
    viz_outdir: Optional[Path] = None,
    viz_max_points: int = 5000,
    viz_seed: int = 42,
    viz_dbscan: bool = False,
    viz_dbscan_eps: float = 0.5,
    viz_dbscan_min_samples: int = 5,
    viz_dbscan_max_points_per_cluster: Optional[int] = None,
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

        # Vẽ cụm nếu được yêu cầu
        if viz and viz_outdir is not None:
            try:
                out_img = viz_outdir / f"{rn}_clusters_{viz_method.lower()}.png"
                title = f"{rn} clusters K={centers.shape[0]} ({viz_method.upper()})"
                visualize_clusters_2d(
                    doc_matrix=doc_matrix,
                    labels=labels,
                    centers=centers,
                    method=viz_method,
                    out_path=out_img,
                    title=title,
                    max_points=viz_max_points,
                    seed=viz_seed,
                    dbscan=viz_dbscan,
                    dbscan_eps=viz_dbscan_eps,
                    dbscan_min_samples=viz_dbscan_min_samples,
                    dbscan_max_points_per_cluster=viz_dbscan_max_points_per_cluster,
                )
            except Exception as e:  # chỉ log, không làm hỏng flow tính weight
                print(f"[viz] Bỏ qua vẽ cho {rn} do lỗi: {e}")

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
    viz: bool = False,
    viz_method: str = "pca",
    viz_outdir: Optional[Path] = None,
    viz_max_points: int = 5000,
    viz_seed: int = 42,
    viz_dbscan: bool = False,
    viz_dbscan_eps: float = 0.5,
    viz_dbscan_min_samples: int = 5,
    viz_dbscan_max_points_per_cluster: Optional[int] = None,
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

        # Vẽ cụm nếu được yêu cầu
        if viz and viz_outdir is not None:
            try:
                out_img = viz_outdir / f"{rn}_clusters_{viz_method.lower()}.png"
                title = f"{rn} clusters K={centers.shape[0]} ({viz_method.upper()})"
                visualize_clusters_2d(
                    doc_matrix=doc_matrix,
                    labels=labels,
                    centers=centers,
                    method=viz_method,
                    out_path=out_img,
                    title=title,
                    max_points=viz_max_points,
                    seed=viz_seed,
                    dbscan=viz_dbscan,
                    dbscan_eps=viz_dbscan_eps,
                    dbscan_min_samples=viz_dbscan_min_samples,
                    dbscan_max_points_per_cluster=viz_dbscan_max_points_per_cluster,
                )
            except Exception as e:
                print(f"[viz] Bỏ qua vẽ cho {rn} do lỗi: {e}")

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


def compute_mor_post_entropy_weights(
    runs: RunDict,
    qids: List[str],
    q_texts: List[str],
    indices_base: Path,
    kmeans_k: int,
    topk: int,
    a: float,
    b: float,
    c: float,
    d: float,
    entropy_topk: int,
    entropy_tau: float,
    viz: bool = False,
    viz_method: str = "pca",
    viz_outdir: Optional[Path] = None,
    viz_max_points: int = 5000,
    viz_seed: int = 42,
    viz_dbscan: bool = False,
    viz_dbscan_eps: float = 0.5,
    viz_dbscan_min_samples: int = 5,
    viz_dbscan_max_points_per_cluster: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Giống compute_mor_post_weights nhưng trừ thêm d * entropy.

    Entropy được tính trên top-entropy_topk điểm số của run đó cho từng qid: p=softmax(scores/tau), H=-Σ p log p.
    s = a*V_pre_q + b*I_moran + c*V_post - d*H
    → min-max theo từng run → weight; BM25: phần bù 1 - mean(weight dense).
    """

    def is_bm25_run(name: str) -> bool:
        n = name.lower()
        return "bm25" in n

    def get_model_name(alias: str) -> str:
        a_ = alias.lower()
        if a_ in ("mpnet", "all-mpnet-base-v2", "st-mpnet"):
            return "sentence-transformers/all-mpnet-base-v2"
        if a_ in ("contriever", "facebook/contriever"):
            return "facebook/contriever"
        if a_ in ("gtr", "gtr-base", "gtr-t5-base"):
            return "sentence-transformers/gtr-t5-base"
        if a_ in ("gtr-large", "gtr-t5-large"):
            return "sentence-transformers/gtr-t5-large"
        if a_ in ("tas-b", "tasb", "msmarco-distilbert-base-tas-b"):
            return "sentence-transformers/msmarco-distilbert-base-tas-b"
        if a_ in ("dpr", "msmarco-dot", "msmarco-bert-base-dot-v5"):
            return "sentence-transformers/msmarco-bert-base-dot-v5"
        if a_ in ("ance", "msmarco-ance"):
            return "sentence-transformers/msmarco-bert-base-dot-v5"
        if a_ in ("simcse", "simcse-bert-base", "unsup-simcse-bert-base"):
            return "princeton-nlp/sup-simcse-roberta-base"
        return alias

    def dense_alias_from_run(run_name: str) -> str:
        name = run_name
        if name.startswith("dense_"):
            part = name[len("dense_"):]
            if "_" in part:
                alias_ = part.rsplit("_", 1)[0]
            else:
                alias_ = part
            return alias_
        return name

    run_names = sorted(runs.keys())
    bm25_runs = [rn for rn in run_names if is_bm25_run(rn)]
    dense_runs = [rn for rn in run_names if not is_bm25_run(rn) and rn.startswith("dense_")]

    # 1) Tính s_run(q) cho các run dense
    scores_per_run: Dict[str, Dict[str, float]] = {rn: {} for rn in dense_runs}

    for rn in dense_runs:
        idx_dir = indices_base / rn
        if not idx_dir.exists():
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

        # Vẽ cụm nếu được yêu cầu
        if viz and viz_outdir is not None:
            try:
                out_img = viz_outdir / f"{rn}_clusters_{viz_method.lower()}.png"
                title = f"{rn} clusters K={centers.shape[0]} ({viz_method.upper()})"
                visualize_clusters_2d(
                    doc_matrix=doc_matrix,
                    labels=labels,
                    centers=centers,
                    method=viz_method,
                    out_path=out_img,
                    title=title,
                    max_points=viz_max_points,
                    seed=viz_seed,
                    dbscan=viz_dbscan,
                    dbscan_eps=viz_dbscan_eps,
                    dbscan_min_samples=viz_dbscan_min_samples,
                    dbscan_max_points_per_cluster=viz_dbscan_max_points_per_cluster,
                )
            except Exception as e:
                print(f"[viz] Bỏ qua vẽ cho {rn} do lỗi: {e}")

        # Encode queries theo model của run để tính V_pre(q)
        alias = dense_alias_from_run(rn)
        model_name = get_model_name(alias)
        model = SentenceTransformer(model_name)
        q_matrix = model.encode(q_texts, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
        q_matrix = l2_normalize(q_matrix)
        q_pre_scores_rn: Dict[str, float] = {}
        for qid, qv in zip(qids, q_matrix):
            q_pre_scores_rn[qid] = mor_pre_score(qv, centers, labels)

        # Tính s_run(q)
        for qid in qids:
            # Cho MoR-post
            pairs_mor = runs[rn].get(qid, [])[:topk]
            if not pairs_mor:
                scores_per_run[rn][qid] = 0.0
                continue
            idxs = [docid_to_idx.get(docid, -1) for docid, _ in pairs_mor]
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

            # Cho entropy
            pairs_ent = runs[rn].get(qid, [])[:entropy_topk]
            if not pairs_ent:
                H = 1e6
            else:
                scores = np.array([float(s) for _, s in pairs_ent], dtype=np.float64)
                probs = softmax(scores, tau=entropy_tau)
                H = float(-np.sum(probs * np.log(probs + 1e-12)))

            s = a * V_pre_q + b * I_moran + c * V_post - d * H
            scores_per_run[rn][qid] = max(0.0, float(s))

    # 2) Min-max từng run dense → weight
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

    # 3) BM25: phần bù
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
    parser.add_argument("--mode", type=str, default="all", choices=["pre", "post", "post_entropy", "entropy", "both", "all"])
    parser.add_argument("--entropy-topk", type=int, default=10, help="Top-K docs for entropy calculation")
    parser.add_argument("--entropy-tau", type=float, default=1.0, help="Temperature for softmax in entropy calculation")
    parser.add_argument("--d", type=float, default=0.3, help="Coefficient for entropy term in post+entropy score")
    # Tuỳ chọn vẽ cụm
    parser.add_argument("--viz", action="store_true", help="Bật vẽ cụm 2D (PCA/t-SNE) cho mỗi run dense")
    parser.add_argument("--viz-method", type=str, default="pca", choices=["pca", "tsne"], help="Phương pháp giảm chiều để vẽ")
    parser.add_argument("--viz-out", type=str, default="viz", help="Thư mục xuất hình ảnh cụm")
    parser.add_argument("--viz-max-points", type=int, default=5000, help="Giới hạn số điểm vẽ (lấy mẫu nếu vượt)")
    parser.add_argument("--viz-seed", type=int, default=42, help="Random seed cho lấy mẫu và t-SNE/PCA")
    parser.add_argument("--viz-dbscan", action="store_true", help="Phân cụm DBSCAN trên không gian 2D đã chiếu và plot")
    parser.add_argument("--viz-dbscan-eps", type=float, default=0.5, help="eps cho DBSCAN trên 2D")
    parser.add_argument("--viz-dbscan-min-samples", type=int, default=5, help="min_samples cho DBSCAN trên 2D")
    parser.add_argument("--viz-dbscan-max-points-per-cluster", type=int, default=0, help="Giới hạn số điểm mỗi cụm DBSCAN (0 = không giới hạn)")
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
            viz=bool(args.viz),
            viz_method=str(args.viz_method),
            viz_outdir=Path(args.viz_out),
            viz_max_points=int(args.viz_max_points),
            viz_seed=int(args.viz_seed),
            viz_dbscan=bool(args.viz_dbscan),
            viz_dbscan_eps=float(args.viz_dbscan_eps),
            viz_dbscan_min_samples=int(args.viz_dbscan_min_samples),
            viz_dbscan_max_points_per_cluster=(int(args.viz_dbscan_max_points_per_cluster) if int(args.viz_dbscan_max_points_per_cluster) > 0 else None),
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
            viz=bool(args.viz),
            viz_method=str(args.viz_method),
            viz_outdir=Path(args.viz_out),
            viz_max_points=int(args.viz_max_points),
            viz_seed=int(args.viz_seed),
            viz_dbscan=bool(args.viz_dbscan),
            viz_dbscan_eps=float(args.viz_dbscan_eps),
            viz_dbscan_min_samples=int(args.viz_dbscan_min_samples),
            viz_dbscan_max_points_per_cluster=(int(args.viz_dbscan_max_points_per_cluster) if int(args.viz_dbscan_max_points_per_cluster) > 0 else None),
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

    if args.mode in ("post_entropy",) or run_in_all:
        runs = read_runs(run_root)
        weights_post_entropy = compute_mor_post_entropy_weights(
            runs=runs,
            qids=qids,
            q_texts=q_texts,
            indices_base=idx_base,
            kmeans_k=args.kmeans_k,
            topk=args.topk,
            a=args.a,
            b=args.b,
            c=args.c,
            d=args.d,
            entropy_topk=args.entropy_topk,
            entropy_tau=args.entropy_tau,
            viz=bool(args.viz),
            viz_method=str(args.viz_method),
            viz_outdir=Path(args.viz_out),
            viz_max_points=int(args.viz_max_points),
            viz_seed=int(args.viz_seed),
            viz_dbscan=bool(args.viz_dbscan),
            viz_dbscan_eps=float(args.viz_dbscan_eps),
            viz_dbscan_min_samples=int(args.viz_dbscan_min_samples),
            viz_dbscan_max_points_per_cluster=(int(args.viz_dbscan_max_points_per_cluster) if int(args.viz_dbscan_max_points_per_cluster) > 0 else None),
        )
        (out_dir / "mor_post_entropy.json").write_text(json.dumps(weights_post_entropy, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote MoR-post+Entropy weights to {out_dir / 'mor_post_entropy.json'}")


if __name__ == "__main__":
    main()

#python3 scripts/05_compute_weights_mor.py --mode pre --indices indices --runs runs --queries data/beir/queries.jsonl --viz --viz-method pca --viz-out viz --viz-max-points 4000
#python3 scripts/05_compute_weights_mor.py --mode pre --indices indices --runs runs --queries data/beir/queries.jsonl --viz --viz-method tsne --viz-dbscan --viz-dbscan-eps 0.7 --viz-dbscan-min-samples 8 --viz-out viz_tsne_dbscan