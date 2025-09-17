#!/usr/bin/env python3

import argparse
import collections
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np


RunRow = Tuple[str, str, int, float, str]


def read_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    rel: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            qid, docid, score = line.strip().split("\t")
            rel[qid][docid] = int(score)
    return rel


def read_run_file(path: Path) -> Dict[str, List[Tuple[str, float]]]:
    per_q_best: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            qid, _, docid, _rank, score, _ = parts
            s = float(score)
            prev = per_q_best[qid].get(docid)
            if prev is None or s > prev:
                per_q_best[qid][docid] = s
    per_q: Dict[str, List[Tuple[str, float]]] = {}
    for qid, d in per_q_best.items():
        items = list(d.items())
        items.sort(key=lambda x: x[1], reverse=True)
        per_q[qid] = items
    return per_q


def collect_runs(run_root: Path) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    runs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    for sub in run_root.rglob("test.txt"):
        run_name = sub.parent.name
        runs[run_name] = read_run_file(sub)
    return runs


def weighted_normalized_sum_fusion(
    runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
    weights: Dict[str, float],
    topk: int,
) -> Dict[str, List[Tuple[str, float]]]:
    fused: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    for run_name, per_q in runs.items():
        w = float(weights.get(run_name, 0.0))
        if w <= 0.0:
            continue
        for qid, pairs in per_q.items():
            if not pairs:
                continue
            scores = np.array([s for _, s in pairs], dtype=np.float32)
            if scores.size == 0:
                continue
            mn, mx = float(scores.min()), float(scores.max())
            denom = (mx - mn) if mx > mn else 1.0
            for docid, s in pairs:
                fused[qid][docid] += w * ((float(s) - mn) / denom)
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


def precision_at_k(true_rel: Dict[str, int], ranked_docs: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    preds = ranked_docs[:k]
    if not preds:
        return 0.0
    rel_set = {d for d, r in true_rel.items() if r > 0}
    hit = sum(1 for d in preds if d in rel_set)
    return hit / float(k)


def recall_at_k(true_rel: Dict[str, int], ranked_docs: List[str], k: int) -> float:
    rel_set = {d for d, r in true_rel.items() if r > 0}
    if not rel_set:
        return 0.0
    preds = ranked_docs[:k]
    hit = sum(1 for d in preds if d in rel_set)
    return hit / float(len(rel_set))


def ap_at_k(true_rel: Dict[str, int], ranked_docs: List[str], k: int) -> float:
    rel_set = {d for d, r in true_rel.items() if r > 0}
    if not rel_set:
        return 0.0
    ap = 0.0
    hit = 0
    for i, d in enumerate(ranked_docs[:k], start=1):
        if d in rel_set:
            hit += 1
            ap += hit / float(i)
    return ap / float(min(len(rel_set), k))


def mrr_at_k(true_rel: Dict[str, int], ranked_docs: List[str], k: int) -> float:
    rel_set = {d for d, r in true_rel.items() if r > 0}
    for i, d in enumerate(ranked_docs[:k], start=1):
        if d in rel_set:
            return 1.0 / float(i)
    return 0.0


def evaluate_all_metrics(
    qrels: Dict[str, Dict[str, int]],
    fused: Dict[str, List[Tuple[str, float]]],
    k_list: Iterable[int],
) -> Dict[str, Dict[int, float]]:
    metrics: Dict[str, Dict[int, float]] = {
        "ndcg": {},
        "precision": {},
        "recall": {},
        "map": {},
        "mrr": {},
    }

    for k in k_list:
        vals_ndcg: List[float] = []
        vals_p: List[float] = []
        vals_r: List[float] = []
        vals_ap: List[float] = []
        vals_mrr: List[float] = []
        for qid, pairs in fused.items():
            ranked = [docid for docid, _ in pairs]
            tr = qrels.get(qid, {})
            vals_ndcg.append(ndcg_at_k(tr, ranked, k))
            vals_p.append(precision_at_k(tr, ranked, k))
            vals_r.append(recall_at_k(tr, ranked, k))
            vals_ap.append(ap_at_k(tr, ranked, k))
            vals_mrr.append(mrr_at_k(tr, ranked, k))
        metrics["ndcg"][k] = float(np.mean(vals_ndcg)) if vals_ndcg else 0.0
        metrics["precision"][k] = float(np.mean(vals_p)) if vals_p else 0.0
        metrics["recall"][k] = float(np.mean(vals_r)) if vals_r else 0.0
        metrics["map"][k] = float(np.mean(vals_ap)) if vals_ap else 0.0
        metrics["mrr"][k] = float(np.mean(vals_mrr)) if vals_mrr else 0.0
    return metrics


def generate_weight_combinations(run_names: List[str], step: float) -> List[Dict[str, float]]:
    n = len(run_names)
    total = int(round(1.0 / step))
    results: List[Dict[str, float]] = []

    def backtrack(idx: int, remaining: int, current: List[int]) -> None:
        if idx == n - 1:
            current.append(remaining)
            weights = {run_names[i]: current[i] * step for i in range(n)}
            results.append(weights)
            current.pop()
            return
        for v in range(0, remaining + 1):
            current.append(v)
            backtrack(idx + 1, remaining - v, current)
            current.pop()

    backtrack(0, total, [])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search weighted normalized-sum fusion (weights step 0.1)")
    parser.add_argument("--qrels", type=str, required=True)
    parser.add_argument("--runs", type=str, required=True)
    parser.add_argument("--out-csv", type=str, default="results/weighted_normsum_grid.csv")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--k-list", type=str, default="5,20", help="Comma-separated cutoffs to report metrics at")
    parser.add_argument("--step", type=float, default=0.1, help="Weight step (e.g., 0.1)")
    args = parser.parse_args()

    qrels = read_qrels(Path(args.qrels))
    runs = collect_runs(Path(args.runs))

    if not runs:
        print("No runs found.")
        return

    run_names = sorted(runs.keys())
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]

    combos = generate_weight_combinations(run_names, step=args.step)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    agg_metrics = {
        "ndcg": {k: [] for k in k_list},
        "precision": {k: [] for k in k_list},
        "recall": {k: [] for k in k_list},
        "map": {k: [] for k in k_list},
        "mrr": {k: [] for k in k_list},
    }

    best = {"combo": None, "metric": "ndcg", "k": k_list[0], "score": -1.0}

    for weights in combos:
        fused = weighted_normalized_sum_fusion(runs, weights=weights, topk=args.topk)
        metrics = evaluate_all_metrics(qrels, fused, k_list=k_list)
        row = {"weights": ";".join(f"{rn}:{weights[rn]:.1f}" for rn in run_names)}
        for m in ("ndcg", "precision", "recall", "map", "mrr"):
            for k in k_list:
                val = metrics[m][k]
                row[f"{m}@{k}"] = val
                agg_metrics[m][k].append(val)
        rows.append(row)

        # update best by first k in k_list on ndcg by default
        nd = metrics["ndcg"][k_list[0]]
        if nd > best["score"]:
            best = {"combo": row["weights"], "metric": "ndcg", "k": k_list[0], "score": nd}

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["weights"] + [f"{m}@{k}" for m in ("ndcg", "precision", "recall", "map", "mrr") for k in k_list]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # print averages and best
    print(f"Wrote CSV -> {out_csv}")
    for m in ("ndcg", "precision", "recall", "map", "mrr"):
        for k in k_list:
            vals = agg_metrics[m][k]
            mean_val = float(np.mean(vals)) if vals else 0.0
            print(f"AVG {m}@{k} = {mean_val:.4f} over {len(vals)} combos")
    if best["combo"] is not None:
        print(f"BEST {best['metric']}@{best['k']} = {best['score']:.4f} at {best['combo']}")


if __name__ == "__main__":
    main()


# python scripts/08_weighted_normsum_grid.py \
#   --qrels /home/hoan/hybridrag/data/qrels.tsv \
#   --runs /home/hoan/hybridrag/runs \
#   --out-csv /home/hoan/hybridrag/results/weighted_normsum_grid.csv \
#   --topk 100 \
#   --k-list 5,20 \
#   --step 0.1 | cat