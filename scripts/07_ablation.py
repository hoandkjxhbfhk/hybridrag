#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any

import subprocess
import yaml


def run_cmd(cmd: str) -> str:
    print("Running:", cmd)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (proc.stdout or "") + ("\nERR:\n" + proc.stderr if proc.stderr else "")
    print(out)
    return out


def parse_ndcg(stdout: str, fusion: str) -> (float, float):
    ndcg5 = ndcg20 = None
    for line in stdout.splitlines():
        if line.startswith(f"{fusion}:"):
            try:
                parts = line.split("=")
                ndcg5 = float(parts[1].split(",")[0])
                ndcg20 = float(parts[2])
            except Exception:
                pass
    return ndcg5, ndcg20


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation sweeps and save results CSV")
    parser.add_argument("--plan", type=str, default="",
                        help="YAML file specifying ablation plan; if empty, use defaults")
    parser.add_argument("--qrels", type=str, default="data/beir/qrels_test.tsv")
    parser.add_argument("--runs", type=str, default="runs")
    parser.add_argument("--weights", type=str, default="weights")
    parser.add_argument("--out", type=str, default="results/ablation.csv")
    args = parser.parse_args()

    plan_path = Path(args.plan) if args.plan else None
    if plan_path and plan_path.exists():
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    else:
        plan = {
            "fusion": ["normalized_sum", "rrf", "weighted_sum_mor_pre", "weighted_sum_mor_post"],
            "topk_list": [50],
            "kmeans_k_list": [4, 8],
            "abc_list": [
                {"a": 0.1, "b": 0.3, "c": 0.6},
                {"a": 0.2, "b": 0.3, "c": 0.5},
            ],
        }

    results: List[Dict[str, Any]] = []

    for topk in plan.get("topk_list", [50]):
        for kmeans_k in plan.get("kmeans_k_list", [64]):
            # Luôn tính lại weights cho mỗi cấu hình a/b/c khi có mor_post trong fusion
            for abc in plan.get("abc_list", [{"a": 0.1, "b": 0.3, "c": 0.6}]):
                a, b, c = abc.get("a", 0.1), abc.get("b", 0.3), abc.get("c", 0.6)
                # Tính weights pre/post
                cmd_weights = (
                    f"python scripts/05_compute_weights_mor.py --queries data/beir/queries.jsonl "
                    f"--indices indices --runs {args.runs} --out {args.weights} --kmeans-k {kmeans_k} "
                    f"--topk {topk} --a {a} --b {b} --c {c} --mode both"
                )
                run_cmd(cmd_weights)

                for fusion in plan.get("fusion", []):
                    cmd_eval = (
                        f"python scripts/06_fuse_and_eval.py --qrels {args.qrels} --runs {args.runs} "
                        f"--fusion {fusion} --weights {args.weights} --out fusion --topk {topk}"
                    )
                    out = run_cmd(cmd_eval)
                    ndcg5, ndcg20 = parse_ndcg(out, fusion)
                    results.append({
                        "fusion": fusion,
                        "topk": topk,
                        "kmeans_k": kmeans_k,
                        "a": a,
                        "b": b,
                        "c": c,
                        "ndcg5": ndcg5,
                        "ndcg20": ndcg20,
                    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fusion", "topk", "kmeans_k", "a", "b", "c", "ndcg5", "ndcg20"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"Wrote ablation results -> {out_path}")


if __name__ == "__main__":
    main()
