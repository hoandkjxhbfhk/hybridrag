#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import yaml


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
            "fusion": ["normalized_sum", "rrf", "weighted_sum_mor_pre"],
            "topk": 50,
        }

    # Gọi 06_fuse_and_eval.py cho từng fusion, gom kết quả
    results: List[Dict] = []
    for fusion in plan.get("fusion", []):
        cmd = (
            f"python scripts/06_fuse_and_eval.py --qrels {args.qrels} --runs {args.runs} "
            f"--fusion {fusion} --weights {args.weights} --out fusion --topk {plan.get('topk', 50)}"
        )
        print("Running:", cmd)
        # Đơn giản: chạy subprocess và đọc stdout để lấy NDCG
        import subprocess
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(proc.stdout)
        # parse dòng: "{fusion}: Average NDCG@5=..., NDCG@20=..."
        ndcg5 = ndcg20 = None
        for line in proc.stdout.splitlines():
            if line.startswith(f"{fusion}:"):
                try:
                    parts = line.split("=")
                    ndcg5 = float(parts[1].split(",")[0])
                    ndcg20 = float(parts[2])
                except Exception:
                    pass
        results.append({
            "fusion": fusion,
            "ndcg5": ndcg5,
            "ndcg20": ndcg20,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fusion", "ndcg5", "ndcg20"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"Wrote ablation results -> {out_path}")


if __name__ == "__main__":
    main()
