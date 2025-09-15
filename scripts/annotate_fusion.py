#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union


RunRow = Tuple[str, str, int, float, str]  # qid, docid, rank, score, runname


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                snippet = (line[:200] + "...") if len(line) > 200 else line
                raise SystemExit(f"Invalid JSON at {path}:{i}: {e}\nSnippet: {snippet}")


def load_queries_map(path: Path) -> Dict[str, Dict[str, Union[str, Dict]]]:
    qmap: Dict[str, Dict[str, Union[str, Dict]]] = {}
    for rec in read_jsonl(path):
        qid = str(rec.get("qid") or rec.get("id"))
        if not qid:
            continue
        qmap[qid] = {
            "text": rec.get("text", ""),
            "title": rec.get("title", ""),
            "metadata": rec.get("metadata", {}),
        }
    return qmap


def load_corpus_map(path: Path) -> Dict[str, Dict[str, Union[str, Dict]]]:
    cmap: Dict[str, Dict[str, Union[str, Dict]]] = {}
    for rec in read_jsonl(path):
        pid = str(rec.get("_id") or rec.get("id"))
        if not pid:
            continue
        cmap[pid] = {
            "text": rec.get("text", ""),
            "title": rec.get("title", ""),
            "metadata": rec.get("metadata", {}),
        }
    return cmap


def read_run(path: Path) -> List[RunRow]:
    rows: List[RunRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            qid, _, docid, rank, score, runname = parts
            try:
                rows.append((qid, docid, int(rank), float(score), runname))
            except Exception:
                continue
    return rows


def annotate_run(
    run_path: Path,
    qmap: Dict[str, Dict[str, Union[str, Dict]]],
    cmap: Dict[str, Dict[str, Union[str, Dict]]],
    out_dir: Path,
    topk: int,
    fmt: str,
    trunc: int,
) -> Path:
    rows = read_run(run_path)
    # group by qid, then take topk by rank
    rows.sort(key=lambda r: (r[0], r[2]))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (run_path.stem + (".jsonl" if fmt == "jsonl" else ".tsv"))

    if fmt == "jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            last_qid = None
            kept = 0
            for qid, docid, rank, score, runname in rows:
                if qid != last_qid:
                    last_qid = qid
                    kept = 0
                if kept >= topk:
                    continue
                kept += 1
                q = qmap.get(qid, {"text": ""})
                d = cmap.get(docid, {"text": ""})
                d_text = str(d.get("text", ""))
                if trunc and len(d_text) > trunc:
                    d_text = d_text[:trunc] + "..."
                obj = {
                    "qid": qid,
                    "query": q.get("text", ""),
                    "rank": rank,
                    "docid": docid,
                    "score": score,
                    "run": runname,
                    "doc_title": d.get("title", ""),
                    "doc_text": d_text,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    else:
        # tsv header
        with out_path.open("w", encoding="utf-8") as f:
            f.write("qid\tquery\trank\tdocid\tscore\trun\tdoc_title\tdoc_text\n")
            last_qid = None
            kept = 0
            for qid, docid, rank, score, runname in rows:
                if qid != last_qid:
                    last_qid = qid
                    kept = 0
                if kept >= topk:
                    continue
                kept += 1
                q = qmap.get(qid, {"text": ""})
                d = cmap.get(docid, {"text": ""})
                d_text = str(d.get("text", ""))
                if trunc and len(d_text) > trunc:
                    d_text = d_text[:trunc] + "..."
                line = (
                    f"{qid}\t{str(q.get('text','')).replace('\t',' ').replace('\n',' ')}\t{rank}\t{docid}\t{score:.6f}\t{runname}\t"
                    f"{str(d.get('title','')).replace('\t',' ').replace('\n',' ')}\t{d_text.replace('\t',' ').replace('\n',' ')}\n"
                )
                f.write(line)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate TREC run files with original query/doc texts")
    parser.add_argument("--queries", type=str, default="data/beir/queries.jsonl")
    parser.add_argument("--corpus", type=str, default="data/beir/corpus.jsonl")
    parser.add_argument("--runs", type=str, default="fusion", help="Run file or directory containing fused_*.txt")
    parser.add_argument("--out", type=str, default="annotated")
    parser.add_argument("--topk", type=int, default=10, help="Rows per query to include")
    parser.add_argument("--format", type=str, default="tsv", choices=["tsv", "jsonl"])
    parser.add_argument("--truncate", type=int, default=400, help="Truncate doc text to N chars (0 to disable)")
    args = parser.parse_args()

    qmap = load_queries_map(Path(args.queries))
    cmap = load_corpus_map(Path(args.corpus))

    runs_path = Path(args.runs)
    out_dir = Path(args.out)

    targets: List[Path] = []
    if runs_path.is_file():
        targets = [runs_path]
    else:
        if runs_path.exists():
            targets = sorted([p for p in runs_path.iterdir() if p.is_file() and p.name.startswith("fused_") and p.suffix == ".txt"])

    if not targets:
        raise SystemExit(f"No run files found under: {runs_path}")

    for rp in targets:
        out_path = annotate_run(rp, qmap, cmap, out_dir, args.topk, args.format, args.truncate)
        print(f"Wrote annotated: {out_path}")


if __name__ == "__main__":
    main()



# Annotate toàn bộ file fused_*.txt trong folder fusion, mỗi query lấy top 3 kết quả, xuất TSV, cắt text tài liệu 120 ký tự:
# python scripts/annotate_fusion.py --queries data/beir/queries.jsonl --corpus data/beir/corpus.jsonl --runs fusion --out annotated --topk 3 --format tsv --truncate 120
# Annotate một file cụ thể và xuất JSONL đầy đủ:
# python scripts/annotate_fusion.py --queries data/beir/queries.jsonl --corpus data/beir/corpus.jsonl --runs fusion/fused_weighted_sum_mor_post.txt --out annotated --topk 10 --format jsonl --truncate 0