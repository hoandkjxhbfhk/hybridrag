#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict


def read_text_file(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()


def build_corpus_records(txt_paths: List[Path]) -> List[Dict]:
    records: List[Dict] = []
    for i, p in enumerate(sorted(txt_paths)):
        text = read_text_file(p)
        if not text:
            continue
        doc_id = f"doc-{i}"
        records.append({
            "_id": doc_id,
            "text": text,
            "title": p.stem,
            "metadata": {
                "source_file": str(p),
                "num_chars": len(text),
            },
        })
    return records


def write_jsonl(records: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BEIR-like corpus.jsonl from local .txt files")
    parser.add_argument("--in", dest="in_dir", type=str, default="data/raw", help="Input directory of .txt files")
    parser.add_argument("--out", dest="out_dir", type=str, default="data/beir", help="Output directory for BEIR data")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    txt_paths = list(in_dir.glob("*.txt"))

    if not txt_paths:
        raise SystemExit(f"No .txt files found in {in_dir}")

    records = build_corpus_records(txt_paths)
    out_path = out_dir / "corpus.jsonl"
    write_jsonl(records, out_path)

    print(f"Wrote {len(records)} documents to {out_path}")


if __name__ == "__main__":
    main()
