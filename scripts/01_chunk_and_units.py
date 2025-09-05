#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def chunk_text(text: str, chunk_len: int, stride: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_len]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if i + chunk_len >= len(words):
            break
        i += max(1, chunk_len - stride)
    return chunks


def process_corpus(in_path: Path, out_path: Path, chunk_len: int, stride: int) -> int:
    out_records: List[Dict] = []
    for rec in read_jsonl(in_path):
        base_id = rec.get("_id") or rec.get("id")
        text = rec.get("text", "")
        title = rec.get("title", "")
        metadata = rec.get("metadata", {}) or {}
        parts = chunk_text(text, chunk_len=chunk_len, stride=stride)
        if not parts:
            continue
        for j, part in enumerate(parts):
            pid = f"{base_id}-ch{j}"
            out_records.append({
                "_id": pid,
                "text": part,
                "title": title,
                "metadata": {**metadata, "base_id": base_id, "chunk_index": j},
            })
    write_jsonl(out_records, out_path)
    return len(out_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk corpus.jsonl into passage-level records")
    parser.add_argument("--in", dest="in_path", type=str, required=True, help="Input corpus.jsonl path")
    parser.add_argument("--chunk-len", dest="chunk_len", type=int, default=400, help="Chunk length in words")
    parser.add_argument("--stride", dest="stride", type=int, default=100, help="Stride in words (overlap)")
    parser.add_argument("--out", dest="out_path", type=str, required=True, help="Output corpus.jsonl path")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    n = process_corpus(in_path, out_path, args.chunk_len, args.stride)
    print(f"Wrote {n} chunks to {out_path}")


if __name__ == "__main__":
    main()
