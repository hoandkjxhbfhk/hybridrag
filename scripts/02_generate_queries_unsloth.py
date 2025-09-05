#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from unsloth import FastLanguageModel
from transformers import TextStreamer

MODEL_ID_DEFAULT = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"


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


def write_qrels(rows: List[Tuple[str, str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, pid, score in rows:
            f.write(f"{qid}\t{pid}\t{score}\n")


def build_query_prompt(text: str, num: int) -> str:
    return (
        "Bạn là trợ lý tạo câu hỏi tìm kiếm.\n"
        f"Viết {num} câu hỏi ngắn bằng tiếng Việt mà đoạn sau trả lời trực tiếp.\n"
        "- Chỉ in danh sách câu hỏi, mỗi dòng một câu.\n"
        "- Không giải thích, không đánh số.\n\n"
        f"Đoạn văn:\n{text}\n"
    )


def build_subq_prompt(query_text: str, num: int) -> str:
    return (
        "Bạn là trợ lý phân rã câu hỏi.\n"
        f"Hãy phân rã câu sau thành {num} câu hỏi con ngắn gọn, cụ thể.\n"
        "- Chỉ in danh sách câu hỏi con, mỗi dòng một câu.\n"
        "- Không giải thích, không đánh số.\n\n"
        f"Câu hỏi:\n{query_text}\n"
    )


def generate_list(model, tokenizer, prompt: str, num: int, max_new_tokens: int, temperature: float) -> List[str]:
    inputs = tokenizer([prompt], return_tensors="pt")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num,
        streamer=None,  # đặt None để không in ra STDOUT; dùng streamer nếu muốn stream
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    lines: List[str] = []
    for d in decoded:
        for ln in d.split("\n"):
            s = ln.strip().lstrip("-•*").strip()
            if 3 <= len(s) <= 200:
                lines.append(s)
    # Lấy đúng num phần tử, bổ sung nếu thiếu
    while len(lines) < num:
        lines.append(lines[-1] if lines else "Câu hỏi gì được trả lời bởi đoạn trên?")
    return lines[:num]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate queries and qrels using Unsloth FastLanguageModel")
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--out-queries", type=str, required=True)
    parser.add_argument("--out-qrels", type=str, required=True)
    parser.add_argument("--mode", type=str, default="both", choices=["q", "sq", "both"])
    parser.add_argument("--model", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument("--per-doc", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    out_queries = Path(args.out_queries)
    out_qrels = Path(args.out_qrels)

    # Khởi tạo model theo phong cách code gốc
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,  # để Unsloth tự chọn (fp16/bf16 nếu có GPU)
        load_in_4bit=not args.no_4bit,
    )
    FastLanguageModel.for_inference(model)

    queries_out: List[Dict] = []
    qrels_rows: List[Tuple[str, str, int]] = []

    for rec in read_jsonl(corpus_path):
        pid = rec["_id"]
        text = rec.get("text", "")
        base_id = rec.get("metadata", {}).get("base_id", pid)
        base_qid_prefix = f"{base_id}#"

        if args.mode in ("q", "both"):
            prompt_q = build_query_prompt(text, args.per_doc)
            qs = generate_list(
                model,
                tokenizer,
                prompt_q,
                num=args.per_doc,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            for k, q in enumerate(qs):
                qid = f"{base_qid_prefix}{k}"
                queries_out.append({"qid": qid, "text": q})
                qrels_rows.append((qid, pid, 1))

        if args.mode in ("sq", "both"):
            seed_q = queries_out[-1]["text"] if queries_out else text[:160]
            prompt_sq = build_subq_prompt(seed_q, args.per_doc)
            sqs = generate_list(
                model,
                tokenizer,
                prompt_sq,
                num=args.per_doc,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            for k, sq in enumerate(sqs):
                qid = f"{base_qid_prefix}s{k}"
                queries_out.append({"qid": qid, "text": sq})
                qrels_rows.append((qid, pid, 1))

    write_jsonl(queries_out, out_queries)
    write_qrels(qrels_rows, out_qrels)

    print(f"Wrote {len(queries_out)} queries -> {out_queries}")
    print(f"Wrote {len(qrels_rows)} qrels  -> {out_qrels}")


if __name__ == "__main__":
    main()
