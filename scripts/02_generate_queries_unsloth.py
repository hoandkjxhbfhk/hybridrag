#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from unsloth import FastLanguageModel
from transformers import TextStreamer
import re

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


def build_query_prompt(text: str) -> str:
    return (
        "Bạn là trợ lý tạo câu hỏi tìm kiếm.\n"
        "Viết 1 câu hỏi ngắn bằng tiếng Việt mà đoạn sau trả lời trực tiếp.\n"
        "- Chỉ in câu hỏi duy nhất, không đánh số, không giải thích.\n\n"
        f"Đoạn văn:\n{text}\n"
    )


def build_subq_prompt(query_text: str) -> str:
    return (
        "Bạn là trợ lý phân rã câu hỏi.\n"
        "Hãy tạo 1 câu hỏi con ngắn gọn, cụ thể từ câu sau.\n"
        "- Chỉ in câu hỏi, không giải thích.\n\n"
        f"Câu hỏi:\n{query_text}\n"
    )


def generate_list(model, tokenizer, prompt: str, num: int, max_new_tokens: int, temperature: float) -> List[str]:
    import torch

    lines: List[str] = []
    for _ in range(num):
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="low",
        )
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        streamer = TextStreamer(tokenizer)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        extracted: List[str] = []
        for d in decoded:
            # 1) Try to capture the assistant final segment when special tokens survive
            m = re.search(r"<\|start\|>assistant(?:<\|channel\|>final)?<\|message\|>(.*?)<\|return\|>", d, flags=re.S)
            segment: str
            if m:
                segment = m.group(1).strip()
            else:
                # 2) When special tokens are stripped, Unsloth templates often collapse to
                #    'assistantanalysis... assistantfinal...'. Keep only the final content
                #    and drop any analysis content.
                tmp = re.sub(r"(?:^|\n)assistantanalysis.*?(?=(?:^|\n)assistantfinal|$)", "", d, flags=re.S)
                if "assistantfinal" in tmp:
                    segment = tmp.split("assistantfinal", 1)[1]
                else:
                    segment = tmp

            # From the chosen segment, pick the first plausible question line
            picked = None
            for ln in segment.split("\n"):
                s = ln.strip()
                if not s:
                    continue
                # Remove common list markers and role leftovers
                s = s.lstrip("-•*:").strip()
                s = re.sub(r"^assistant(?:<\|channel\|>)?(?:final|analysis)\s*", "", s, flags=re.I)
                s = s.strip("\"'“”")
                if 3 <= len(s) <= 200 and s.endswith("?"):
                    picked = s
                    break
            if not picked:
                # As a fallback, search any question-like substring
                m2 = re.search(r"([^\n\r\?]{3,200}\?)", segment)
                if m2:
                    picked = m2.group(1).strip().strip("\"'“”")
            if picked:
                extracted.append(picked)
        if extracted:
            lines.append(extracted[0])
        if len(lines) >= num:
            break

    while len(lines) < num:
        lines.append("Câu hỏi gì được trả lời bởi đoạn trên?")
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
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    out_queries = Path(args.out_queries)
    out_qrels = Path(args.out_qrels)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        dtype=None,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        full_finetuning=False,
        trust_remote_code=True,
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
            prompt_q = build_query_prompt(text)
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
            seed_q = queries_out[-1]["text"] if queries_out else text[:]
            prompt_sq = build_subq_prompt(seed_q)
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
