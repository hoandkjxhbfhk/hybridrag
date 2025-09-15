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
    import json
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                snippet = (line[:200] + "...") if len(line) > 200 else line
                raise SystemExit(
                    f"Invalid JSON at {path}:{i}: {e}\nSnippet: {snippet}"
                )


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
        "Yêu cầu định dạng đầu ra:\n"
        "- Mỗi câu hỏi trên một dòng riêng.\n"
        "- Không đánh số, không ký tự đầu dòng, không giải thích.\n"
        "- Kết thúc mỗi câu bằng dấu hỏi chấm (?).\n\n"
        f"Đoạn văn:\n{text}\n"
    )


def build_subq_prompt(query_text: str, num: int) -> str:
    return (
        "Bạn là trợ lý phân rã câu hỏi.\n"
        f"Hãy tạo {num} câu hỏi con ngắn gọn, cụ thể từ câu sau.\n"
        "Yêu cầu định dạng đầu ra:\n"
        "- Mỗi câu hỏi trên một dòng riêng.\n"
        "- Không đánh số, không ký tự đầu dòng, không giải thích.\n"
        "- Kết thúc mỗi câu bằng dấu hỏi chấm (?).\n\n"
        f"Câu hỏi:\n{query_text}\n"
    )


def _extract_assistant_segment(decoded: str) -> str:
    # 1) Try to capture the assistant final segment when special tokens survive
    m = re.search(r"<\|start\|>assistant(?:<\|channel\|>final)?<\|message\|>(.*?)<\|return\|>", decoded, flags=re.S)
    if m:
        return m.group(1).strip()
    # 2) When special tokens are stripped, Unsloth templates often collapse to
    #    'assistantanalysis... assistantfinal...'. Keep only the final content
    #    and drop any analysis content.
    tmp = re.sub(r"(?:^|\n)assistantanalysis.*?(?=(?:^|\n)assistantfinal|$)", "", decoded, flags=re.S)
    if "assistantfinal" in tmp:
        return tmp.split("assistantfinal", 1)[1].strip()
    return decoded.strip()


def _pick_n_questions(segment: str, n: int) -> List[str]:
    items: List[str] = []
    for ln in segment.split("\n"):
        s = ln.strip()
        if not s:
            continue
        # Remove list markers and role residues
        s = re.sub(r"^\s*(?:\d+\.|[-•*:+])\s*", "", s)
        s = re.sub(r"^assistant(?:<\|channel\|>)?(?:final|analysis)\s*", "", s, flags=re.I)
        s = s.strip("\"'“”")
        if 3 <= len(s) <= 200 and s.endswith("?"):
            items.append(s)
        if len(items) >= n:
            break
    if len(items) < n:
        # Fallback: search all question-like substrings
        qs = re.findall(r"([^\n\r\?]{3,200}\?)", segment)
        for q in qs:
            q = q.strip().strip("\"'“”")
            if q and q not in items:
                items.append(q)
            if len(items) >= n:
                break
    return items[:n]


def generate_many(model, tokenizer, prompt: str, num: int, max_new_tokens: int, temperature: float) -> List[str]:
    
    
    import torch

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

    # Keep streaming to stdout as feedback during generation
    streamer = TextStreamer(tokenizer)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
        )
    # Decode only newly generated tokens to avoid including the prompt/template
    start = inputs["input_ids"].shape[-1]
    gen_only = out[0, start:]
    decoded = tokenizer.decode(gen_only, skip_special_tokens=True)
    segment = _extract_assistant_segment(decoded)
    lines = _pick_n_questions(segment, num)
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
        # Đổi quy ước qid: dùng trực tiếp chunk id (pid) thay vì base_id
        # Ví dụ: qid = "doc-0-ch3#0" thay vì "doc-0#0"
        base_qid_prefix = f"{pid}#"

        if args.mode in ("q", "both"):
            prompt_q = build_query_prompt(text, num=args.per_doc)
            qs = generate_many(
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
            prompt_sq = build_subq_prompt(seed_q, num=args.per_doc)
            sqs = generate_many(
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
