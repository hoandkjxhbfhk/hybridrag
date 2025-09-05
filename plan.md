

# 0) Mục tiêu & giả định tối thiểu

* **Input**: một thư mục `data/raw/` chứa các tài liệu `.txt`.
* **Đầu ra**: bộ dữ liệu kiểu **BEIR** (corpus, queries, qrels), các **run files** cho nhiều retrievers, **fusion** (normalized-sum / RRF / weighted-sum + MoR pre/post), **report** NDCG\@5/20 và **bảng ablation**.
* **Phần cứng**: máy cá nhân (GPU càng tốt) để chạy LLM OSS qua **unsloth** (tạo sub-questions / queries / propositions).
* **Thư viện chính**: `sentence-transformers` (dense), `faiss` (ANN), `pyserini` hoặc `whoosh` (BM25), `sklearn` (KMeans), `numpy/scipy`, `beir` (EvaluateRetrieval).

---

# 1) Cấu trúc thư mục đề xuất

```
your-repo/
  data/
    raw/                  # .txt đầu vào
    beir/                 # dữ liệu chuẩn hoá theo format BEIR
      corpus.jsonl        # {doc_id, text, title(opt), meta(opt)}
      queries.jsonl       # {qid, text}
      qrels_test.tsv      # qrels (test split)
      splits.json         # {"test": [qid,...]} nếu cần
  indices/
    bm25_qd/ ...          # chỉ mục cho từng (retriever × granularity)
    dense_mpnet_qp/ ...
    ...
  runs/
    bm25/                 # kết quả truy hồi dạng TREC run
      test.txt
    mpnet/
      test.txt
    ...
  weights/                # trọng số (nếu dùng JSON tĩnh) hoặc cache MoR
  fusion/
    fused_normalized_sum.txt
    fused_rrf.txt
    fused_weighted_mor_pre.txt
    fused_weighted_mor_post.txt
  scripts/
    00_prepare_local_texts.py
    01_chunk_and_units.py
    02_generate_queries_unsloth.py
    03_build_indices.py
    04_retrieve.py
    05_compute_weights_mor.py
    06_fuse_and_eval.py
    07_ablation.py
  mor/
    fusion.py             # normalized_sum, rrf, weighted_sum
    weights.py            # V_pre, I_Moran, V_post, routing
    eval_utils.py         # BEIR EvaluateRetrieval wrapper
  retrievers/
    bm25.py               # pyserini/whoosh
    dense.py              # ST/FAISS encode + search
  README.md
  requirements.txt
  run_all.sh
```

---

# 2) Chuẩn hoá dữ liệu → format BEIR (corpus/queries/qrels)

### 2.1 Chuẩn hoá tài liệu

* Đọc `data/raw/*.txt`, gán `doc_id = f"doc-{i}"`.
* **Chunking**: tách theo độ dài token (ví dụ 256–512 token). Lưu:

  * **Granularity “d” (document/ passage)**: chunk → `doc_id-ch{j}` với `text=chunk_text`.
  * **Granularity “p” (proposition)**: dùng LLM OSS (unsloth) tách chunk thành **các mệnh đề ngắn** (atomic sentences) → `doc_id-ch{j}-p{k}`.
* Lưu tất cả vào `corpus.jsonl` (mỗi dòng: `{"_id": pid, "text": "...", "title": "", "metadata": {...}}`).

### 2.2 Sinh query & qrels (zero-shot)

* Với mỗi **chunk** (hoặc **proposition**) sinh 1–3 **queries** bằng LLM:

  * “Viết câu hỏi ngắn mà đoạn dưới đây trả lời trực tiếp…”
  * Nếu cần **sub-questions (sq)**: yêu cầu LLM phân rã câu hỏi tổng (hoặc từ chunk) thành các **sub-questions**.
* Mỗi query có `qid = f"{base_doc_id}#k"` (giữ `#` để phân biệt whole/sub).
* **Qrels**: đánh nhãn **relevant=1** cho cặp (qid, đúng pid nguồn). Có thể mở rộng: nếu dùng propositions, qrels map tới `-p{k}`; nếu chỉ chunk, map tới `-ch{j}`.
* Gộp queries thành `queries.jsonl`, qrels vào `qrels_test.tsv` (tab: `query-id \t corpus-id \t score` với header).

> Mẹo: Giữ **mapping** giữa (qid → granularity cần dùng) để sau này chạy đúng index theo case.

---

# 3) Dựng retrievers & chỉ mục (4 “độ hạt” như bài)

Ta tạo **4 biến thể** cho mỗi retriever $\mathcal{R}_i$:

* **q·d**: query gốc ↔ chunk/passages gốc (case “chunk\_whole”)
* **q·p**: query gốc ↔ propositions (case “prop\_whole”)
* **sq·d**: sub-questions ↔ chunk (case “chunk\_sub”)
* **sq·p**: sub-questions ↔ propositions (case “prop\_sub”)

### 3.1 BM25 (sparse)

* Dùng **Pyserini** (ưu tiên) hoặc **Whoosh** nếu offline.
* Chỉ mục riêng cho `d` và `p`.

### 3.2 Dense retrievers (ST/FAISS)

* Chọn vài backbone: `all-mpnet-base-v2`, `contriever`, `gtr-base`, `simcse`, `tas-b` (tuỳ máy).
* Encode **corpus** (d/p) thành vectors, build **FAISS index** (IVF/Flat tuỳ RAM).
* Encode **queries**/**sub-queries** để search.

> Lưu **run files** theo format **TREC (6 cột)** vào `runs/<encoder>/<case>.txt`.

---

# 4) Trộn (fusion) & trọng số (MoR)

`normalized_sum`, `rrf`, `weighted_sum` và bổ sung **tín hiệu MoR**:

### 4.1 `V_pre(q, R_i, D)` (pre-retrieval)

* Với mỗi retriever $R_i$, lấy **embedding corpus** (đã có ở dense; với BM25 cần “proxy” embedding: dùng một encoder chung để nhúng văn bản corpus một lần).
* Chạy **KMeans** (K ≈ 128–512 tùy size). Lưu **centroids** + kích thước cụm.
* Cho mỗi **query vector** $\vec q$ (dựa trên **không gian** của $R_i$; với BM25 dùng encoder chung):

  * Tính $\vec v_k = \vec m_k - \vec q$, rồi độ đo tổng hợp theo công thức (chuẩn hoá hướng/độ lớn; có cân kích thước cụm).
  * Trực giác: càng **gần** “cụm to” → điểm càng **cao**.

### 4.2 `I_Moran(q, R_i, D)` (post-retrieval, quan hệ trong top-k)

* Lấy **top-K** tài liệu (K=20/50) của $R_i$ cho q.
* Tính **Moran’s I** trên ma trận khoảng cách/cận kề (cosine) giữa các doc embeddings trong top-K.
* **Cao** → các doc **gần nhau** → chất lượng likely tốt.

### 4.3 `V_post(q, R_i, D)` (post-retrieval, so với toàn corpus)

* Với mỗi tài liệu top-K, tính **V\_pre(d\_n, R\_i, D)** (thay q bằng doc embedding).
* Lấy **trung bình**: $V_{post} = \| \frac{1}{K}\sum V_{pre}(d_n) \|$.

### 4.4 Gán trọng số theo truy vấn

* **MoR-pre**: $f_{pre} = V_{pre}$.
* **MoR-post**: $f_{post} = a \cdot V_{pre} + b \cdot I_{Moran} + c \cdot V_{post}$
  (ví dụ $a=0.1, b=0.3, c=0.6$ — bạn sẽ tune trong ablation).
* Chuẩn hoá **f** theo mỗi truy vấn (min-max hoặc l1 norm) trước khi “weighted\_sum”.

> Kết quả: `weights/<encoder_case>.json` **không cần** nếu bạn tính “on-the-fly”; hoặc cache để tái sử dụng.

---

# 5) Đánh giá (BEIR) & báo cáo

* Dùng `EvaluateRetrieval` để lấy **NDCG\@5/20** per-query và trung bình.
* Lưu `fusion/*.txt` (TREC run) và log kết quả:

  ```
  Average NDCG@5  = ...
  Average NDCG@20 = ...
  ```
* Sinh **CSV** tổng hợp: hàng = cấu hình (retriever/case/fusion/weights), cột = NDCG\@5/20.

---

# 6) Kế hoạch ablation

Chạy so sánh có kiểm soát:

1. **Retrievers**: BM25 vs dense (1 model), + dần nhiều model.
2. **Granularity**: q·d vs q·p; sq·d vs sq·p.
3. **Fusion**: normalized\_sum vs RRF (với các k) vs weighted\_sum.
4. **Weights**: none (đều 1) vs **MoR-pre** vs **MoR-post** (quét (a,b,c)).
5. **KMeans K**: 64/128/256; **top-K** cho post: 20/50.
6. **Sub-questions**: bật/tắt.
7. **Propositions**: bật/tắt.
8. **Encoder chung cho BM25** (proxy embeddings) khác nhau.

> Kết xuất một bảng/đồ thị: mỗi hàng một cấu hình, cột NDCG\@5/20; highlight best.

---

# 7) Ví dụ lệnh chạy (gợi ý)

```bash
# 0) Tạo conda
pip install -U pip
pip install -r requirements.txt

# 1) Chuẩn hoá dữ liệu + chunk/proposition
python scripts/00_prepare_local_texts.py --in data/raw --out data/beir
python scripts/01_chunk_and_units.py --in data/beir/corpus.jsonl --chunk-len 400 --stride 100 --out data/beir/corpus.jsonl
python scripts/02_generate_queries_unsloth.py --corpus data/beir/corpus.jsonl --out-queries data/beir/queries.jsonl --out-qrels data/beir/qrels_test.tsv --mode both  # q & sq

# 2) Build indices
python scripts/03_build_indices.py --corpus data/beir/corpus.jsonl --out indices --retrievers bm25,mpnet,contriever --gran qd,qp,sqd,sqp

# 3) Retrieve
python scripts/04_retrieve.py --queries data/beir/queries.jsonl --indices indices --topk 100 --out runs

# 4) Tính trọng số MoR
python scripts/05_compute_weights_mor.py --corpus data/beir/corpus.jsonl --runs runs --kmeans-k 128 --topk 20 --out weights --mode both

# 5) Fusion + Eval
python scripts/06_fuse_and_eval.py \
  --qrels data/beir/qrels_test.tsv \
  --runs runs \
  --fusion normalized_sum rrf weighted_sum_mor_pre weighted_sum_mor_post \
  --weights weights \
  --out fusion

# 6) Ablation sweep
python scripts/07_ablation.py --plan configs/ablation.yaml --out results/ablation.csv
```

---

# Lưu ý & bẫy thường gặp

* **Id chuẩn hoá** giữa chunk/prop phải **nhất quán** (vd: `doc-1-ch0-p3`). Điều này quyết định logic `extract_base_pid` và mapping qrels.
* **Chỉ mục riêng cho từng granularity**; đừng lẫn `d` và `p`.
* **BM25** không có embedding → cần **proxy embeddings** (một encoder chung) cho phần **MoR** (KMeans/Moran).
* **Seed & cache**: cố định seed KMeans/FAISS để kết quả lặp lại.
* **Số KMeans cluster**: quá nhỏ → tín hiệu thô; quá lớn → nhiễu/chi phí cao.
* **Top-K cho post**: K=20/50 đủ cân bằng tốc độ/độ ổn định.

---


