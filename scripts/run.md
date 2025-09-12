python scripts/00_prepare_local_texts.py --in data/raw --out data/beir | cat

python scripts/01_chunk_and_units.py --in data/beir/corpus.jsonl --chunk-len 400 --stride 100 --out data/beir/corpus.jsonl | cat

python scripts/02_generate_queries_unsloth.py --corpus data/beir/corpus.jsonl --out-queries data/beir/queries.jsonl --out-qrels data/beir/qrels_test.tsv --mode both --per-doc 1 | cat


python scripts/03_build_indices.py --corpus data/beir/corpus.jsonl --out indices --retrievers bm25,mpnet --gran qd | cat

python scripts/04_retrieve.py --queries data/beir/queries.jsonl --indices indices --topk 50 --out runs | cat

python scripts/05_compute_weights_mor.py --queries data/beir/queries.jsonl --indices indices --out weights --kmeans-k 4 | cat

python scripts/06_fuse_and_eval.py --qrels data/beir/qrels_test.tsv --runs runs --fusion weighted_sum_mor_pre --weights weights --out fusion --topk 50 | cat

python scripts/07_ablation.py --out results/ablation.csv | cat

python scripts/07_ablation.py --plan configs/ablation.yaml --out results/ablation.csv | cat




python scripts/03_build_indices.py --corpus data/beir/corpus.jsonl --out indices --retrievers "bm25,mpnet,contriever,simcse,dpr,ance,tas-b,gtr"


python scripts/04_retrieve.py --queries data/beir/queries.jsonl --indices indices --topk 100 --out runs


python scripts/05_compute_weights_mor.py --queries data/beir/queries.jsonl --indices indices --runs runs --out weights --kmeans-k 64 --topk 20 --mode post


python scripts/06_fuse_and_eval.py --qrels data/beir/qrels_test.tsv --runs runs --fusion normalized_sum rrf weighted_sum_mor_post --weights weights --out fusion


