# -*- coding: utf-8 -*-
"""临时脚本：扫 RRF 的 k 常数，看混合检索能不能追上纯向量。"""
import config, data
from embedder import OllamaEmbedder
from metrics import evaluate
from retrievers import BM25Retriever, HybridRetriever, VectorRetriever

docs = data.load_corpus()
rows = data.load_eval_set()
emb = OllamaEmbedder()

for size in (250, 400):
    chunks = data.chunk_documents(docs, size)
    bm = BM25Retriever(chunks)
    ve = VectorRetriever(chunks, emb)
    print(f"--- size={size}, chunks={len(chunks)} ---")
    print("vector R@1=%.3f MRR=%.3f" % (evaluate(ve, rows, chunks)["recall@1"],
                                        evaluate(ve, rows, chunks)["mrr"]))
    for k in (1, 3, 5, 10, 20, 60):
        h = HybridRetriever([bm, ve], k_const=k)
        r = evaluate(h, rows, chunks)
        print(f"  rrf_k={k:3d} R@1={r['recall@1']:.3f} R@5={r['recall@5']:.3f} MRR={r['mrr']:.3f} "
              f"直问={r.get('direct.recall@1',0):.3f} 转述={r.get('paraphrase.recall@1',0):.3f}")
