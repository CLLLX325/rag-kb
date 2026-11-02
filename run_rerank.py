# -*- coding: utf-8 -*-
"""重排实验：向量召回 Top-10，再用 cross-encoder 重排，看 R@1 能不能推上去。

实验设计：
    对照组 = 纯向量检索（P1 already 跑过）
    实验组 = 向量召回 10 条候选，再用 bge-reranker-v2-m3 重排

    切片长度 250 / 400 / 600 都跑一遍，因为 P1 发现向量在短切片上表现更好，
    重排的价值可能随切片长度变化。

注意：这里的 P95 延迟包含重排耗时，是诚实的端到端数字，不是"只算第一段"的数字。
"""

import time

import config
import data
from embedder import OllamaEmbedder
from metrics import evaluate
from rerank import BgeReranker
from retrievers import VectorRetriever


class RerankRetriever:
    """两段式检索器：先向量召回 candidate_k 条，再重排取前 top_k。"""

    def __init__(self, first_stage, reranker, chunks, candidate_k=10, top_k=10):
        self.first_stage = first_stage
        self.reranker = reranker
        self.chunks = chunks
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.name = first_stage.name + "+rerank"

    def search(self, query, k=10):
        hits = self.first_stage.search(query, k=max(self.candidate_k, k))
        return self.reranker.rerank(query, hits, self.chunks, top_k=k)


def main():
    docs = data.load_corpus()
    rows = data.load_eval_set()
    embedder = OllamaEmbedder()

    print("加载重排模型（首次约 10 秒）...")
    reranker = BgeReranker()
    print(f"设备：{reranker.device}")

    table = []
    for size in config.CHUNK_SIZES:
        chunks = data.chunk_documents(docs, size)
        # VectorRetriever 自己会调 embedder 建矩阵，这里只要把 embedder 传进去
        base = VectorRetriever(chunks, embedder)
        combined = RerankRetriever(base, reranker, chunks)

        t0 = time.perf_counter()
        base_res = evaluate(base, rows, chunks)
        base_time = time.perf_counter() - t0
        rerank_res = evaluate(combined, rows, chunks, top_k=10)

        table.append((size, "vector", base_res))
        table.append((size, "vector+rerank", rerank_res))
        print(f"切片 {size}：召回 R@1={base_res['recall@1']:.3f} → 重排后 {rerank_res['recall@1']:.3f}"
              f"（P95 {base_res['p95_ms']:.0f}ms → {rerank_res['p95_ms']:.0f}ms）")

    print()
    header = "| 切片 | 检索方式 | R@1 | R@5 | MRR | P95(ms) |"
    print(header)
    print("|---|---|---|---|---|---|")
    for size, name, res in table:
        print(f"| {size} | {name} | {res['recall@1']:.3f} | {res['recall@5']:.3f} "
              f"| {res['mrr']:.3f} | {res['p95_ms']:.0f} |")

    print(f"\n重排模型累计打分次数：{reranker.calls}")
    import json
    with open("results-rerank.json", "w", encoding="utf-8") as f:
        json.dump([{"size": s, "mode": n, **r} for s, n, r in table], f,
                  ensure_ascii=False, indent=2)
    print("结果已写入 results-rerank.json")


if __name__ == "__main__":
    main()
