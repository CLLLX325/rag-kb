# -*- coding: utf-8 -*-
"""跑完整实验矩阵：切片长度 × 检索方式。

为什么要把实验做成一个脚本：
因为"我觉得混合检索更好"和"混合检索在转述问法上 Recall@1 从 0.44 提到 0.72"
是两件事。前者面试官不信，后者是简历上能写的数字。

用法：
    python run_experiment.py
输出：
    终端打印 Markdown 表格，同时写入 results.json
"""

import json

import config
import data
from embedder import OllamaEmbedder
from metrics import evaluate
from retrievers import BM25Retriever, HybridRetriever, VectorRetriever


def build_retrievers(chunks, embedder):
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, embedder)
    # 混合检索内部再包一层，所以它俩的排名会被融合
    hybrid = HybridRetriever([BM25Retriever(chunks), VectorRetriever(chunks, embedder)])
    return [bm25, vector, hybrid]


def main():
    docs = data.load_corpus()
    rows = data.load_eval_set()
    embedder = OllamaEmbedder()
    print(f"语料 {len(docs)} 篇，评测集 {len(rows)} 条\n")

    all_results = []
    for size in config.CHUNK_SIZES:
        chunks = data.chunk_documents(docs, size)
        print(f"=== 切片 {size} 字：共 {len(chunks)} 个切片 ===")
        for retriever in build_retrievers(chunks, embedder):
            r = evaluate(retriever, rows, chunks)
            r["chunk_size"] = size
            r["chunks"] = len(chunks)
            all_results.append(r)
            print(f"{r['name']:7s} R@1={r['recall@1']:.3f} R@5={r['recall@5']:.3f} "
                  f"R@10={r['recall@10']:.3f} MRR={r['mrr']:.3f} "
                  f"P95={r['p95_ms']:.1f}ms | 直问 R@1={r.get('direct.recall@1', 0):.3f} "
                  f"转述 R@1={r.get('paraphrase.recall@1', 0):.3f}")
        print()

    # 打印成 Markdown 表格，可以直接贴进 README 或简历
    print("\n| 切片 | 检索方式 | R@1 | R@5 | R@10 | MRR | P95(ms) |")
    print("|---|---|---|---|---|---|---|")
    for r in all_results:
        print(f"| {r['chunk_size']} | {r['name']} | {r['recall@1']:.3f} | {r['recall@5']:.3f} | "
              f"{r['recall@10']:.3f} | {r['mrr']:.3f} | {r['p95_ms']:.1f} |")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n结果已写入 results.json；本次真实调用 embedding 接口 "
          f"{embedder.calls} 次（其余命中缓存）")


if __name__ == "__main__":
    main()
