# -*- coding: utf-8 -*-
"""用 bge-reranker-v2-m3 做重排（cross-encoder）。

为什么要重排：
    第一阶段检索（BM25 / 向量）是"双塔"结构：查询和文档分别编码成向量再比相似度，
    快，但查询和文档之间没有真正"互相看见"。
    cross-encoder 把 (查询, 文档) 拼成一个序列送进模型，让两者在注意力层里直接交互，
    精度高得多，代价是每条候选都要跑一次前向，慢。

    所以工程上的标准做法是两段式：
        召回 取 Top-10（快、便宜） → 重排 只对这 10 条精算（准、但次数少）

    这也正好对应 P1 的发现：向量检索 Recall@5 = 1.000 但 Recall@1 只有 0.63~0.78，
    说明"正确答案一定在候选里，只是排得不准"——重排能解决的就是这个问题。

为什么不用 sentence-transformers：
    bge-reranker-v2-m3 本质就是一个 XLMRobertaForSequenceClassification，
    用 transformers 直接加载只要十几行，少一个依赖、少一层抽象。
"""

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = r"C:\deepwork\models\bge-reranker-v2-m3"


class BgeReranker:
    """把 (query, doc) 打分，分数越高越相关。"""

    def __init__(self, model_dir=MODEL_DIR, max_length=512, batch_size=8, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.calls = 0  # 统计真正跑了多少次前向，用来算延迟成本

    @torch.no_grad()
    def score(self, query, docs):
        """返回与 docs 等长的分数列表。"""
        scores = []
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i:i + self.batch_size]
            pairs = [[query, d] for d in batch]
            inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits.view(-1).float()
            scores.extend(logits.cpu().tolist())
            self.calls += len(batch)
        return scores

    def rerank(self, query, hits, chunks, top_k=10):
        """对第一阶段的候选重排，返回 [(chunk_idx, score)]。"""
        if not hits:
            return []
        docs = [chunks[idx]["text"] for idx, _ in hits]
        scores = self.score(query, docs)
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        return [(hits[i][0], scores[i]) for i in order[:top_k]]
