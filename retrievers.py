# -*- coding: utf-8 -*-
"""三种检索器，接口完全一致：search(query, k) -> [(切片下标, 分数), ...]

为什么接口要一致：
这样评测脚本不需要 if/else，换检索器只是换一个对象。
真实工程里也是这么做的——上层不关心底层是 BM25 还是向量。

三者的区别，一句话：
    BM25  = 按"词有没有出现"打分，专治精确名词（批次号、专有名词）
    向量   = 按"意思像不像"打分，专治换个说法问同一件事
    RRF    = 把两边的排名融合，取长补短
"""

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

import config


def tokenize(text):
    """中文分词，并丢掉空白与标点。

    BM25 本质上在数"词频"，所以分词质量直接决定它的上限。
    这里用 jieba 的默认模式，够用且无需训练。
    """
    return [w for w in jieba.lcut(text) if w.strip() and not w.isspace()]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query, k=10):
        scores = self.bm25.get_scores(tokenize(query))
        # 只要前 k 个，用 argpartition 比全排序快，切 100 个以上时差别明显
        idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]


class VectorRetriever:
    name = "vector"

    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = embedder.embed([c["text"] for c in chunks])
        # 先归一化，之后余弦相似度就等于点积，省一次除法
        self.matrix = self.matrix / (np.linalg.norm(self.matrix, axis=1, keepdims=True) + 1e-9)

    def search(self, query, k=10):
        q = self.embedder.embed_one(query)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self.matrix @ q
        idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]


class HybridRetriever:
    """RRF（Reciprocal Rank Fusion，倒数排名融合）。

    公式：score(d) = Σ 1 / (k + rank_i(d))
    它只用"排名"不用"分数"，因此不需要把 BM25 的分数和余弦相似度对齐到同一量纲
    ——这是它被广泛使用的原因，几乎不需要调参。
    """

    name = "hybrid"

    def __init__(self, retrievers, k_const=None):
        self.retrievers = retrievers
        self.k_const = k_const or config.RRF_K
        self.chunks = retrievers[0].chunks

    def search(self, query, k=10):
        fused = {}
        for r in self.retrievers:
            for rank, (idx, _score) in enumerate(r.search(query, k=k * 2)):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.k_const + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        return ranked[:k]
