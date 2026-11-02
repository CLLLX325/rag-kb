# -*- coding: utf-8 -*-
"""单元测试：只测"逻辑正确性"，不依赖模型服务。

为什么要有测试：
改代码时最怕的是"看起来还能跑，其实把排序弄反了"。
下面四个用例覆盖了最容易出错的地方：切分、排序、融合、指标计算。
"""

import config
import data
from metrics import docs_at_k, evaluate
from retrievers import BM25Retriever, HybridRetriever

DOCS = data.load_corpus()


def test_corpus_loaded():
    """语料必须全部读到，评测集 gold 必须都能对上文档。"""
    # 不写死篇数：扩充语料时不该改测试，但至少要保证语料没有加载一半
    assert len(DOCS) >= 20
    assert len({d["doc_id"] for d in DOCS}) == len(DOCS), "存在重复 doc_id"
    doc_ids = {d["doc_id"] for d in DOCS}
    for row in data.load_eval_set():
        assert set(row["gold"]) <= doc_ids, f"{row['id']} 的 gold 不在语料里"


def test_chunk_size_respected():
    """除了极短文档，每个切片都不应显著超过目标长度（允许标题前缀的少量溢出）。"""
    for size in config.CHUNK_SIZES:
        chunks = data.chunk_documents(DOCS, size)
        assert len(chunks) > 10
        oversize = [c for c in chunks if len(c["text"]) > size * 1.6]
        assert not oversize, f"size={size} 有 {len(oversize)} 个切片超长"


def test_chunk_smaller_size_more_chunks():
    """切片越小，切出来的块越多——这是切分逻辑的单调性。"""
    counts = [len(data.chunk_documents(DOCS, s)) for s in (250, 400, 600)]
    assert counts[0] > counts[1] > counts[2]


def test_bm25_finds_exact_term():
    """BM25 对精确名词必须有效：问"运费险"应该命中运费文档。"""
    chunks = data.chunk_documents(DOCS, 400)
    retriever = BM25Retriever(chunks)
    hits = retriever.search("运费险赔付上限", k=3)
    top_docs = [chunks[i]["doc_id"] for i, _ in hits]
    assert "03-shipping-fee" in top_docs


def test_hybrid_scores_sorted_desc():
    """融合后的排名必须严格降序，否则说明排序写反了。"""
    chunks = data.chunk_documents(DOCS, 400)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever([bm25, bm25])   # 同一个检索器用两次，分数应翻倍且顺序不变
    hits = hybrid.search("退款多久到账", k=5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_metrics_synthetic():
    """用一个构造出来的"完美检索器"验证指标计算没有写错。"""
    chunks = data.load_corpus()
    chunks = [{"id": d["doc_id"], "doc_id": d["doc_id"], "text": d["text"]} for d in chunks]
    rows = data.load_eval_set()[:4]

    class Perfect:
        name = "perfect"

        def search(self, query, k=10):
            gold = next(r["gold"][0] for r in rows if r["query"] == query)
            idx = next(i for i, c in enumerate(chunks) if c["doc_id"] == gold)
            return [(idx, 1.0)]

    result = evaluate(Perfect(), rows, chunks)
    assert result["recall@1"] == 1.0
    assert result["mrr"] == 1.0


def test_docs_at_k_dedup():
    """同一文档的多个切片只能算一个名额。"""
    chunks = [{"doc_id": "a"}, {"doc_id": "a"}, {"doc_id": "b"}]
    assert docs_at_k([(0, 1.0), (1, 0.9), (2, 0.8)], chunks, 3) == ["a", "b"]
