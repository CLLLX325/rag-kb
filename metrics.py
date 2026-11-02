# -*- coding: utf-8 -*-
"""评测指标。

为什么不用"看着还不错"来判断 RAG 好坏：
因为检索结果太容易产生错觉——你随手问几个问题都能出答案，但换个说法就崩了。
有了固定评测集 + 固定指标，才敢说"这个改动让系统变好了"。

指标含义：
    Recall@k  前 k 个结果里有没有正确答案（多选题：有就算对）
    MRR       第一个正确答案排在第几名，取倒数（第 1 名=1.0，第 3 名=0.33）
    P95 延迟  95% 的请求有多快，比平均值更能反映真实体验
"""

import time

import numpy as np


def docs_at_k(hits, chunks, k):
    """把"切片排名"转成"文档排名"（去重，保留第一次出现的位置）。

    为什么要去重到文档级：
    同一篇文档可能被切成 5 个切片，如果按切片算 Recall@3，
    一篇文档就占了 3 个名额，指标会虚高。
    """
    seen = []
    for idx, _score in hits[:k]:
        doc_id = chunks[idx]["doc_id"]
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def evaluate(retriever, rows, chunks, ks=(1, 5, 10), top_k=10):
    """跑完整个评测集，返回一个可以直接打印的指标字典。"""
    per_type = {}
    latencies = []
    hit_at = {k: 0 for k in ks}
    rr = 0.0

    for row in rows:
        t0 = time.perf_counter()
        hits = retriever.search(row["query"], k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        ranked = docs_at_k(hits, chunks, top_k)
        gold = set(row["gold"])
        rank = next((i + 1 for i, d in enumerate(ranked) if d in gold), None)

        if rank:
            rr += 1.0 / rank
        for k in ks:
            if set(ranked[:k]) & gold:
                hit_at[k] += 1

        bucket = per_type.setdefault(row["type"], {"n": 0, "hit1": 0, "rr": 0.0})
        bucket["n"] += 1
        bucket["hit1"] += int(bool(rank) and rank <= 1)
        bucket["rr"] += (1.0 / rank) if rank else 0.0

    n = len(rows)
    result = {
        "name": retriever.name,
        **{f"recall@{k}": hit_at[k] / n for k in ks},
        "mrr": rr / n,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "n": n,
    }
    for t, b in per_type.items():
        result[f"{t}.recall@1"] = b["hit1"] / b["n"]
        result[f"{t}.mrr"] = b["rr"] / b["n"]
    return result
