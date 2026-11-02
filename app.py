# -*- coding: utf-8 -*-
"""对外提供两个接口的 HTTP 服务。

    GET  /search?q=...&k=5   只做检索，返回命中的政策片段（方便调试）
    POST /chat               检索 + 生成，返回带引用的答案

为什么检索和生成分成两个接口：
调试 RAG 时 90% 的问题出在检索。如果只能调 /chat，你看到的是一段错答案，
分不清是"没找到"还是"找到了但模型说错"。拆开之后一眼就能定位。
"""

import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
import data
from embedder import OllamaEmbedder
from retrievers import HybridRetriever, VectorRetriever

app = FastAPI(title="售后政策 RAG 知识库", version="1.0")

# 全局状态：启动时构建一次，之后所有请求复用（不要每个请求都重算向量）
STATE = {}


@app.on_event("startup")
def build_index():
    docs = data.load_corpus()
    chunks = data.chunk_documents(docs, 400)      # 实验里 400 字这档最好
    embedder = OllamaEmbedder()
    vector = VectorRetriever(chunks, embedder)

    from retrievers import BM25Retriever
    hybrid = HybridRetriever([BM25Retriever(chunks), VectorRetriever(chunks, embedder)])

    STATE.update({"chunks": chunks, "embedder": embedder,
                  "vector": vector, "hybrid": hybrid})
    print(f"索引就绪：{len(chunks)} 个切片")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=200)
    top_k: int = Field(3, ge=1, le=10)


SYSTEM_PROMPT = """你是电商平台的售后客服。请严格遵守：
1. 只依据【参考资料】回答，不要使用资料以外的知识。
2. 在引用了某条资料的句子末尾标注 [1]、[2]，编号对应资料序号。
3. 如果参考资料不足以回答，直接说"根据现有政策无法确认"，不要猜测。
4. 回答控制在 150 字以内，先说结论，再说条件。"""


def _build_context(hits, chunks):
    lines = []
    for i, (idx, _score) in enumerate(hits, 1):
        lines.append(f"[{i}] {chunks[idx]['text']}")
    return "\n\n".join(lines)


@app.get("/search")
def search(q: str, k: int = 5, mode: str = "vector"):
    if mode not in ("vector", "hybrid"):
        raise HTTPException(400, "mode 只能是 vector 或 hybrid")
    t0 = time.perf_counter()
    hits = STATE[mode].search(q, k=k)
    cost = (time.perf_counter() - t0) * 1000
    return {
        "query": q,
        "mode": mode,
        "latency_ms": round(cost, 2),
        "hits": [
            {"rank": i, "doc_id": STATE["chunks"][idx]["doc_id"],
             "score": round(score, 4), "text": STATE["chunks"][idx]["text"][:120]}
            for i, (idx, score) in enumerate(hits, 1)
        ],
    }


@app.post("/chat")
def chat(req: ChatRequest):
    t0 = time.perf_counter()
    hits = STATE["vector"].search(req.question, k=req.top_k)
    if not hits:
        return {"answer": "根据现有政策无法确认。", "citations": [], "latency_ms": 0}

    context = _build_context(hits, STATE["chunks"])
    payload = {
        "model": config.CHAT_MODEL,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 400},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】{req.question}"},
        ],
    }
    resp = httpx.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=180)
    resp.raise_for_status()
    answer = resp.json()["message"]["content"]

    return {
        "answer": answer,
        "citations": [
            {"id": i, "doc_id": STATE["chunks"][idx]["doc_id"]}
            for i, (idx, _s) in enumerate(hits, 1)
        ],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
