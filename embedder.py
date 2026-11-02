# -*- coding: utf-8 -*-
"""把文本变成向量（调 Ollama 的 /api/embed）。

为什么要单独一个文件：
embedding 是 RAG 里最贵的操作，也是最需要"缓存"的操作。
把它封装起来，后面无论切片怎么变，都只走这一个入口。

缓存的意义：跑实验时要对同一批切片反复算向量（换个检索器还要重算吗？不用），
第二次直接读磁盘，一次实验从几分钟降到几秒。
"""

import hashlib
import json

import httpx
import numpy as np

import config


class OllamaEmbedder:
    def __init__(self, model=None, cache_name=None):
        self.model = model or config.EMBED_MODEL
        # 用模型名区分缓存，换模型不会读到错的数据
        self.cache_path = config.CACHE_DIR / f"emb-{cache_name or self.model.replace(':', '_')}.json"
        self.cache = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.client = httpx.Client(timeout=180.0)
        self.calls = 0   # 统计真正打了几次网络请求，用来验证缓存有没有生效

    @staticmethod
    def _key(text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _embed_batch(self, texts):
        """一次请求编码一批文本。Ollama 会按输入顺序返回同样数量的向量。"""
        resp = self.client.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        self.calls += 1
        return resp.json()["embeddings"]

    def embed(self, texts, batch_size=16):
        """返回 shape=(len(texts), dim) 的 float32 矩阵。命中的走缓存。"""
        todo, todo_idx = [], []
        for i, t in enumerate(texts):
            if self._key(t) not in self.cache:
                todo.append(t)
                todo_idx.append(i)

        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            for text, vec in zip(batch, self._embed_batch(batch)):
                self.cache[self._key(text)] = [round(x, 6) for x in vec]  # 少存小数点，文件小一半

        if todo:
            config.CACHE_DIR.mkdir(exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

        arr = np.array([self.cache[self._key(t)] for t in texts], dtype=np.float32)
        return arr

    def embed_one(self, text):
        return self.embed([text])[0]
