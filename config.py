# -*- coding: utf-8 -*-
"""全局配置。

设计原则：所有"可以调的东西"集中在这里。
这样别人读代码时只要看这一个文件，就知道项目有哪些旋钮。
"""

from pathlib import Path

ROOT = Path(__file__).parent
CORPUS_DIR = ROOT / "corpus"        # 12 篇政策文档
EVAL_SET_PATH = ROOT / "eval_set.jsonl"  # 36 条评测集
CACHE_DIR = ROOT / ".cache"         # 向量缓存，避免重复调用模型

# ---------- 模型服务 ----------
OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "embeddinggemma:300m-qat-q8_0"   # 768 维
CHAT_MODEL = "qwen2.5:7b"

# ---------- 切分 ----------
# 为什么要试三个值：切片太短会丢上下文，太长会稀释语义，这是 RAG 最常见的调参点
CHUNK_SIZES = [250, 400, 600]
CHUNK_OVERLAP_RATIO = 0.2   # 相邻切片重叠 20%，避免答案正好被切断

# ---------- 检索 ----------
TOP_K = 10          # 评测时取前 10 条
RRF_K = 60          # RRF 公式里的平滑常数，原论文取 60
