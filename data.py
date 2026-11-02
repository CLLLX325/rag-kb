# -*- coding: utf-8 -*-
"""读语料 + 切分。

一个 RAG 系统的第一步永远是"把文档变成能检索的小块"。
这一步的产物（切片列表）贯穿后面所有环节，所以结构要固定：
    每个切片 = {id, doc_id, text}
"""

import json
import re
from pathlib import Path

import config


def load_corpus():
    """读取 corpus/ 下所有 .md，返回 [{doc_id, title, text}]。

    doc_id 用文件名（不含扩展名），比如 "01-return-policy"，
    评测集里的 gold 字段直接引用它。
    """
    docs = []
    for path in sorted(config.CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        first_line = text.splitlines()[0].lstrip("# ").strip()
        docs.append({
            "doc_id": path.stem,
            "title": first_line,
            "text": text,
        })
    return docs


def _split_sections(text):
    """按 markdown 二级标题切成小节。

    为什么先按标题切、再按长度切：
    标题天然是语义边界，顺着边界切出来的切片更"整"。
    """
    sections = []
    current_title, current_lines = "", []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title, current_lines = line[3:].strip(), []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(t, b) for t, b in sections if b]


def _window(text, size, overlap):
    """把一段长文本按固定长度开窗，相邻窗口重叠 overlap 个字符。"""
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text) - overlap, step)
            if text[i:i + size].strip()]


def chunk_documents(docs, size, overlap_ratio=None):
    """把文档切成固定长度的切片列表。

    每个切片前面会带上「文档标题 / 小节标题」，
    因为一个小片段脱离上下文后往往看不出来在讲什么
    （比如"48 小时内完成审核"——不说清楚是谁在审什么就废了）。
    """
    overlap = int(size * (config.CHUNK_OVERLAP_RATIO if overlap_ratio is None else overlap_ratio))
    chunks = []
    for doc in docs:
        buf, seq = "", 0

        def flush():
            """把攒好的缓冲区变成一个切片。用闭包是为了少传一堆参数。"""
            nonlocal buf, seq
            if buf.strip():
                chunks.append({
                    "id": f"{doc['doc_id']}#{seq}",
                    "doc_id": doc["doc_id"],
                    "text": f"【{doc['title']}】\n{buf.strip()}",
                })
                seq += 1
            buf = ""

        for sec_title, body in _split_sections(doc["text"]):
            block = f"## {sec_title}\n{body}" if sec_title else body
            if len(block) > size:
                # 单个小节就超过目标长度，只能按字数开窗
                flush()
                for piece in _window(block, size, overlap):
                    buf = piece
                    flush()
            elif len(buf) + len(block) + 1 <= size:
                # 小节能装下就合并，避免切片碎成一句话一条
                buf = f"{buf}\n{block}" if buf else block
            else:
                flush()
                buf = block
        flush()
    return chunks


def load_eval_set():
    """读取 36 条评测集，每行一个 JSON 对象。"""
    rows = []
    with open(config.EVAL_SET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text):
    """把文本压成可比较的形式，用于调试时定位切片来源。"""
    return re.sub(r"\s+", "", text)
