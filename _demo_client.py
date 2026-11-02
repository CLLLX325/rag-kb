# -*- coding: utf-8 -*-
"""冒烟脚本：启动服务后跑一遍两个接口，确认能用。

用法：
    另开一个终端跑  uvicorn app:app --port 8123
    然后跑本脚本   python _demo_client.py
"""

import json

import httpx

BASE = "http://127.0.0.1:8123"


def main():
    # 1. 只检索：看命中了哪些政策片段
    r = httpx.get(f"{BASE}/search", params={"q": "退款多久能到账", "k": 3}, timeout=60)
    r.raise_for_status()
    data = r.json()
    print("== /search ==")
    print("耗时", data["latency_ms"], "ms")
    for h in data["hits"]:
        print(f"  {h['rank']}. {h['doc_id']}  分数 {h['score']}")
        print(f"     {h['text'][:60]}...")

    # 2. 检索 + 生成：看答案和引用是否正确
    r = httpx.post(f"{BASE}/chat", json={"question": "退款多久能到账？", "top_k": 3}, timeout=180)
    r.raise_for_status()
    data = r.json()
    print("\n== /chat ==")
    print("耗时", data["latency_ms"], "ms")
    print("答案：", data["answer"])
    print("引用：", json.dumps(data["citations"], ensure_ascii=False))

    # 3. 故意问一个语料里没有的问题，验证"不瞎编"
    r = httpx.post(f"{BASE}/chat", json={"question": "你们支持比特币支付吗？", "top_k": 3}, timeout=180)
    data = r.json()
    print("\n== /chat（语料外问题） ==")
    print("答案：", data["answer"])


if __name__ == "__main__":
    main()
