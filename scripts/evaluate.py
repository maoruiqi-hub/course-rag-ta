r"""轻量检索评测：检查 8 个问题能否命中预期幻灯片。

这个脚本只做检索，不调用对话模型，因此成本低、结果可复现。
运行：
    .\.venv\Scripts\python.exe scripts\evaluate.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from course_ta.config import build_embeddings, get_settings
from course_ta.documents import load_and_split
from course_ta.knowledge_base import KnowledgeBase

QUESTIONS = [
    ("RAG 的两个核心组件是什么？", 2),
    ("文档切分使用什么参数？", 3),
    ("stuff 和 map-reduce 有什么区别？", 4),
    ("PPTX 的引用如何标注？", 5),
    ("Agent 有哪三个工具？", 6),
    ("这个项目用什么框架做界面？", 1),
    ("知识库向量存储用什么？", 2),
    ("chunk_overlap 是多少？", 3),
]


def main() -> int:
    settings = get_settings()
    sample = Path(__file__).resolve().parents[1] / "data" / "samples" / "demo_langchain.pptx"
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        temp_settings = replace(settings, chroma_dir=Path(tmp) / "chroma")
        kb = KnowledgeBase(temp_settings, build_embeddings(temp_settings))
        kb.add_documents(load_and_split(sample), "eval-demo")

        hits = 0
        print(f"{'#':>2}  {'命中':<4}  {'预期':<6}  Top-5 页码")
        print("-" * 68)
        for index, (question, expected) in enumerate(QUESTIONS, start=1):
            results = kb.search(question, k=5)
            pages = [doc.metadata.get("page") for doc, _ in results]
            hit = expected in pages
            hits += int(hit)
            print(f"{index:>2}  {'✓' if hit else '✗':<4}  {expected:<6}  {pages}  {question}")
        print("-" * 68)
        print(f"Hit@5: {hits}/{len(QUESTIONS)} = {hits / len(QUESTIONS):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
