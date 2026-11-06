r"""检索评测：检查问题能否命中人工标注的幻灯片 / 页码。

只做检索、不调用对话模型，因此成本低、结果可复现。

运行：
    # 1) 仓库自带的 6 页演示课件（默认）
    python scripts\evaluate.py

    # 2) 真实课程课件（需自备 PPTX）
    python scripts\evaluate.py --file "第7章.pptx" --questions data\eval\chapter7_questions.json

    # 输出 Markdown 表格，方便贴进文档
    python scripts\evaluate.py ... --markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from course_ta.config import build_embeddings, get_settings  # noqa: E402
from course_ta.documents import load_and_split  # noqa: E402
from course_ta.knowledge_base import KnowledgeBase  # noqa: E402

DEFAULT_FILE = ROOT / "data" / "samples" / "demo_langchain.pptx"
DEFAULT_QUESTIONS = [
    {"q": "RAG 的两个核心组件是什么？", "expected": [2]},
    {"q": "文档切分使用什么参数？", "expected": [3]},
    {"q": "stuff 和 map-reduce 有什么区别？", "expected": [4]},
    {"q": "PPTX 的引用如何标注？", "expected": [5]},
    {"q": "Agent 有哪三个工具？", "expected": [6]},
    {"q": "这个项目用什么框架做界面？", "expected": [1]},
    {"q": "知识库向量存储用什么？", "expected": [2]},
    {"q": "chunk_overlap 是多少？", "expected": [3]},
]


def unique_pages(results) -> list[int]:
    """同一页可能被切成多个片段，按出现顺序去重后再算排名。"""
    pages: list[int] = []
    for document, _ in results:
        page = document.metadata.get("page")
        if isinstance(page, int) and page not in pages:
            pages.append(page)
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索评测（Hit@K / MRR）")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="要评测的课件")
    parser.add_argument("--questions", type=Path, help="问题 JSON，见 data/eval/")
    parser.add_argument("--k", type=int, default=5, help="检索片段数")
    parser.add_argument("--markdown", action="store_true", help="输出 Markdown 表格")
    args = parser.parse_args()

    questions = DEFAULT_QUESTIONS
    if args.questions:
        questions = json.loads(args.questions.read_text(encoding="utf-8"))["questions"]

    settings = get_settings()
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        temp_settings = replace(settings, chroma_dir=Path(tmp) / "chroma")
        kb = KnowledgeBase(temp_settings, build_embeddings(temp_settings))
        chunks = load_and_split(args.file)
        kb.add_documents(chunks, "eval")

        rows = []
        hit1 = hit3 = hitk = 0
        reciprocal = 0.0
        for item in questions:
            expected = set(item["expected"])
            pages = unique_pages(kb.search(item["q"], k=args.k))
            rank = next((i for i, page in enumerate(pages, 1) if page in expected), None)
            hit1 += int(rank == 1)
            hit3 += int(rank is not None and rank <= 3)
            hitk += int(rank is not None)
            reciprocal += 1 / rank if rank else 0.0
            rows.append((item["q"], sorted(expected), pages, rank))

    n = len(questions)
    print(f"资料：{args.file.name} · {len(chunks)} 个片段 · {n} 个问题 · Top-{args.k}\n")
    if args.markdown:
        print(f"| # | 问题 | 标注页 | Top-{args.k} 命中页（去重） | 排名 |")
        print("|---|---|---|---|---|")
        for index, (question, expected, pages, rank) in enumerate(rows, 1):
            mark = f"✓ {rank}" if rank else "✗"
            print(f"| {index} | {question} | {expected} | {pages} | {mark} |")
    else:
        for index, (question, expected, pages, rank) in enumerate(rows, 1):
            mark = f"✓{rank}" if rank else "✗ "
            print(f"{index:>2} {mark:<3} 标注 {expected!s:<10} 检索 {pages!s:<22} {question}")
    print(
        f"\nHit@1 = {hit1}/{n} ({hit1 / n:.0%}) · Hit@3 = {hit3}/{n} ({hit3 / n:.0%}) · "
        f"Hit@{args.k} = {hitk}/{n} ({hitk / n:.0%}) · MRR = {reciprocal / n:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
