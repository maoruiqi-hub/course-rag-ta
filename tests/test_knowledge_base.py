from __future__ import annotations

from langchain_core.documents import Document

from course_ta.knowledge_base import KnowledgeBase


def make_docs(source: str = "demo.pptx") -> list[Document]:
    return [
        Document(
            page_content="RAG 包含索引、检索和生成三个环节。",
            metadata={
                "source": source,
                "file_hash": "hash-a",
                "file_type": "pptx",
                "page": 1,
                "page_label": "第 1 张幻灯片",
                "location": "第 1 张幻灯片",
                "section": "",
            },
        ),
        Document(
            page_content="文本切分可以使用 RecursiveCharacterTextSplitter。",
            metadata={
                "source": source,
                "file_hash": "hash-a",
                "file_type": "pptx",
                "page": 2,
                "page_label": "第 2 张幻灯片",
                "location": "第 2 张幻灯片",
                "section": "",
            },
        ),
    ]


def test_add_search_dedup_and_clear(settings, fake_embeddings) -> None:
    kb = KnowledgeBase(settings, fake_embeddings)
    assert kb.count() == 0
    assert kb.add_documents(make_docs(), "hash-a") == 2
    assert kb.count() == 2
    assert kb.has_file("hash-a")
    assert not kb.has_file("hash-b")

    assert kb.add_documents(make_docs(), "hash-a") == 0
    assert kb.count() == 2

    results = kb.search("RAG 的组成", k=2)
    assert results
    assert all(0.0 <= score <= 1.0 for _, score in results)
    assert results[0][0].metadata["source"] == "demo.pptx"

    files = kb.list_files()
    assert len(files) == 1
    assert files[0]["chunks"] == 2
    assert files[0]["page_count"] == 2

    assert kb.delete_file("hash-a") == 2
    assert kb.count() == 0

    kb.add_documents(make_docs(), "hash-a")
    kb.clear()
    assert kb.count() == 0


def test_empty_search(settings, fake_embeddings) -> None:
    kb = KnowledgeBase(settings, fake_embeddings)
    assert kb.search("不存在", k=3) == []
    assert kb.list_files() == []
    assert kb.stats() == {"files": 0, "chunks": 0}
