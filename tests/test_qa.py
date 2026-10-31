from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from course_ta.qa import (
    answer_question,
    condense_question,
    dedupe_retrieval,
    format_context,
    history_to_messages,
    make_sources,
    recent_history,
    route_summary_mode,
    summarize_documents,
)


def doc(text: str, page: int = 1) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "demo.pptx",
            "file_hash": "h",
            "file_type": "pptx",
            "page": page,
            "page_label": f"第 {page} 张幻灯片",
            "location": f"第 {page} 张幻灯片",
            "section": "",
        },
    )


def test_format_context_and_sources() -> None:
    docs = [doc("RAG 有索引和检索生成两个组件。", 1), doc("切分使用递归字符切分器。", 2)]
    context = format_context(docs)
    assert "[1] 文件《demo.pptx》 第 1 张幻灯片" in context
    assert "[2]" in context

    sources = make_sources([(docs[0], 0.9), (docs[0], 0.8), (docs[1], 0.6)])
    assert len(sources) == 2
    assert sources[0].index == 1
    assert sources[1].index == 2
    assert sources[0].to_dict()["score"] == 0.9


def test_dedupe_retrieval_keeps_source_indices_aligned() -> None:
    docs = [doc("RAG 有索引和检索生成两个组件。", 1), doc("切分使用递归字符切分器。", 2)]
    pairs = dedupe_retrieval([(docs[0], 0.9), (docs[0], 0.8), (docs[1], 0.6)])
    assert len(pairs) == 2
    context = format_context([document for document, _ in pairs])
    sources = make_sources(pairs)
    assert "[1]" in context and "[2]" in context
    assert [source.index for source in sources] == [1, 2]
    assert [source.page_label for source in sources] == ["第 1 张幻灯片", "第 2 张幻灯片"]


def test_route_summary_mode() -> None:
    short = [doc("短文本")]
    long = [doc("长" * 6000)]
    assert route_summary_mode(short, threshold=100) == "stuff"
    assert route_summary_mode(long, threshold=100) == "map-reduce"
    assert route_summary_mode(short, force_map=True) == "map-reduce"


def test_summarize_stuff() -> None:
    llm = FakeListChatModel(responses=["stuff 摘要"])
    result = summarize_documents([doc("RAG 内容")], llm, threshold=1000)
    assert result.mode == "stuff"
    assert result.text == "stuff 摘要"
    assert result.chunks == 1


def test_summarize_map_reduce_and_progress() -> None:
    llm = FakeListChatModel(responses=["局部一", "局部二", "最终摘要"])
    docs = [doc("内容" * 1000, page=index) for index in range(1, 5)]
    progress: list[tuple[int, int]] = []
    result = summarize_documents(
        docs,
        llm,
        threshold=1000,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert result.mode == "map-reduce"
    assert result.text == "最终摘要"
    assert progress == [(1, 2), (2, 2)]


def test_history_helpers_and_condense() -> None:
    history = [
        {"role": "user", "content": "RAG 是什么？"},
        {"role": "assistant", "content": "检索增强生成。"},
        {"role": "user", "content": "它有什么缺点？"},
    ]
    messages = history_to_messages(history)
    assert len(messages) == 3
    assert messages[0].content == "RAG 是什么？"
    assert recent_history(history, 2) == history[-2:]

    llm = FakeListChatModel(responses=["RAG 有什么缺点？"])
    assert condense_question(llm, history, "它有什么缺点？") == "RAG 有什么缺点？"
    assert condense_question(llm, [], "原问题") == "原问题"


def test_answer_question_with_fake_llm() -> None:
    llm = FakeListChatModel(responses=["答案 [1]"])
    answer = answer_question(llm, "RAG 是什么？", [doc("RAG 是检索增强生成。")], [])
    assert answer == "答案 [1]"
