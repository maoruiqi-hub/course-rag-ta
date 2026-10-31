from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from course_ta.agent import (
    calculator,
    extract_answer,
    extract_trace,
    get_current_time,
    make_search_tool,
)
from course_ta.knowledge_base import KnowledgeBase


def test_calculator() -> None:
    assert calculator.invoke({"expression": "(128*3+7)/2"}) == "195.5"
    assert calculator.invoke({"expression": "2**10"}) == "1024"
    assert "失败" in calculator.invoke({"expression": "__import__('os')"})
    assert "失败" in calculator.invoke({"expression": "1/0"})


def test_current_time() -> None:
    value = get_current_time.invoke({})
    assert "北京时间" in value
    assert len(value) >= 19


def test_search_tool_returns_page(settings, fake_embeddings) -> None:
    kb = KnowledgeBase(settings, fake_embeddings)
    kb.add_documents(
        [
            Document(
                page_content="RAG 包含索引、检索和生成。",
                metadata={
                    "source": "demo.pptx",
                    "file_hash": "h",
                    "file_type": "pptx",
                    "page": 1,
                    "page_label": "第 1 张幻灯片",
                    "location": "第 1 张幻灯片",
                    "section": "",
                },
            )
        ],
        "h",
    )
    search = make_search_tool(kb)
    result = search.invoke({"query": "RAG"})
    assert "demo.pptx" in result
    assert "第 1 张幻灯片" in result


def test_extract_answer_and_trace() -> None:
    result = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "1+1"},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="2", tool_call_id="1", name="calculator"),
            AIMessage(content="答案是 2"),
        ]
    }
    assert extract_answer(result) == "答案是 2"
    trace = extract_trace(result)
    assert any("calculator" in item for item in trace)
    assert any("2" in item for item in trace)
