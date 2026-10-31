from __future__ import annotations

from pathlib import Path

import pytest

from course_ta.documents import (
    LoadError,
    find_saved_file,
    hash_bytes,
    load_and_split,
    load_document,
    normalize_text,
    safe_filename,
    save_upload,
    split_documents,
)

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_normalize_text_removes_control_and_extra_space() -> None:
    assert normalize_text("  a\t b\r\n\r\n c \x00 ") == "a b\nc"


def test_save_upload_sanitizes_and_finds_file(tmp_path: Path) -> None:
    data = b"fake pptx bytes"
    path, digest = save_upload("../evil name.pptx", data, tmp_path)
    assert path.exists()
    assert ".." not in path.name
    assert path.parent.name == digest[:12]
    assert path.name == "evil name.pptx"
    assert digest == hash_bytes(data)
    assert find_saved_file(tmp_path, digest, "../evil name.pptx") == path
    assert safe_filename("../x").endswith("x")


def test_load_txt_utf8_and_gbk(tmp_path: Path) -> None:
    text = "第一段：RAG 是检索增强生成。\n第二段：它包含索引与生成。"
    utf8 = tmp_path / "notes.md"
    utf8.write_text(text, encoding="utf-8")
    docs = load_document(utf8)
    assert docs
    assert docs[0].metadata["page"] == -1
    assert "行" in docs[0].metadata["page_label"]
    assert "RAG" in docs[0].page_content

    gbk = tmp_path / "gbk.txt"
    gbk.write_bytes(text.encode("gbk"))
    docs = load_document(gbk)
    assert "检索增强生成" in docs[0].page_content


def test_load_pptx_has_slide_pages() -> None:
    docs = load_document(SAMPLES / "demo_langchain.pptx")
    assert len(docs) == 6
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["page_label"] == "第 1 张幻灯片"
    assert "RAG" in docs[1].page_content


def test_load_docx_has_section_and_paragraph_location() -> None:
    docs = load_document(SAMPLES / "demo_langchain.docx")
    assert docs
    sections = {doc.metadata["section"] for doc in docs}
    assert "RAG 基础" in sections
    rag_doc = next(doc for doc in docs if doc.metadata["section"] == "RAG 基础")
    assert rag_doc.metadata["page"] == -1
    assert rag_doc.metadata["page_label"] == "章节：RAG 基础"
    assert "段" in rag_doc.metadata["location"]


def test_load_pdf_has_page_numbers() -> None:
    docs = load_document(SAMPLES / "demo_langchain.pdf")
    assert len(docs) == 4
    assert [doc.metadata["page"] for doc in docs] == [1, 2, 3, 4]
    assert docs[0].metadata["page_label"] == "第 1 页"


def test_split_documents_preserves_metadata() -> None:
    docs = load_document(SAMPLES / "course_notes.md")
    chunks = split_documents(docs, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > len(docs)
    for chunk in chunks:
        assert chunk.metadata["source"] == "course_notes.md"
        assert "chunk_index" in chunk.metadata
        assert chunk.page_content.strip()


def test_load_and_split_limit_chunk_size() -> None:
    chunks = load_and_split(SAMPLES / "course_notes.md", chunk_size=200, chunk_overlap=20)
    assert chunks
    assert max(len(chunk.page_content) for chunk in chunks) <= 260


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "x.exe"
    path.write_bytes(b"x")
    with pytest.raises(LoadError):
        load_document(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LoadError):
        load_document(tmp_path / "missing.pdf")
