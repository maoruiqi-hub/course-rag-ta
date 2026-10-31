"""文档加载与切分。

统一把 PDF / PPTX / DOCX / TXT / Markdown 转成带页码信息的
``langchain_core.documents.Document``，为后续引用打基础。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt", ".md"}

# 中文文档优先按段落、句号切，避免把一句话从中间截断。
TEXT_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]


class LoadError(RuntimeError):
    """文件无法解析时抛出，界面层负责展示中文提示。"""


def normalize_text(text: str) -> str:
    """清理控制字符与多余空白，保留段落结构。"""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_filename(name: str) -> str:
    """去掉路径和 Windows 非法字符，防止路径穿越。"""
    name = Path(name).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return (name or "upload")[:120]


def save_upload(name: str, data: bytes, upload_dir: Path) -> tuple[Path, str]:
    """保存上传文件，返回（本地路径，sha256）。

    文件名带内容哈希前缀，既方便去重，也避免同名覆盖。
    """
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise LoadError(f"不支持的文件类型：{suffix or '未知'}")

    digest = hash_bytes(data)
    folder = upload_dir / digest[:12]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / safe_filename(name)
    if not target.exists():
        target.write_bytes(data)
    return target, digest


def find_saved_file(upload_dir: Path, file_hash: str, source: str) -> Path | None:
    """根据哈希和原文件名找回上传文件，供“生成摘要”使用。"""
    expected = upload_dir / file_hash[:12] / safe_filename(source)
    if expected.exists():
        return expected
    matches = sorted(upload_dir.glob(f"{file_hash[:12]}/*"))
    if matches:
        return matches[0]
    # 兼容旧版本平铺保存的文件
    matches = sorted(upload_dir.glob(f"{file_hash[:12]}_*"))
    return matches[0] if matches else None


def _make_document(
    text: str,
    path: Path,
    file_hash: str,
    *,
    page: int,
    page_label: str,
    location: str,
    section: str = "",
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": path.name,
            "file_hash": file_hash,
            "file_type": path.suffix.lower().lstrip("."),
            "page": page,
            "page_label": page_label,
            "location": location,
            "section": section,
        },
    )


def _load_pdf(path: Path, file_hash: str) -> list[Document]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path), strict=False)
        pages = list(reader.pages)
    except Exception as exc:
        raise LoadError(f"PDF 解析失败：{exc}") from exc

    documents: list[Document] = []
    for index, page in enumerate(pages, start=1):
        try:
            text = normalize_text(page.extract_text() or "")
        except Exception:
            text = ""
        if len(text) < 5:
            continue
        documents.append(
            _make_document(
                text,
                path,
                file_hash,
                page=index,
                page_label=f"第 {index} 页",
                location=f"第 {index} 页",
            )
        )

    if not documents:
        raise LoadError("PDF 中没有提取到文本，可能是扫描件；本项目暂不支持 OCR。")
    average = sum(len(doc.page_content) for doc in documents) / len(documents)
    if average < 30:
        raise LoadError("PDF 文本量过少，可能是扫描件；本项目暂不支持 OCR。")
    return documents


def _shape_text(shape) -> list[str]:
    """递归提取 PPT 形状中的文字、表格和组合形状。"""
    parts: list[str] = []
    try:
        if getattr(shape, "has_text_frame", False):
            for paragraph in shape.text_frame.paragraphs:
                text = "".join(run.text for run in paragraph.runs).strip()
                if text:
                    parts.append(text)
    except Exception:
        pass

    try:
        if getattr(shape, "has_table", False):
            for row in shape.table.rows:
                cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
    except Exception:
        pass

    try:
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for child in shape.shapes:
                parts.extend(_shape_text(child))
    except Exception:
        pass
    return parts


def _load_pptx(path: Path, file_hash: str) -> list[Document]:
    try:
        from pptx import Presentation

        presentation = Presentation(str(path))
    except Exception as exc:
        raise LoadError(f"PPTX 解析失败：{exc}") from exc

    documents: list[Document] = []
    for index, slide in enumerate(presentation.slides, start=1):
        parts: list[str] = []
        for shape in slide.shapes:
            parts.extend(_shape_text(shape))
        try:
            if slide.has_notes_slide:
                notes = normalize_text(slide.notes_slide.notes_text_frame.text)
                if notes:
                    parts.append(f"备注：{notes}")
        except Exception:
            pass

        text = normalize_text("\n".join(parts))
        if len(text) < 5:
            continue
        documents.append(
            _make_document(
                text,
                path,
                file_hash,
                page=index,
                page_label=f"第 {index} 张幻灯片",
                location=f"第 {index} 张幻灯片",
            )
        )

    if not documents:
        raise LoadError("PPTX 中没有提取到文本，可能全部是图片。")
    return documents


def _load_docx(path: Path, file_hash: str) -> list[Document]:
    try:
        from docx import Document as DocxDocument

        doc = DocxDocument(str(path))
    except Exception as exc:
        raise LoadError(f"DOCX 解析失败：{exc}") from exc

    sections: list[dict] = []
    current = {"name": "正文", "paragraphs": [], "start": 1}
    paragraph_no = 0

    def flush() -> None:
        if current["paragraphs"]:
            current["end"] = paragraph_no
            sections.append(
                {
                    "name": current["name"],
                    "paragraphs": current["paragraphs"][:],
                    "start": current["start"],
                    "end": paragraph_no,
                }
            )
            current["paragraphs"] = []

    for paragraph in doc.paragraphs:
        text = normalize_text(paragraph.text)
        if not text:
            continue
        style = (paragraph.style.name or "").lower() if paragraph.style else ""
        if style.startswith(("heading", "标题")):
            flush()
            current = {"name": text, "paragraphs": [], "start": paragraph_no + 1}
        else:
            paragraph_no += 1
            current["paragraphs"].append(text)
    flush()

    # 把表格追加到最后一节；Word 没有稳定的“页”概念，引用到章节/段落。
    tables: list[str] = []
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
            if any(cells):
                tables.append(" | ".join(cells))
    if tables:
        if sections:
            sections[-1]["paragraphs"].extend(["表格："] + tables)
        else:
            sections.append(
                {
                    "name": "表格",
                    "paragraphs": ["表格："] + tables,
                    "start": 0,
                    "end": 0,
                }
            )

    documents: list[Document] = []
    group_size = 12
    for section in sections:
        paragraphs = section["paragraphs"]
        if not paragraphs:
            continue
        for offset in range(0, len(paragraphs), group_size):
            group = paragraphs[offset : offset + group_size]
            if section["start"] > 0:
                start = section["start"] + offset
                end = start + len(group) - 1
                location = f"章节：{section['name']} / 第 {start}–{end} 段"
            else:
                location = f"章节：{section['name']}"
            documents.append(
                _make_document(
                    "\n".join(group),
                    path,
                    file_hash,
                    page=-1,
                    page_label=f"章节：{section['name']}",
                    location=location,
                    section=section["name"],
                )
            )

    if not documents:
        raise LoadError("DOCX 中没有提取到文本。")
    return documents


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _load_text(path: Path, file_hash: str) -> list[Document]:
    text = normalize_text(_read_text(path))
    if not text:
        raise LoadError("文本文件为空。")

    lines = text.split("\n")
    block_size = 50
    documents: list[Document] = []
    for start in range(0, len(lines), block_size):
        block = lines[start : start + block_size]
        content = normalize_text("\n".join(block))
        if len(content) < 5:
            continue
        first = start + 1
        last = min(start + block_size, len(lines))
        documents.append(
            _make_document(
                content,
                path,
                file_hash,
                page=-1,
                page_label=f"第 {first}–{last} 行",
                location=f"第 {first}–{last} 行",
            )
        )
    if not documents:
        raise LoadError("文本文件没有有效内容。")
    return documents


def load_document(path: Path) -> list[Document]:
    """按扩展名加载单个文件，返回带统一 metadata 的文档列表。"""
    path = Path(path)
    if not path.exists():
        raise LoadError(f"文件不存在：{path.name}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise LoadError(f"不支持的文件类型：{suffix or '未知'}")

    file_hash = hash_file(path)
    loaders = {
        ".pdf": _load_pdf,
        ".pptx": _load_pptx,
        ".docx": _load_docx,
        ".txt": _load_text,
        ".md": _load_text,
    }
    documents = loaders[suffix](path, file_hash)
    for document in documents:
        document.metadata["file_hash"] = file_hash
    return documents


def split_documents(
    documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50
) -> list[Document]:
    """切分文档，保留页码等 metadata，并补充 chunk 序号。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=TEXT_SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    cleaned: list[Document] = []
    for index, chunk in enumerate(chunks):
        chunk.page_content = normalize_text(chunk.page_content)
        if len(chunk.page_content) < 10:
            continue
        chunk.metadata["chunk_index"] = index
        cleaned.append(chunk)
    return cleaned


def load_and_split(path: Path, chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    """加载并切分，供索引和摘要共用。"""
    return split_documents(load_document(path), chunk_size, chunk_overlap)
