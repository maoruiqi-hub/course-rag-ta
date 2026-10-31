"""Chroma 知识库：写入、去重、检索、清空。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from course_ta.config import Settings, build_embeddings

COLLECTION_NAME = "course_materials"


class KnowledgeBase:
    """对 Chroma 的薄封装，界面层不直接操作向量库。"""

    def __init__(self, settings: Settings, embeddings: Embeddings | None = None):
        self.settings = settings
        self.embeddings = embeddings or build_embeddings(settings)
        self.store = self._new_store()

    def _new_store(self) -> Chroma:
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(self.settings.chroma_dir),
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ 写入
    def count(self) -> int:
        try:
            return int(self.store._collection.count())
        except Exception:
            return len(self.store.get().get("ids", []))

    def has_file(self, file_hash: str) -> bool:
        if not file_hash:
            return False
        result = self.store.get(where={"file_hash": file_hash}, limit=1)
        return bool(result.get("ids"))

    def add_documents(self, documents: list[Document], file_hash: str) -> int:
        """写入一批已切分的文档；相同哈希默认跳过。"""
        documents = [doc for doc in documents if doc.page_content.strip()]
        if not documents:
            return 0
        if self.has_file(file_hash):
            return 0
        ids = [f"{file_hash[:16]}-{index:05d}" for index in range(len(documents))]
        self.store.add_documents(documents=documents, ids=ids)
        return len(documents)

    def delete_file(self, file_hash: str) -> int:
        result = self.store.get(where={"file_hash": file_hash})
        ids = result.get("ids", [])
        if ids:
            self.store.delete(ids=ids)
        return len(ids)

    def clear(self) -> None:
        """删除整个 collection，再创建一个空 collection。"""
        try:
            self.store.delete_collection()
        finally:
            self.store = self._new_store()

    # ------------------------------------------------------------------ 检索
    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """返回（文档，相关度 0~1），相关度越高越匹配。"""
        if not query.strip():
            return []
        k = max(1, min(k or self.settings.top_k, 20))
        try:
            pairs = self.store.similarity_search_with_score(query, k=k)
        except Exception:
            return []
        results: list[tuple[Document, float]] = []
        for document, distance in pairs:
            try:
                score = max(0.0, min(1.0, 1.0 - float(distance)))
            except (TypeError, ValueError):
                score = 0.0
            results.append((document, score))
        return results

    def as_retriever(self, k: int | None = None):
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": max(1, min(k or self.settings.top_k, 20))},
        )

    # ------------------------------------------------------------------ 元数据
    def list_files(self) -> list[dict[str, Any]]:
        """按 file_hash 聚合出资料列表，供侧边栏和摘要下拉框使用。"""
        try:
            result = self.store.get(include=["metadatas"])
        except Exception:
            return []

        grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"chunks": 0, "pages": set()})
        for metadata in result.get("metadatas", []):
            if not metadata:
                continue
            file_hash = str(metadata.get("file_hash", ""))
            if not file_hash:
                continue
            item = grouped[file_hash]
            item["file_hash"] = file_hash
            item["source"] = metadata.get("source", "未知文件")
            item["file_type"] = metadata.get("file_type", "")
            item["chunks"] += 1
            page = metadata.get("page", -1)
            if isinstance(page, int) and page > 0:
                item["pages"].add(page)

        files: list[dict[str, Any]] = []
        for item in grouped.values():
            pages = item.pop("pages")
            item["page_count"] = len(pages)
            files.append(item)
        return sorted(files, key=lambda item: str(item.get("source", "")))

    def stats(self) -> dict[str, int]:
        files = self.list_files()
        return {
            "files": len(files),
            "chunks": sum(int(item["chunks"]) for item in files),
        }
