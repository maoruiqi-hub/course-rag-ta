# Changelog

## v1.0.0 (2026-10-31)

### Added

- Streamlit 三标签页：文档总结、知识库问答、智能体
- PDF / PPTX / DOCX / TXT / Markdown 解析，保留页码/章节元数据
- Chroma 持久化、文件哈希去重、Top-K 检索
- stuff / map-reduce 两种摘要策略，结果缓存与 Markdown 下载
- 多轮问答、流式输出、引用来源卡片、资料不足拒答
- `create_agent` 智能体：知识库检索、计算器、当前时间
- 本地 FastEmbed 与 SiliconFlow 两种 embedding 方案
- 27 个离线单元测试、ruff、GitHub Actions
- RAG 检索评测脚本与演示资料生成脚本
