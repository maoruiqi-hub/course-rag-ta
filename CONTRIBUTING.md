# 贡献指南

感谢关注这个课程项目。它主要用于演示 LangChain RAG 应用，欢迎提交 issue 或 PR。

## 开发环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

## 提交前检查

```powershell
ruff check .
ruff format --check .
pytest -m "not integration"
```

## 代码约定

- UI 逻辑放在 `app.py`，可测试的业务逻辑放在 `course_ta/`；
- 新增文档格式时，必须提供 `source`、`file_hash`、`file_type`、`page`、`page_label`、`location` 元数据；
- 不提交 `.env`、`chroma_db/`、`data/uploads/`、`.cache/`；
- 不在日志或异常信息中打印 API key；
- 新增功能要补单元测试，测试不能依赖真实 API。

## 提交信息

使用 Conventional Commits：

```text
feat: add pptx page-aware loader
fix: handle empty retrieved context
docs: update setup instructions
test: cover summary routing
```
