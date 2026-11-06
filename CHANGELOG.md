# Changelog

## v1.1.0 (2026-11-05)

### Added

- PPTX 页眉页脚去噪 `strip_boilerplate`：剔除每页重复的版权声明与页码（真实课件 Hit@3 67% → 89%）
- 检索评测支持任意课件与问题集：`--file` / `--questions` / `--markdown`，新增 Hit@1/3、MRR 指标
- 第 7 章课件评测问题集 `data/eval/chapter7_questions.json`（18 题，人工标注）
- 问答页与智能体页的示例问题按钮
- 常见 API 错误（余额不足、key 无效、限流、超时）的中文提示 `friendly_error`
- README 演示动图与基于真实课件的截图

### Changed

- 默认 embedding 改为本地 FastEmbed，与 `.env.example` 和文档保持一致
- 对话输入框固定在消息下方；提问后自动收起示例问题
- 侧边栏对 PPTX 显示“N 张幻灯片”，摘要提示词按资料的位置写法标注来源
- 文本清洗统一处理不间断空格、零宽字符和全角空格
- 摘要限制在 800 字以内、聚焦知识内容；展示时把标题降为四级，避免比页面标题还大
- 来源卡片去掉重复的位置信息；工具调用轨迹自动换行
- 智能体标签页图标改为 🧠（🤖 在 Windows 上渲染成单色小图）

### Removed

- 演示视频脚本（改用 README 动图展示）

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
