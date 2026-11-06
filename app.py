"""课程资料智能助教 · Streamlit 入口。

页面只负责交互；解析、检索、摘要和 Agent 逻辑都在 ``course_ta`` 包中，
方便单元测试与复用。
"""

from __future__ import annotations

import time

import streamlit as st

from course_ta.agent import build_agent, extract_answer, extract_trace, run_agent
from course_ta.config import ConfigError, friendly_error, get_settings, validate_settings
from course_ta.documents import (
    find_saved_file,
    load_and_split,
    save_upload,
)
from course_ta.knowledge_base import KnowledgeBase
from course_ta.qa import (
    build_qa_chain,
    condense_question,
    dedupe_retrieval,
    demote_headings,
    format_context,
    history_to_messages,
    make_sources,
    recent_history,
    summarize_documents,
)

st.set_page_config(
    page_title="课程资料智能助教",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .block-container {max-width: 1180px; padding-top: 3.2rem; padding-bottom: 3rem;}
    header[data-testid="stHeader"] {background: transparent;}
    .hero {
        background: linear-gradient(135deg, #0f2742 0%, #173f5f 58%, #c2410c 160%);
        color: #fff7ed; border-radius: 22px; padding: 30px 34px 26px 34px;
        margin-bottom: 18px; box-shadow: 0 16px 40px rgba(15, 39, 66, .16);
    }
    .hero .eyebrow {font-size: .78rem; letter-spacing: .16em; text-transform: uppercase;
        color: #fdba74; font-weight: 700;}
    .hero h1 {font-size: 2.25rem; line-height: 1.15; margin: .3rem 0 .55rem 0; color: #fff7ed;}
    .hero p {font-size: 1.0rem; margin: 0; color: #e7dccb; max-width: 820px;}
    .pill {
        display: inline-block; padding: 5px 11px; margin: 10px 8px 0 0;
        border: 1px solid rgba(253, 186, 116, .55); border-radius: 999px;
        color: #ffedd5; font-size: .78rem; background: rgba(255, 255, 255, .06);
    }
    .section-note {color: #6b7280; font-size: .9rem; margin-top: -.4rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 8px 18px;}
    div[data-testid="stChatMessage"] {border-radius: 14px;}
    .footer {color: #9ca3af; font-size: .8rem; text-align: center; padding-top: 16px;}
    [data-testid="stAppDeployButton"], .stAppDeployButton {display: none !important;}
    .stButton button[kind="secondary"] {border-radius: 999px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_llm():
    from course_ta.config import build_llm

    return build_llm(get_settings())


@st.cache_resource(show_spinner=False)
def get_kb() -> KnowledgeBase:
    from course_ta.config import build_embeddings

    settings = validate_settings()
    return KnowledgeBase(settings, build_embeddings(settings))


QA_EXAMPLES = [
    "LangChain 有哪些核心模块？",
    "典型的 RAG 应用程序包含哪两个主要组件？",
    "ReAct 框架的设计理念是什么？",
]
AGENT_EXAMPLES = [
    "课件里 AutoGen 支持哪些多智能体对话模式？",
    "请计算 (128*3+7)/2，并告诉我现在的北京时间",
]


def ask_box(messages: list[dict], examples: list[str], key: str, placeholder: str) -> str | None:
    """输入框 + 示例问题。对话为空时展示示例，点击即提问；一旦提问就收起示例。"""
    slot = st.empty()
    clicked = None
    if not messages:
        with slot.container():
            st.caption("试试这些问题：")
            columns = st.columns(len(examples))
            for index, (column, text) in enumerate(zip(columns, examples, strict=True)):
                if column.button(text, key=f"{key}_example_{index}", use_container_width=True):
                    clicked = text
    question = st.chat_input(placeholder, key=f"{key}_input") or clicked
    if question:
        slot.empty()
    return question


def init_state() -> None:
    defaults = {
        "qa_messages": [],
        "agent_messages": [],
        "summary_cache": {},
        "agent": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">开源软件构建技术 · LangChain 课程作业</div>
            <h1>课程资料智能助教</h1>
            <p>上传课件或讲义，生成复习摘要并就资料提问；回答标注文件名与页码，可回原文核对。</p>
            <span class="pill">LangChain 1.x · LCEL</span>
            <span class="pill">DeepSeek API</span>
            <span class="pill">BGE / Qwen Embedding</span>
            <span class="pill">Chroma</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def setup_page(error: Exception) -> None:
    hero()
    st.error(str(error))
    st.markdown("### 三步完成配置")
    st.code(
        "Copy-Item .env.example .env\n"
        "# 编辑 .env，填入 LLM_API_KEY；Embedding 默认本地，无需 key\n"
        "streamlit run app.py",
        language="powershell",
    )
    st.markdown(
        "对话模型默认使用 DeepSeek（`deepseek-flash`）。向量化默认使用本地 "
        "FastEmbed `BAAI/bge-small-zh-v1.5`，无需 embedding key；"
        "如果 SiliconFlow 账户有余额，也可以切换到云端 `Qwen/Qwen3-Embedding-0.6B` 或 `BAAI/bge-m3`。"
    )
    st.stop()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 引用来源（{len(sources)}）", expanded=False):
        for source in sources:
            with st.container(border=True):
                st.markdown(
                    f"**[{source['index']}] {source['source']}**  \n"
                    f"`{source['location']}` · 相关度 {source['score']:.2f}"
                )
                st.text(source["snippet"])


def index_uploads(uploaded_files, knowledge_base: KnowledgeBase, settings) -> None:
    if not uploaded_files:
        st.warning("请先选择要上传的文件。")
        return

    progress = st.progress(0.0, text="准备索引…")
    total = len(uploaded_files)
    success = 0
    skipped = 0
    for index, uploaded in enumerate(uploaded_files, start=1):
        try:
            path, digest = save_upload(uploaded.name, uploaded.getvalue(), settings.upload_dir)
            if knowledge_base.has_file(digest):
                skipped += 1
                st.toast(f"《{uploaded.name}》已存在，跳过")
                continue
            with st.spinner(f"正在解析《{uploaded.name}》…"):
                chunks = load_and_split(path)
            with st.spinner(f"正在向量化《{uploaded.name}》（{len(chunks)} 个片段）…"):
                added = knowledge_base.add_documents(chunks, digest)
            success += 1
            st.toast(f"《{uploaded.name}》已入库：{added} 个片段")
        except Exception as exc:
            st.error(f"《{uploaded.name}》处理失败：{friendly_error(exc)}")
        progress.progress(index / total, text=f"已处理 {index}/{total}")
    progress.empty()
    if success or skipped:
        st.success(f"索引完成：新增 {success} 个文件，跳过 {skipped} 个已存在文件。")


def sidebar(settings) -> tuple[KnowledgeBase, int]:
    knowledge_base = get_kb()
    with st.sidebar:
        st.markdown("## 📥 资料库")
        uploaded_files = st.file_uploader(
            "上传课程资料",
            type=["pdf", "pptx", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="支持 PDF、PPTX、DOCX、TXT、Markdown；单个文件建议不超过 100MB。",
        )
        if st.button("开始索引", type="primary", use_container_width=True):
            index_uploads(uploaded_files, knowledge_base, settings)

        stats = knowledge_base.stats()
        st.caption(f"当前知识库：{stats['files']} 个文件 · {stats['chunks']} 个片段")

        files = knowledge_base.list_files()
        if files:
            for item in files:
                unit = "张幻灯片" if item["file_type"] == "pptx" else "页"
                page_text = f"{item['page_count']} {unit}" if item["page_count"] else "无固定页码"
                st.markdown(f"**{item['source']}**")
                st.caption(f"{item['file_type'].upper()} · {page_text} · {item['chunks']} 片段")
        else:
            st.info("知识库为空，先上传课件并点击“开始索引”。")

        st.divider()
        top_k = st.slider("检索片段数 Top-K", 3, 8, value=min(max(settings.top_k, 3), 8))
        st.caption("数值越大，上下文越丰富，但 token 消耗也越高。")

        with st.expander("⚙️ 模型与配置"):
            st.caption(f"对话模型：`{settings.llm_model}`")
            embedding_name = (
                settings.fastembed_model
                if settings.embedding_provider == "fastembed"
                else settings.embedding_model
            )
            st.caption(f"Embedding：`{embedding_name}`（{settings.embedding_provider}）")
            st.caption(f"LLM Key：`{settings.masked_llm_key()}`")
            if settings.embedding_provider != "fastembed":
                st.caption(f"Embedding Key：`{settings.masked_embedding_key()}`")
            else:
                st.caption("Embedding Key：本地模型，无需 key")
            st.caption(f"Chroma：`{settings.chroma_dir}`")

        if st.button("刷新连接", use_container_width=True):
            st.cache_resource.clear()
            get_settings.cache_clear()
            st.session_state.agent = None
            st.rerun()

        st.divider()
        confirm_clear = st.checkbox("我确认要清空知识库", key="confirm_clear")
        if st.button("清空知识库", disabled=not confirm_clear, use_container_width=True):
            knowledge_base.clear()
            st.session_state.summary_cache = {}
            st.session_state.agent = None
            st.success("知识库已清空。")
            st.rerun()

    return knowledge_base, top_k


def summary_tab(settings, knowledge_base: KnowledgeBase) -> None:
    st.subheader("📄 文档总结")
    st.markdown(
        '<p class="section-note">短文使用 stuff；长文自动切换 map-reduce，'
        "也可以手动强制长文档模式来演示两种策略。</p>",
        unsafe_allow_html=True,
    )
    files = knowledge_base.list_files()
    if not files:
        st.info("还没有可总结的资料。请先在左侧上传并索引。")
        return

    labels = {}
    for item in files:
        label = f"{item['source']}（{item['chunks']} 片段）"
        if label in labels:
            label = f"{label} · {item['file_hash'][:6]}"
        labels[label] = item

    selected_label = st.selectbox("选择要总结的资料", list(labels.keys()))
    selected = labels[selected_label]
    force_map = st.toggle(
        "强制使用 map-reduce（适合演示长文档总结）",
        value=False,
        help="关闭时按字符数自动选择；打开后无论长短都会分批总结。",
    )

    cache_key = f"{selected['file_hash']}:{'map' if force_map else 'auto'}"
    cached = st.session_state.summary_cache.get(cache_key)

    if st.button("生成摘要", type="primary"):
        path = find_saved_file(settings.upload_dir, selected["file_hash"], str(selected["source"]))
        if path is None:
            st.error("找不到原始上传文件，可能已被移动。请重新上传该资料。")
        else:
            try:
                with st.spinner("正在解析与切分资料…"):
                    chunks = load_and_split(path)
                started = time.perf_counter()
                progress = st.progress(0.0, text="正在生成摘要…")

                def on_progress(done: int, total: int) -> None:
                    progress.progress(
                        done / max(total, 1), text=f"正在总结第 {done}/{total} 批内容…"
                    )

                result = summarize_documents(
                    chunks,
                    get_llm(),
                    force_map=force_map,
                    threshold=settings.summary_threshold,
                    progress=on_progress,
                )
                progress.empty()
                elapsed = time.perf_counter() - started
                st.session_state.summary_cache[cache_key] = (result, elapsed)
                cached = (result, elapsed)
            except Exception as exc:
                st.error(f"摘要生成失败：{friendly_error(exc)}")

    if cached:
        result, elapsed = cached
        mode_text = "map-reduce（分批总结）" if result.mode == "map-reduce" else "stuff（一次总结）"
        st.caption(
            f"模式：{mode_text} · {result.chunks} 个片段 · "
            f"约 {result.characters:,} 字符 · 耗时 {elapsed:.1f}s"
        )
        with st.container(border=True):
            st.markdown(demote_headings(result.text))
        st.download_button(
            "下载 Markdown 摘要",
            data=result.text.encode("utf-8"),
            file_name=f"{selected['source'].rsplit('.', 1)[0]}_summary.md",
            mime="text/markdown",
        )


def qa_tab(settings, knowledge_base: KnowledgeBase, top_k: int) -> None:
    left, right = st.columns([5, 1])
    with left:
        st.subheader("💬 知识库问答")
        st.markdown(
            '<p class="section-note">回答只依据已索引资料；展开“引用来源”可以核对页码和原文。</p>',
            unsafe_allow_html=True,
        )
    with right:
        if st.button("清空对话", key="clear_qa", use_container_width=True):
            st.session_state.qa_messages = []
            st.rerun()

    if knowledge_base.count() == 0:
        st.info("知识库为空。请先在左侧上传课件并建立索引。")
        return

    # 先占位消息区，再放输入框，保证输入框始终在对话下方
    chat_box = st.container()
    question = ask_box(
        st.session_state.qa_messages, QA_EXAMPLES, "qa", "例如：RAG 的两个核心组件是什么？"
    )

    with chat_box:
        for message in st.session_state.qa_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    render_sources(message["sources"])

        if question:
            history = list(st.session_state.qa_messages)
            st.session_state.qa_messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("正在检索资料…"):
                    try:
                        search_query = condense_question(get_llm(), history, question)
                        pairs = dedupe_retrieval(knowledge_base.search(search_query, k=top_k))
                    except Exception as exc:
                        st.error(f"检索失败：{friendly_error(exc)}")
                        pairs = []
                        search_query = question

                if not pairs:
                    answer = "没有在已索引资料中检索到相关内容。可以换一种问法，或补充上传资料。"
                    st.markdown(answer)
                    sources: list[dict] = []
                else:
                    documents = [document for document, _ in pairs]
                    sources = [source.to_dict() for source in make_sources(pairs)]
                    try:
                        chain = build_qa_chain(get_llm())
                        answer = st.write_stream(
                            chain.stream(
                                {
                                    "history": history_to_messages(recent_history(history)),
                                    "context": format_context(documents),
                                    "question": question,
                                }
                            )
                        )
                    except Exception as exc:
                        st.error(f"回答生成失败：{friendly_error(exc)}")
                        answer = ""
                    if answer:
                        render_sources(sources)

                if answer:
                    st.session_state.qa_messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )


def agent_tab(settings, knowledge_base: KnowledgeBase) -> None:
    left, right = st.columns([5, 1])
    with left:
        st.subheader("🧠 智能体")
        st.markdown(
            '<p class="section-note">用 create_agent 挂载三个工具：课程资料检索、计算器、当前时间。'
            "工具调用轨迹会显示在回答下方。</p>",
            unsafe_allow_html=True,
        )
    with right:
        if st.button("清空对话", key="clear_agent", use_container_width=True):
            st.session_state.agent_messages = []
            st.rerun()

    if not settings.enable_agent:
        st.info("Agent 功能已关闭。设置 ENABLE_AGENT=true 后重启即可开启。")
        return

    chat_box = st.container()
    question = ask_box(
        st.session_state.agent_messages,
        AGENT_EXAMPLES,
        "agent",
        "试试：RAG 的流程是什么？或者计算 (128*3+7)/2",
    )

    with chat_box:
        for message in st.session_state.agent_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("trace"):
                    with st.expander("🧰 工具调用轨迹", expanded=False):
                        st.code("\n".join(message["trace"]), language="text", wrap_lines=True)
        if question:
            run_agent_turn(knowledge_base, question)


def run_agent_turn(knowledge_base: KnowledgeBase, question: str) -> None:
    """执行一轮 Agent 对话，并把结果写回 session。"""
    history = list(st.session_state.agent_messages)
    st.session_state.agent_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考并调用工具…"):
            try:
                agent = st.session_state.agent
                if agent is None:
                    agent = build_agent(get_llm(), knowledge_base)
                    st.session_state.agent = agent
                result = run_agent(agent, history, question)
                answer = extract_answer(result)
                trace = extract_trace(result)
            except Exception as exc:
                answer = f"Agent 暂时不可用：{friendly_error(exc)}\n\n可以先使用“知识库问答”页面。"
                trace = []
        st.markdown(answer)
        if trace:
            with st.expander("🧰 工具调用轨迹", expanded=True):
                st.code("\n".join(trace), language="text", wrap_lines=True)

    st.session_state.agent_messages.append({"role": "assistant", "content": answer, "trace": trace})


def main() -> None:
    hero()
    try:
        settings = validate_settings()
    except ConfigError as exc:
        setup_page(exc)
        return

    init_state()
    knowledge_base, top_k = sidebar(settings)

    tab_summary, tab_qa, tab_agent = st.tabs(["📄 文档总结", "💬 知识库问答", "🧠 智能体"])
    with tab_summary:
        summary_tab(settings, knowledge_base)
    with tab_qa:
        qa_tab(settings, knowledge_base, top_k)
    with tab_agent:
        agent_tab(settings, knowledge_base)

    st.divider()
    st.markdown(
        '<div class="footer">课程资料智能助教 · LangChain 1.x · '
        "DeepSeek · FastEmbed / SiliconFlow · Chroma</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
