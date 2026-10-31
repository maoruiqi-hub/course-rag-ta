"""生成用于演示和测试的小型课程资料（PPTX / DOCX / PDF）。

这些文件完全由脚本生成，内容为本项目自建的 LangChain 复习笔记，
不包含任何第三方课件，便于公开仓库和截图。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"

SLIDES = [
    (
        "课程资料智能助教",
        ["LangChain 1.x + Streamlit", "文档总结 · 知识库问答 · Agent"],
    ),
    (
        "RAG 的两个核心组件",
        ["索引：加载、拆分、向量化、存入 Chroma", "检索与生成：Retriever + ChatModel"],
    ),
    (
        "文档切分策略",
        ["RecursiveCharacterTextSplitter", "chunk_size=500，chunk_overlap=50"],
    ),
    (
        "文本总结",
        ["stuff：一次放入全部内容", "map-reduce：分批总结后再合并"],
    ),
    (
        "引用与页码",
        ["PDF 按页、PPTX 按幻灯片", "Word 引用章节和段落"],
    ),
    (
        "Agent 工具",
        ["知识库检索", "计算器", "当前时间"],
    ),
]


def make_pptx(path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Pt

    presentation = Presentation()
    for title, bullets in SLIDES:
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = title
        body = slide.placeholders[1].text_frame
        body.clear()
        for index, bullet in enumerate(bullets):
            paragraph = body.paragraphs[0] if index == 0 else body.add_paragraph()
            paragraph.text = bullet
            paragraph.font.size = Pt(22)
        slide.notes_slide.notes_text_frame.text = "本页由 make_demo_files.py 自动生成。"
    presentation.save(path)


def make_docx(path: Path) -> None:
    from docx import Document

    document = Document()
    document.add_heading("LangChain 课程笔记", level=0)
    document.add_paragraph("这是一份由脚本生成的中文示例文档，用于演示 Word 的章节与段落引用。")

    sections = [
        (
            "RAG 基础",
            [
                "RAG 通过检索外部资料来补充模型知识，并降低幻觉。",
                "索引阶段包括加载、拆分和向量化三个步骤。",
                "检索阶段使用 Retriever 找出与问题最相关的片段。",
            ],
        ),
        (
            "文本总结",
            [
                "短文档可以使用 stuff 策略，一次把全部内容放入提示词。",
                "长文档可以使用 map-reduce，先分批总结，再合并结果。",
            ],
        ),
        (
            "工程实践",
            [
                "API key 应放在 .env 中，不能提交到 Git。",
                "上传文件名需要清理，避免路径穿越。",
                "单元测试应使用假模型，避免依赖真实 API。",
            ],
        ),
    ]
    for heading, paragraphs in sections:
        document.add_heading(heading, level=1)
        for paragraph in paragraphs:
            document.add_paragraph(paragraph)

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "格式"
    table.cell(0, 1).text = "引用方式"
    table.cell(1, 0).text = "PPTX"
    table.cell(1, 1).text = "幻灯片页码"
    document.save(path)


def make_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    font_name = "Helvetica"
    for candidate in (r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc"):
        if Path(candidate).exists():
            try:
                pdfmetrics.registerFont(TTFont("DemoChinese", candidate))
                font_name = "DemoChinese"
                break
            except Exception:
                continue

    pages = [
        ["LangChain 课程笔记", "RAG 的两个核心组件：索引、检索与生成"],
        ["文档切分", "chunk_size=500，chunk_overlap=50，优先按中文标点切分"],
        ["引用与页码", "PDF 按页、PPTX 按幻灯片、Word 按章节与段落"],
        ["Agent", "模型 + 工具 + 循环：检索、计算、当前时间"],
    ]
    pdf = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    for index, lines in enumerate(pages, start=1):
        pdf.setFont(font_name, 20)
        pdf.drawString(60, height - 90, lines[0])
        pdf.setFont(font_name, 13)
        y = height - 140
        for line in lines[1:]:
            pdf.drawString(60, y, line)
            y -= 28
        pdf.setFont("Helvetica", 9)
        pdf.drawRightString(width - 50, 40, f"Page {index}")
        pdf.showPage()
    pdf.save()


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    make_pptx(SAMPLES / "demo_langchain.pptx")
    make_docx(SAMPLES / "demo_langchain.docx")
    make_pdf(SAMPLES / "demo_langchain.pdf")
    for path in sorted(SAMPLES.glob("demo_langchain.*")):
        print(f"generated: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
