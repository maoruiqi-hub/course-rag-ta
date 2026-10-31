"""可选的 Agent 页面：知识库检索、计算器、当前时间。"""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage

from course_ta.knowledge_base import KnowledgeBase

AGENT_SYSTEM = (
    "你是一名课程资料智能助教。你可以使用三个工具："
    "search_course_materials 检索课程资料，calculator 计算数学表达式，"
    "get_current_time 获取当前北京时间。\n"
    "回答课程问题时，优先调用检索工具，并在答案中标注资料页码；"
    "资料不足时明确说明，不要编造。回答保持简洁、中文。"
)

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ValueError("只支持数字")
    if isinstance(node, ast.BinOp):
        operation = _ALLOWED_BINOPS.get(type(node.op))
        if operation is None:
            raise ValueError("不支持的运算符")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("指数过大")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("除数不能为 0")
        return operation(left, right)
    if isinstance(node, ast.UnaryOp):
        operation = _ALLOWED_UNARYOPS.get(type(node.op))
        if operation is None:
            raise ValueError("不支持的一元运算符")
        return operation(_eval_node(node.operand))
    raise ValueError("表达式只允许数字和 + - * / // % ** 运算")


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式，例如 (128*3+7)/2。支持 + - * / // % ** 和括号。"""
    expression = expression.strip()
    if len(expression) > 200:
        return "表达式过长。"
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree)
        if abs(value) > 1e15:
            return "结果过大。"
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    except Exception as exc:
        return f"计算失败：{exc}"


@tool
def get_current_time() -> str:
    """获取当前北京时间（UTC+8），用于回答“现在几点 / 今天几号”等问题。"""
    now = datetime.now(timezone(timedelta(hours=8)))
    return now.strftime("%Y-%m-%d %H:%M:%S（北京时间）")


def make_search_tool(knowledge_base: KnowledgeBase):
    @tool
    def search_course_materials(query: str) -> str:
        """在已上传的课程资料中检索相关片段，返回文件名、页码和原文摘录。"""
        results = knowledge_base.search(query, k=4)
        if not results:
            return "知识库为空，或没有检索到相关内容。"
        blocks = []
        for index, (document, score) in enumerate(results, start=1):
            metadata = document.metadata or {}
            source = metadata.get("source", "未知文件")
            location = metadata.get("location") or metadata.get("page_label", "未知位置")
            content = document.page_content.strip()[:500]
            blocks.append(f"[{index}] 《{source}》 {location}（相关度 {score:.2f}）\n{content}")
        return "\n\n".join(blocks)

    return search_course_materials


def build_agent(llm, knowledge_base: KnowledgeBase):
    return create_agent(
        model=llm,
        tools=[make_search_tool(knowledge_base), calculator, get_current_time],
        system_prompt=AGENT_SYSTEM,
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def extract_answer(result: dict) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _content_to_text(message.content).strip()
            if text:
                return text
    return "Agent 没有返回内容。"


def extract_trace(result: dict) -> list[str]:
    """把工具调用与返回整理成可折叠的轨迹文本。"""
    trace: list[str] = []
    for message in result.get("messages", []):
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                name = call.get("name", "tool")
                args = call.get("args", {})
                trace.append(f"调用 {name}：{args}")
        elif isinstance(message, ToolMessage):
            content = _content_to_text(message.content).strip().replace("\n", " ")
            trace.append(f"{message.name or '工具'} 返回：{content[:400]}")
    return trace


def run_agent(agent, history: list[dict], question: str) -> dict:
    messages = [
        {"role": item.get("role", "user"), "content": item.get("content", "")}
        for item in history[-8:]
        if item.get("content")
    ]
    messages.append({"role": "user", "content": question})
    return agent.invoke({"messages": messages})
