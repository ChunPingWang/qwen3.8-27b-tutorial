"""一個跑在本機 Ollama + Qwen3.8 27B 上的 LangChain / LangGraph ReAct Agent。

重點：
  1. ChatOllama 直接對接本機 http://localhost:11434，不需要任何 API key。
  2. Agent 具備三個工具，模型會自行決定何時呼叫。
  3. 全程透過 TokenUsageCallbackHandler 記錄每一步的 token 與延遲（可觀測性）。

使用：
    python agent.py                       # 跑內建示範問題
    python agent.py "台北現在幾點？"        # 問自己的問題
    python agent.py --think "..."          # 開啟 Qwen 的 thinking 模式
"""

from __future__ import annotations

import argparse
import ast
import operator
import os
from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from token_usage import TokenUsageCallbackHandler

MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.8:27b")
BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

SYSTEM_PROMPT = (
    "你是一個嚴謹的助理，使用繁體中文回答。"
    "需要計算、查目前時間或查詢產品知識庫時，一律呼叫工具，不要憑記憶回答。"
    "拿到工具結果後，用一到三句話直接給出結論。"
)

# 一個假的本地知識庫，用來示範「工具讀取外部資料」這件事。
KNOWLEDGE_BASE = {
    "退貨": "商品到貨 7 天內可申請退貨，需保持包裝完整，運費由賣方負擔。",
    "保固": "所有硬體享 2 年原廠保固，人為損壞不在保固範圍內。",
    "運費": "單筆訂單滿 1000 元免運，未滿收取 80 元運費。",
    "付款": "支援信用卡、ATM 轉帳與貨到付款，貨到付款加收 30 元手續費。",
}

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    """只允許數字與四則運算的 AST 求值，避免 eval() 的風險。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("只支援數字與 + - * / // % ** 運算")


@tool
def calculate(expression: str) -> str:
    """計算一個數學算式，例如 "1280 * 3 * 0.85"。只支援 + - * / // % ** 與括號。"""
    try:
        return str(_eval_node(ast.parse(expression, mode="eval")))
    except Exception as exc:  # 把錯誤回給模型，讓它有機會自行修正算式
        return f"計算失敗：{exc}"


@tool
def get_current_time(utc_offset_hours: int = 8) -> str:
    """取得目前時間。utc_offset_hours 為時區偏移，台北為 8。"""
    now = datetime.now(timezone(timedelta(hours=utc_offset_hours)))
    return now.strftime("%Y-%m-%d %H:%M:%S (UTC%z)")


@tool
def search_policy(keyword: str) -> str:
    """查詢客服政策知識庫（退貨、保固、運費、付款）。傳入一個關鍵字。"""
    hits = [v for k, v in KNOWLEDGE_BASE.items() if k in keyword or keyword in k]
    return "\n".join(hits) if hits else f"知識庫中查無「{keyword}」，可用主題：{'、'.join(KNOWLEDGE_BASE)}"


TOOLS = [calculate, get_current_time, search_policy]


def build_agent(think: bool = False, num_ctx: int = 8192):
    """建立 ReAct Agent。think=False 會關閉 Qwen 的思考模式，速度快很多。"""
    llm = ChatOllama(
        model=MODEL,
        base_url=BASE_URL,
        temperature=0,
        num_ctx=num_ctx,       # 每一步的上下文視窗，直接影響 prompt token 上限
        reasoning=think,       # Qwen3.8 的 thinking 開關
    )
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.8 27B 本地 Agent 示範")
    parser.add_argument("question", nargs="?", help="要問 Agent 的問題")
    parser.add_argument("--think", action="store_true", help="開啟 thinking 模式")
    parser.add_argument("--num-ctx", type=int, default=8192, help="上下文視窗大小")
    parser.add_argument("--jsonl", default="runs/usage.jsonl", help="事件輸出檔（設為 '' 關閉）")
    args = parser.parse_args()

    question = args.question or (
        "我買了 3 台單價 1280 元的設備並使用 85 折，總金額多少？"
        "這個金額有免運嗎？另外現在台北時間幾點？"
    )

    agent = build_agent(think=args.think, num_ctx=args.num_ctx)
    tracker = TokenUsageCallbackHandler(jsonl_path=args.jsonl or None)

    print(f"模型：{MODEL}　thinking：{'on' if args.think else 'off'}　num_ctx：{args.num_ctx}")
    print(f"問題：{question}\n")
    print("--- Agent 執行軌跡 ---")

    result = agent.invoke(
        {"messages": [("user", question)]},
        config={"callbacks": [tracker], "recursion_limit": 12},
    )

    print("\n--- 最終回答 ---")
    print(result["messages"][-1].content)
    print(tracker.report())


if __name__ == "__main__":
    main()
