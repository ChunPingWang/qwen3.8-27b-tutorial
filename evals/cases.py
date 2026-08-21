"""評測資料集。

Eval 的品質等於資料集的品質。挑案例的原則：
  1. 涵蓋**每一個工具**，以及**多工具組合**的情境
  2. 一定要有「不該用工具」的案例 —— Agent 最常見的毛病是濫用工具
  3. 一定要有「知識庫查不到」的案例 —— 用來檢驗模型會不會編造答案
  4. 每個案例都要有明確、可自動判定的期望，而不是靠人看
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List

# 動態期望值：時間類的答案不可能寫死，改成執行當下計算。
TODAY_TAIPEI = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")


@dataclass
class EvalCase:
    id: str
    question: str
    # --- 結果評測：最終答案該長什麼樣 ---
    expect_contains: List[str] = field(default_factory=list)
    expect_not_contains: List[str] = field(default_factory=list)
    # --- 軌跡評測：過程該怎麼走 ---
    expect_tools: List[str] = field(default_factory=list)
    forbid_tools: List[str] = field(default_factory=list)
    # --- 成本護欄 ---
    max_llm_calls: int = 4
    # --- 開放式回答才需要，交給 LLM-as-judge ---
    judge: str = ""


CASES: List[EvalCase] = [
    EvalCase(
        id="policy-warranty",
        question="保固多久？",
        expect_contains=["2 年"],
        expect_tools=["search_policy"],
        max_llm_calls=3,
    ),
    EvalCase(
        id="calc-discount",
        question="我買了 3 台單價 1280 元的設備，打 85 折，總金額多少？",
        expect_contains=["3264"],
        expect_tools=["calculate"],
        forbid_tools=["search_policy"],
        max_llm_calls=3,
    ),
    EvalCase(
        id="multi-step",
        question="3 台單價 1280 元的設備打 85 折後總共多少？這個金額有免運嗎？",
        expect_contains=["3264", "免運"],
        # 必須「先算出金額」再「查運費規則」，只答對一半不算過。
        expect_tools=["calculate", "search_policy"],
        max_llm_calls=5,
    ),
    EvalCase(
        id="current-time",
        question="現在台北時間是幾號？",
        expect_contains=[TODAY_TAIPEI],
        expect_tools=["get_current_time"],
        max_llm_calls=3,
    ),
    EvalCase(
        id="no-tool-needed",
        question="你好，請用一句話介紹你自己。",
        # Agent 最常見的浪費：明明不用工具卻硬要呼叫一輪。
        forbid_tools=["calculate", "get_current_time", "search_policy"],
        max_llm_calls=1,
    ),
    EvalCase(
        id="not-in-kb",
        question="你們有提供到府安裝服務嗎？",
        expect_tools=["search_policy"],
        # 知識庫沒有這項，正確行為是承認查不到，而不是掰一個答案。
        judge="回答必須表明知識庫中查無「到府安裝」的相關規定，"
        "或明確表示無法確認；若編造出任何具體的到府安裝政策則不合格。",
    ),
]
