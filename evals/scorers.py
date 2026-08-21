"""評分函式。

原則：**能用程式判定的，絕不交給模型判定。**
規則式評分免費、瞬間完成、百分之百可重現；LLM-as-judge 貴、慢、而且自己也會出錯。
只有真正開放式的回答（例如「有沒有編造事實」）才值得動用 judge。
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Score:
    name: str
    passed: bool
    detail: str = ""


_FULLWIDTH = str.maketrans("０１２３４５６７８９，：％", "0123456789,:%")


def normalize(text: str) -> str:
    """把答案正規化後再比對，避免「答對卻被判錯」。

    這一步是實務上最容易被低估的：模型會把 3264 寫成 **3,264 元**、
    把 2026-08-21 寫成「2026 年 8 月 21 日」，內容完全正確卻過不了字串比對。
    比對兩邊都套用同一個正規化，才不會讓評分函式的脆弱被誤讀成模型退步。

    但**正規化要保守**：拿掉太多東西會製造假陽性，讓錯的答案也通過。
    這裡只處理「同一個值的不同寫法」，不碰語意。
    """
    t = text.translate(_FULLWIDTH)
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)  # 千分位：3,264 → 3264
    t = re.sub(  # 中文日期：2026 年 8 月 21 日 → 2026-08-21
        r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
        lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
        t,
    )
    return re.sub(r"[\s*_`]", "", t)  # 空白與 markdown 記號


def score_contains(answer: str, expected: List[str]) -> Optional[Score]:
    """結果評測：答案必須包含所有指定字串（正規化後比對）。"""
    if not expected:
        return None
    normalized = normalize(answer)
    missing = [s for s in expected if normalize(s) not in normalized]
    return Score("內容", not missing, f"缺少 {missing}" if missing else "")


def score_not_contains(answer: str, forbidden: List[str]) -> Optional[Score]:
    if not forbidden:
        return None
    normalized = normalize(answer)
    hit = [s for s in forbidden if normalize(s) in normalized]
    return Score("禁用詞", not hit, f"出現 {hit}" if hit else "")


def score_tools(used: List[str], expected: List[str], forbidden: List[str]) -> Optional[Score]:
    """軌跡評測：檢查過程走對了沒有。

    這是 Agent 評測與一般 LLM 評測最大的差別。
    「答案對但路徑錯」（例如沒查知識庫、憑記憶硬答）是最危險的假陽性 ——
    這次矇對了，換個問題就會錯，而且你完全不知道為什麼。
    """
    if not expected and not forbidden:
        return None
    missing = [t for t in expected if t not in used]
    used_forbidden = [t for t in forbidden if t in used]
    problems = []
    if missing:
        problems.append(f"未呼叫 {missing}")
    if used_forbidden:
        problems.append(f"不該呼叫 {used_forbidden}")
    return Score("軌跡", not problems, "；".join(problems))


def score_budget(llm_calls: int, max_llm_calls: int) -> Score:
    """成本護欄：答對了但多繞三圈，在生產環境等於慢三倍、貴三倍。"""
    ok = llm_calls <= max_llm_calls
    return Score("成本", ok, "" if ok else f"{llm_calls} 次 > 上限 {max_llm_calls}")


JUDGE_PROMPT = """你是一位嚴格的評審。請依據給定準則判斷回答是否合格。

【使用者問題】
{question}

【受測 Agent 的回答】
{answer}

【合格準則】
{criteria}

只輸出一行 JSON，不要有任何其他文字：
{{"pass": true 或 false, "reason": "10 字以內的理由"}}"""


def score_llm_judge(judge_llm, question: str, answer: str, criteria: str) -> Optional[Score]:
    """用另一個模型當評審，處理規則寫不出來的開放式判斷。

    注意：judge 自己也會錯，所以
      - 只給它「二元判斷 + 明確準則」，不要叫它打 1~10 分（分數極不穩定）
      - judge 的結果要人工抽驗，把它當成另一個需要被評測的元件
    """
    if not criteria:
        return None
    import json
    import re

    raw = judge_llm.invoke(
        JUDGE_PROMPT.format(question=question, answer=answer, criteria=criteria)
    ).content
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return Score("評審", False, f"評審輸出無法解析：{raw[:40]}")
    try:
        verdict = json.loads(match.group())
    except json.JSONDecodeError:
        return Score("評審", False, f"評審輸出無法解析：{raw[:40]}")
    return Score("評審", bool(verdict.get("pass")), str(verdict.get("reason", ""))[:30])
