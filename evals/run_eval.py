"""評測執行器：跑資料集、算分數、比對基準線。

單元測試問的是「壞了沒有？」（是非題）；
Eval 問的是「現在多好？比上次好還是壞？」（比較題）。

因為模型不確定，同一題跑兩次可能一次過一次不過，
所以每題可以重複跑 N 次取通過率（--repeat），並且把成本一起量進來 ——
一個「準確率 100% 但 token 翻倍」的改動，未必是進步。

用法：
    python -m evals.run_eval                      # 跑全部案例
    python -m evals.run_eval --only calc,multi    # 只跑 id 含這些字的案例
    python -m evals.run_eval --repeat 3           # 每題跑 3 次，量穩定度
    python -m evals.run_eval --judge              # 啟用 LLM-as-judge 案例
    python -m evals.run_eval --save evals/baseline.json
    python -m evals.run_eval --compare evals/baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import MODEL, build_agent  # noqa: E402
from evals.cases import CASES, EvalCase  # noqa: E402
from evals.scorers import (  # noqa: E402
    Score,
    score_budget,
    score_contains,
    score_llm_judge,
    score_not_contains,
    score_tools,
)
from token_usage import TokenUsageCallbackHandler  # noqa: E402


def run_case(agent, case: EvalCase, judge_llm=None) -> Dict[str, Any]:
    """跑一個案例，回傳「答案 + 軌跡 + 成本 + 各項分數」。"""
    tracker = TokenUsageCallbackHandler(verbose=False)
    try:
        result = agent.invoke(
            {"messages": [("user", case.question)]},
            config={"callbacks": [tracker], "recursion_limit": 12},
        )
        answer = result["messages"][-1].content
        error = None
    except Exception as exc:  # 例外本身就是一種不合格，不該讓整份評測中斷
        answer, error = "", f"{type(exc).__name__}: {exc}"

    used_tools = [t.name for t in tracker.tool_calls]
    scores: List[Score] = [
        s
        for s in [
            score_contains(answer, case.expect_contains),
            score_not_contains(answer, case.expect_not_contains),
            score_tools(used_tools, case.expect_tools, case.forbid_tools),
            score_budget(tracker.summary()["llm_calls"], case.max_llm_calls),
            score_llm_judge(judge_llm, case.question, answer, case.judge) if judge_llm else None,
        ]
        if s is not None
    ]
    if error:
        scores.append(Score("執行", False, error))

    usage = tracker.summary()
    return {
        "passed": all(s.passed for s in scores),
        "answer": answer,
        "tools": used_tools,
        "scores": [{"name": s.name, "passed": s.passed, "detail": s.detail} for s in scores],
        "total_tokens": usage["total_tokens"],
        "seconds": usage["llm_seconds"],
        "llm_calls": usage["llm_calls"],
    }


def evaluate(cases: List[EvalCase], repeat: int, think: bool, use_judge: bool) -> Dict[str, Any]:
    agent = build_agent(think=think)
    judge_llm = None
    if use_judge:
        # 評審刻意用不開 thinking 的同一顆模型；正式環境建議換一顆更強的模型當評審。
        from langchain_ollama import ChatOllama

        judge_llm = ChatOllama(model=MODEL, temperature=0, reasoning=False)

    results = {}
    for case in cases:
        runs = []
        for i in range(repeat):
            print(f"  跑 {case.id} ({i + 1}/{repeat}) ...", flush=True)
            runs.append(run_case(agent, case, judge_llm))
        results[case.id] = {
            "pass_rate": sum(r["passed"] for r in runs) / len(runs),
            "avg_tokens": round(statistics.mean(r["total_tokens"] for r in runs)),
            "avg_seconds": round(statistics.mean(r["seconds"] for r in runs), 1),
            "avg_llm_calls": round(statistics.mean(r["llm_calls"] for r in runs), 1),
            "runs": runs,
        }
    return results


def print_report(results: Dict[str, Any], model: str, repeat: int) -> None:
    print("\n" + "=" * 78)
    print(f"評測報告　模型：{model}　每題執行次數：{repeat}")
    print("=" * 78)
    print(f"{'case':<18}{'pass':>7}{'tokens':>9}{'sec':>7}{'calls':>7}  失敗原因")
    print("-" * 78)
    for case_id, r in results.items():
        reasons = sorted(
            {
                f"{s['name']}:{s['detail'] or 'X'}"
                for run in r["runs"]
                for s in run["scores"]
                if not s["passed"]
            }
        )
        print(
            f"{case_id:<18}{r['pass_rate']:>6.0%}{r['avg_tokens']:>9}"
            f"{r['avg_seconds']:>7.1f}{r['avg_llm_calls']:>7.1f}  {'; '.join(reasons)[:30]}"
        )
    print("-" * 78)
    overall = statistics.mean(r["pass_rate"] for r in results.values())
    total_tokens = sum(r["avg_tokens"] for r in results.values())
    print(
        f"{'總計':<16}{overall:>6.0%}{total_tokens:>9}"
        f"{sum(r['avg_seconds'] for r in results.values()):>7.1f}"
    )
    print("=" * 78)


def compare(results: Dict[str, Any], baseline_path: str) -> None:
    """回歸比對：這才是 eval 真正的用途 —— 改動之後有沒有變差。"""
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    print(f"\n與基準線比對：{baseline_path}（基準模型：{baseline.get('model', '?')}）")
    print("-" * 78)
    for case_id, r in results.items():
        base = baseline["results"].get(case_id)
        if base is None:
            print(f"{case_id:<18}新增案例，基準線中沒有")
            continue
        d_pass = r["pass_rate"] - base["pass_rate"]
        d_tok = r["avg_tokens"] - base["avg_tokens"]
        flag = "✗ 退步" if d_pass < 0 else ("✓ 進步" if d_pass > 0 else ("  持平" if d_tok <= 0 else "! 變貴"))
        print(
            f"{case_id:<18}{flag}　通過率 {base['pass_rate']:.0%} → {r['pass_rate']:.0%}"
            f"　token {base['avg_tokens']} → {r['avg_tokens']} ({d_tok:+d})"
        )
    print("-" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.8 27B Agent 評測")
    parser.add_argument("--repeat", type=int, default=1, help="每題重複執行次數")
    parser.add_argument("--only", default="", help="只跑 id 包含這些關鍵字的案例（逗號分隔）")
    parser.add_argument("--think", action="store_true", help="開啟 thinking 模式評測")
    parser.add_argument("--judge", action="store_true", help="啟用 LLM-as-judge 評分")
    parser.add_argument("--save", default="", help="把結果存成基準線 JSON")
    parser.add_argument("--compare", default="", help="與指定基準線比對")
    args = parser.parse_args()

    cases = CASES
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        cases = [c for c in CASES if any(k in c.id for k in keys)]
    if not cases:
        sys.exit("沒有符合條件的案例")

    print(f"模型：{MODEL}　案例數：{len(cases)}　thinking：{'on' if args.think else 'off'}")
    results = evaluate(cases, args.repeat, args.think, args.judge)
    print_report(results, MODEL, args.repeat)

    if args.compare:
        compare(results, args.compare)

    if args.save:
        # 存檔時捨棄逐次明細，基準線只需要彙總數字，才方便版控與 diff。
        payload = {
            "model": MODEL,
            "think": args.think,
            "repeat": args.repeat,
            "results": {k: {x: v[x] for x in v if x != "runs"} for k, v in results.items()},
        }
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"基準線已存檔：{args.save}")

    # 讓 CI 能依據結果決定成敗
    overall = statistics.mean(r["pass_rate"] for r in results.values())
    sys.exit(0 if overall == 1.0 else 1)


if __name__ == "__main__":
    main()
