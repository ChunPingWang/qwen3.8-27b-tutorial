"""可觀測性工具：用 LangChain callback 攔截 Agent 執行期間的每一次 LLM 呼叫與工具呼叫。

Ollama 在每次回應中都會帶回 `prompt_eval_count`（輸入 token）與 `eval_count`（輸出 token），
langchain-ollama 會把它們正規化成 `AIMessage.usage_metadata`。
只要掛上 callback，就能在不改動 Agent 邏輯的前提下把這些數字全部收集起來。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass
class LLMCall:
    """一次 LLM 呼叫（= Agent 迴圈中的一步）的觀測紀錄。"""

    step: int
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_s: float
    finish_reason: str = ""
    tool_calls: List[str] = field(default_factory=list)

    @property
    def output_tps(self) -> float:
        """輸出 token / 秒，衡量本地推論吞吐量。"""
        return self.output_tokens / self.latency_s if self.latency_s > 0 else 0.0


@dataclass
class ToolCall:
    """一次工具執行的觀測紀錄。"""

    name: str
    input: str
    output: str
    latency_s: float
    error: Optional[str] = None


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """收集 token 使用量、延遲與工具呼叫的 callback handler。

    用法：
        tracker = TokenUsageCallbackHandler()
        agent.invoke(payload, config={"callbacks": [tracker]})
        print(tracker.report())
    """

    def __init__(self, verbose: bool = True, jsonl_path: Optional[str] = None) -> None:
        self.llm_calls: List[LLMCall] = []
        self.tool_calls: List[ToolCall] = []
        self.verbose = verbose
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._llm_starts: Dict[Any, float] = {}
        self._tool_starts: Dict[Any, Dict[str, Any]] = {}

    # ---------- LLM ----------

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs) -> None:
        self._llm_starts[run_id] = time.perf_counter()

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs) -> None:
        self._llm_starts[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id=None, **kwargs) -> None:
        started = self._llm_starts.pop(run_id, None)
        latency = time.perf_counter() - started if started is not None else 0.0

        generation = response.generations[0][0]
        message = getattr(generation, "message", None)
        usage = dict(getattr(message, "usage_metadata", None) or {})
        meta = dict(getattr(message, "response_metadata", None) or {})

        # usage_metadata 是 LangChain 的統一格式；抓不到時退回 Ollama 原生欄位。
        input_tokens = usage.get("input_tokens") or meta.get("prompt_eval_count") or 0
        output_tokens = usage.get("output_tokens") or meta.get("eval_count") or 0
        total_tokens = usage.get("total_tokens") or (input_tokens + output_tokens)

        call = LLMCall(
            step=len(self.llm_calls) + 1,
            model=meta.get("model", "unknown"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_s=latency,
            finish_reason=meta.get("done_reason", ""),
            tool_calls=[tc["name"] for tc in (getattr(message, "tool_calls", None) or [])],
        )
        self.llm_calls.append(call)
        self._emit(call)

        if self.verbose:
            planned = f" → 準備呼叫 {', '.join(call.tool_calls)}" if call.tool_calls else ""
            print(
                f"  [LLM #{call.step}] in={call.input_tokens} out={call.output_tokens} "
                f"total={call.total_tokens} ({call.latency_s:.1f}s, "
                f"{call.output_tps:.1f} tok/s){planned}"
            )

    def on_llm_error(self, error: BaseException, *, run_id=None, **kwargs) -> None:
        self._llm_starts.pop(run_id, None)
        if self.verbose:
            print(f"  [LLM error] {error}")

    # ---------- Tool ----------

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs) -> None:
        self._tool_starts[run_id] = {
            "name": (serialized or {}).get("name", "unknown"),
            "input": input_str,
            "t0": time.perf_counter(),
        }

    def on_tool_end(self, output, *, run_id=None, **kwargs) -> None:
        started = self._tool_starts.pop(run_id, None)
        if started is None:
            return
        # LangGraph 的 ToolNode 會包成 ToolMessage，取 content 才是工具真正的回傳值。
        text = getattr(output, "content", output)
        call = ToolCall(
            name=started["name"],
            input=str(started["input"])[:200],
            output=str(text)[:200],
            latency_s=time.perf_counter() - started["t0"],
        )
        self.tool_calls.append(call)
        self._emit(call)
        if self.verbose:
            print(f"  [Tool {call.name}] {call.input} → {call.output} ({call.latency_s:.2f}s)")

    def on_tool_error(self, error: BaseException, *, run_id=None, **kwargs) -> None:
        started = self._tool_starts.pop(run_id, None)
        if started is None:
            return
        call = ToolCall(
            name=started["name"],
            input=str(started["input"])[:200],
            output="",
            latency_s=time.perf_counter() - started["t0"],
            error=str(error),
        )
        self.tool_calls.append(call)
        self._emit(call)
        if self.verbose:
            print(f"  [Tool {call.name} error] {error}")

    # ---------- 匯出 ----------

    def _emit(self, record: Any) -> None:
        """把單筆事件附加寫入 JSONL，方便之後餵給任何觀測系統。"""
        if self.jsonl_path is None:
            return
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"type": type(record).__name__, "ts": time.time(), **asdict(record)}
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def summary(self) -> Dict[str, Any]:
        input_tokens = sum(c.input_tokens for c in self.llm_calls)
        output_tokens = sum(c.output_tokens for c in self.llm_calls)
        llm_seconds = sum(c.latency_s for c in self.llm_calls)
        return {
            "llm_calls": len(self.llm_calls),
            "tool_calls": len(self.tool_calls),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "llm_seconds": round(llm_seconds, 2),
            "tool_seconds": round(sum(c.latency_s for c in self.tool_calls), 2),
            "output_tps": round(output_tokens / llm_seconds, 1) if llm_seconds else 0.0,
        }

    def report(self) -> str:
        """列印一份人類可讀的 token 帳單。"""
        lines = ["", "=" * 74, "Token 使用量報告", "=" * 74]
        # 表頭刻意用等寬的 ASCII，避免全形中文把欄位撐歪。
        header = f"{'step':<6}{'input':>8}{'output':>8}{'total':>8}{'sec':>8}{'tok/s':>9}  tools"
        lines += [header, "-" * 74]
        for c in self.llm_calls:
            tools = ", ".join(c.tool_calls) or "-"
            lines.append(
                f"{c.step:<6}{c.input_tokens:>8}{c.output_tokens:>8}{c.total_tokens:>8}"
                f"{c.latency_s:>8.1f}{c.output_tps:>9.1f}  {tools}"
            )
        s = self.summary()
        lines += [
            "-" * 74,
            f"{'TOTAL':<6}{s['input_tokens']:>8}{s['output_tokens']:>8}{s['total_tokens']:>8}"
            f"{s['llm_seconds']:>8.1f}{s['output_tps']:>9.1f}",
            "",
            f"LLM 呼叫次數：{s['llm_calls']}　工具呼叫次數：{s['tool_calls']}"
            f"　工具耗時：{s['tool_seconds']}s",
        ]
        if self.jsonl_path:
            lines.append(f"事件明細已寫入：{self.jsonl_path}")
        lines.append("=" * 74)
        return "\n".join(lines)
