"""第 1 層：可觀測性元件的單元測試。

統計程式碼本身也是程式碼，一樣會有 bug —— 而且**統計出錯比功能出錯更危險**，
因為你會拿著錯的數字做決策而不自知。
這裡用手工組出來的假 LLMResult 餵給 handler，完全不需要模型。
"""

import json
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from token_usage import TokenUsageCallbackHandler


def make_result(input_tokens=10, output_tokens=5, tool_calls=None, use_usage_metadata=True):
    """組一個假的 LLM 回應，模擬 langchain-ollama 實際回傳的結構。"""
    kwargs = {
        "content": "測試回應",
        "response_metadata": {"model": "qwen3.8:27b", "done_reason": "stop"},
    }
    if use_usage_metadata:
        kwargs["usage_metadata"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    else:
        # 舊版 langchain-ollama 不會填 usage_metadata，只留 Ollama 原生欄位。
        kwargs["response_metadata"].update(
            {"prompt_eval_count": input_tokens, "eval_count": output_tokens}
        )
    if tool_calls:
        kwargs["tool_calls"] = [
            {"name": n, "args": {}, "id": f"call_{i}"} for i, n in enumerate(tool_calls)
        ]
    return LLMResult(generations=[[ChatGeneration(message=AIMessage(**kwargs))]])


def feed(handler, result):
    """模擬一次完整的 LLM 呼叫生命週期。"""
    run_id = uuid4()
    handler.on_chat_model_start({}, [], run_id=run_id)
    handler.on_llm_end(result, run_id=run_id)


class TestTokenCounting:
    def test_單次呼叫的_token_被正確記錄(self):
        h = TokenUsageCallbackHandler(verbose=False)
        feed(h, make_result(input_tokens=100, output_tokens=20))

        assert len(h.llm_calls) == 1
        call = h.llm_calls[0]
        assert (call.input_tokens, call.output_tokens, call.total_tokens) == (100, 20, 120)
        assert call.model == "qwen3.8:27b"
        assert call.finish_reason == "stop"

    def test_多次呼叫會累加而非覆蓋(self):
        # Agent 的成本來自「累加」，這正是最該被測到的行為。
        h = TokenUsageCallbackHandler(verbose=False)
        feed(h, make_result(input_tokens=100, output_tokens=20))
        feed(h, make_result(input_tokens=150, output_tokens=30))

        s = h.summary()
        assert s["llm_calls"] == 2
        assert s["input_tokens"] == 250
        assert s["output_tokens"] == 50
        assert s["total_tokens"] == 300

    def test_沒有_usage_metadata_時退回讀取_ollama_原生欄位(self):
        # 這是 token_usage.py 裡的防禦分支，正常路徑測不到，必須單獨測。
        h = TokenUsageCallbackHandler(verbose=False)
        feed(h, make_result(input_tokens=77, output_tokens=8, use_usage_metadata=False))

        assert h.llm_calls[0].input_tokens == 77
        assert h.llm_calls[0].output_tokens == 8

    def test_步驟編號從1遞增(self):
        h = TokenUsageCallbackHandler(verbose=False)
        for _ in range(3):
            feed(h, make_result())
        assert [c.step for c in h.llm_calls] == [1, 2, 3]

    def test_空統計不會除以零(self):
        h = TokenUsageCallbackHandler(verbose=False)
        assert h.summary()["total_tokens"] == 0
        assert h.summary()["output_tps"] == 0.0
        assert "Token 使用量報告" in h.report()


class TestToolTracking:
    def test_記錄工具呼叫的輸入與輸出(self):
        h = TokenUsageCallbackHandler(verbose=False)
        run_id = uuid4()
        h.on_tool_start({"name": "search_policy"}, "{'keyword': '保固'}", run_id=run_id)
        h.on_tool_end(AIMessage(content="2 年原廠保固"), run_id=run_id)

        assert len(h.tool_calls) == 1
        assert h.tool_calls[0].name == "search_policy"
        # ToolMessage 要被拆出 content，不能存整個物件的 repr。
        assert h.tool_calls[0].output == "2 年原廠保固"
        assert h.tool_calls[0].error is None

    def test_工具例外被記錄且不中斷統計(self):
        h = TokenUsageCallbackHandler(verbose=False)
        run_id = uuid4()
        h.on_tool_start({"name": "calculate"}, "1/0", run_id=run_id)
        h.on_tool_error(ZeroDivisionError("division by zero"), run_id=run_id)

        assert h.tool_calls[0].error == "division by zero"
        assert h.summary()["tool_calls"] == 1

    def test_llm_決定呼叫哪些工具會被記錄下來(self):
        # 這是「軌跡」的來源：模型打算做什麼，而不只是它最後說了什麼。
        h = TokenUsageCallbackHandler(verbose=False)
        feed(h, make_result(tool_calls=["calculate", "get_current_time"]))
        assert h.llm_calls[0].tool_calls == ["calculate", "get_current_time"]


class TestJsonlExport:
    def test_事件被寫成_jsonl(self, tmp_path):
        path = tmp_path / "usage.jsonl"
        h = TokenUsageCallbackHandler(verbose=False, jsonl_path=str(path))
        feed(h, make_result(input_tokens=10, output_tokens=5, tool_calls=["calculate"]))

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["type"] == "LLMCall"
        assert rows[0]["total_tokens"] == 15
        assert rows[0]["tool_calls"] == ["calculate"]

    def test_未指定路徑時不寫檔(self, tmp_path):
        h = TokenUsageCallbackHandler(verbose=False)
        feed(h, make_result())
        assert not list(tmp_path.iterdir())
