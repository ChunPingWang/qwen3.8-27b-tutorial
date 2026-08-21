"""第 2 層：Agent 接線的整合測試（用假模型，不需要 Ollama）。

模型本身不確定，沒辦法斷言「等於」；但**Agent 的骨架是確定性的**：
工具有沒有註冊給模型、模型要求呼叫工具時有沒有真的被執行、
工具結果有沒有送回模型、callback 有沒有被觸發、保險絲有沒有生效 ——
這些全都可以用一個假模型測到毫秒級，而且百分之百穩定。

這一層抓的是「串接錯誤」，這類 bug 佔 Agent 實務問題的絕大多數。
"""

from typing import Any, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langgraph.prebuilt import create_react_agent

from agent import SYSTEM_PROMPT, TOOLS
from token_usage import TokenUsageCallbackHandler


class FakeToolCallingModel(BaseChatModel):
    """按照腳本依序回覆的假模型，用來精準控制 Agent 要走的路徑。"""

    responses: List[AIMessage]
    index: int = 0
    bound_tools: List[str] = []
    seen_messages: List[List[BaseMessage]] = []

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_messages.append(list(messages))
        # 腳本用完後就一直回最後一則，方便測試無限迴圈的情境。
        template = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        # 必須複製並清掉 id：LangGraph 的 add_messages 以 message id 判斷同一則訊息，
        # 重複回傳同一個物件會變成「取代」而不是「附加」，迴圈會提早結束。
        response = template.model_copy(deep=True)
        response.id = None
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolCallingModel":
        self.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"


def ai_tool_call(name: str, args: dict, tokens=(100, 20)) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}],
        usage_metadata={
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
        response_metadata={"model": "fake", "done_reason": "stop"},
    )


def ai_answer(text: str, tokens=(150, 30)) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
        response_metadata={"model": "fake", "done_reason": "stop"},
    )


def build(responses: List[AIMessage]):
    model = FakeToolCallingModel(responses=responses)
    return model, create_react_agent(model, TOOLS, prompt=SYSTEM_PROMPT)


class TestWiring:
    def test_所有工具都被註冊給模型(self):
        model, agent = build([ai_answer("好")])
        agent.invoke({"messages": [("user", "嗨")]})
        # 少註冊一個工具，模型永遠不會呼叫它，而且不會有任何錯誤訊息。
        assert set(model.bound_tools) == {"calculate", "get_current_time", "search_policy"}

    def test_system_prompt_有送進模型(self):
        model, agent = build([ai_answer("好")])
        agent.invoke({"messages": [("user", "嗨")]})
        first_turn = model.seen_messages[0]
        assert isinstance(first_turn[0], SystemMessage)
        assert first_turn[0].content == SYSTEM_PROMPT

    def test_模型要求呼叫工具時工具真的被執行且結果回饋給模型(self):
        model, agent = build(
            [
                ai_tool_call("calculate", {"expression": "1280 * 3 * 0.85"}),
                ai_answer("總金額是 3264 元"),
            ]
        )
        result = agent.invoke({"messages": [("user", "算一下")]})

        # 1) 工具的真實執行結果出現在對話裡
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert [m.content for m in tool_messages] == ["3264.0"]

        # 2) 第二輪模型確實看到了工具結果（而不是憑空作答）
        second_turn = model.seen_messages[1]
        assert any(isinstance(m, ToolMessage) and m.content == "3264.0" for m in second_turn)

        # 3) 最終回答是最後一則訊息
        assert result["messages"][-1].content == "總金額是 3264 元"

    def test_工具丟出的錯誤會被送回模型讓它重試(self):
        model, agent = build(
            [
                ai_tool_call("search_policy", {"keyword": "免運"}),   # 查無結果
                ai_tool_call("search_policy", {"keyword": "運費"}),   # 改查關鍵字
                ai_answer("滿 1000 元免運"),
            ]
        )
        result = agent.invoke({"messages": [("user", "免運門檻？")]})

        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert "查無" in tool_messages[0].content
        assert "免運" in tool_messages[1].content
        assert model.index == 3  # 模型被問了三次


class TestObservabilityIntegration:
    def test_callback_在真實_agent_流程中被正確觸發(self):
        model, agent = build(
            [
                ai_tool_call("get_current_time", {"utc_offset_hours": 8}, tokens=(100, 20)),
                ai_answer("現在是台北時間", tokens=(150, 30)),
            ]
        )
        tracker = TokenUsageCallbackHandler(verbose=False)
        agent.invoke({"messages": [("user", "幾點")]}, config={"callbacks": [tracker]})

        s = tracker.summary()
        assert s["llm_calls"] == 2
        assert s["tool_calls"] == 1
        assert s["input_tokens"] == 250   # 100 + 150
        assert s["output_tokens"] == 50   # 20 + 30
        assert tracker.llm_calls[0].tool_calls == ["get_current_time"]
        assert tracker.tool_calls[0].name == "get_current_time"


class TestSafetyNet:
    """沒有這道保險絲，模型卡在工具迴圈裡會把 token 一直燒下去。

    值得注意的是：`create_react_agent` 不會拋 GraphRecursionError，
    而是在步數快用完時**優雅降級**，回一則普通訊息收尾。
    這種「不是錯誤但也不是正常結果」的狀況，只有寫測試才會發現。
    """

    def test_無限工具迴圈會被步數上限擋下(self):
        model, agent = build([ai_tool_call("get_current_time", {})])  # 永遠只回工具呼叫
        result = agent.invoke(
            {"messages": [("user", "幾點")]},
            config={"recursion_limit": 6},
        )

        assert model.index <= 3, "模型被呼叫的次數必須被上限收斂"
        final = result["messages"][-1]
        assert not final.tool_calls, "應該以一則普通訊息收尾，而不是又一個工具呼叫"

    def test_token_統計在被中斷時仍然保留(self):
        # 出事時最需要知道「燒掉多少」，統計不能跟著中斷一起消失。
        _, agent = build([ai_tool_call("get_current_time", {}, tokens=(100, 20))])
        tracker = TokenUsageCallbackHandler(verbose=False)
        agent.invoke(
            {"messages": [("user", "幾點")]},
            config={"callbacks": [tracker], "recursion_limit": 6},
        )
        assert tracker.summary()["total_tokens"] > 0
