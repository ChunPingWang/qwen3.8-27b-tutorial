"""第 1 層：工具的單元測試。

工具是**確定性**的純函式，可以用最傳統的單元測試斷言「等於」。
這一層完全不碰模型，毫秒級跑完，是 CI 上必跑的最基本防線。
"""

import pytest

from agent import calculate, get_current_time, search_policy

# 用 .invoke() 呼叫，走的是 LangChain 工具真正的入口（含參數驗證），
# 而不是繞過裝飾器直接呼叫底層函式。


class TestCalculate:
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("1 + 1", "2"),
            ("1280 * 3 * 0.85", "3264.0"),
            ("(10 + 5) * 2", "30"),
            ("2 ** 10", "1024"),
            ("-7 + 10", "3"),
        ],
    )
    def test_算式正確求值(self, expression, expected):
        assert calculate.invoke({"expression": expression}) == expected

    def test_除以零回傳錯誤訊息而非拋出例外(self):
        # 工具必須把錯誤「講給模型聽」，模型才有機會自行修正算式。
        result = calculate.invoke({"expression": "1 / 0"})
        assert result.startswith("計算失敗")

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('ls')",  # 任意程式碼執行
            "open('/etc/passwd').read()",     # 檔案讀取
            "[].__class__",                   # 屬性存取
        ],
    )
    def test_拒絕執行非算式內容(self, expression):
        # 模型的輸出等同不可信輸入，工具絕不能用 eval() 直接執行。
        result = calculate.invoke({"expression": expression})
        assert result.startswith("計算失敗")


class TestSearchPolicy:
    def test_命中關鍵字(self):
        assert "2 年" in search_policy.invoke({"keyword": "保固"})

    def test_關鍵字包含也能命中(self):
        assert "免運" in search_policy.invoke({"keyword": "運費多少"})

    def test_查無資料時提示可用主題(self):
        result = search_policy.invoke({"keyword": "外星人"})
        assert "查無" in result
        # 查不到時要告訴模型有哪些選項，它才知道下一步要改查什麼。
        assert "保固" in result


class TestGetCurrentTime:
    def test_預設為台北時區(self):
        assert "UTC+0800" in get_current_time.invoke({})

    def test_可指定其他時區(self):
        assert "UTC+0000" in get_current_time.invoke({"utc_offset_hours": 0})


class TestToolSchema:
    """工具的 docstring 與型別註記就是模型看到的說明書，本身值得測。"""

    @pytest.mark.parametrize("tool", [calculate, get_current_time, search_policy])
    def test_每個工具都有名稱與說明(self, tool):
        assert tool.name
        assert tool.description, f"{tool.name} 缺少 docstring，模型將無從判斷何時該用它"

    def test_參數schema正確產生(self):
        schema = search_policy.args_schema.model_json_schema()
        assert "keyword" in schema["properties"]
