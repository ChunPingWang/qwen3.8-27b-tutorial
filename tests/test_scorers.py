"""第 1 層：評分函式的單元測試。

評分器是**用來判斷模型好壞的尺**。尺本身歪了，量出來的分數全都不能信 ——
而且錯得很隱蔽：你會以為是模型退步，實際上是評分器有 bug。
所以評分器必須用最傳統的單元測試釘死。
"""

import pytest

from evals.scorers import normalize, score_budget, score_contains, score_tools


class TestNormalize:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("**3,264 元**", "3264元"),          # 千分位 + markdown 粗體
            ("2026 年 8 月 21 日", "2026-08-21"),  # 中文日期
            ("2026年8月5日", "2026-08-05"),        # 個位數月日要補零
            ("總計 １２３", "總計123"),             # 全形數字
        ],
    )
    def test_同一個值的不同寫法會被正規化成一致(self, raw, expected):
        assert normalize(raw) == expected

    def test_不改動語意(self):
        # 正規化只能處理格式，不能動到內容，否則會製造假陽性。
        assert normalize("沒有免運") == "沒有免運"
        assert normalize("3265") != normalize("3264")


class TestScoreContains:
    def test_答對但格式不同仍算通過(self):
        # 這正是評測第一版誤判的情境。
        assert score_contains("總金額為 **3,264 元**", ["3264"]).passed
        assert score_contains("現在是 2026 年 8 月 21 日", ["2026-08-21"]).passed
        assert score_contains("享 2 年原廠保固", ["2 年"]).passed

    def test_真的答錯要判不通過(self):
        score = score_contains("總金額為 3,265 元", ["3264"])
        assert not score.passed
        assert "3264" in score.detail

    def test_多個期望值需全部滿足(self):
        assert not score_contains("金額 3264 元", ["3264", "免運"]).passed
        assert score_contains("金額 3264 元，免運", ["3264", "免運"]).passed

    def test_沒有期望值時不計分(self):
        assert score_contains("任何答案", []) is None


class TestScoreTools:
    def test_必要工具沒被呼叫要判不通過(self):
        # 「答案矇對但沒查資料」是最危險的假陽性，必須靠軌跡評測抓出來。
        score = score_tools(used=[], expected=["search_policy"], forbidden=[])
        assert not score.passed
        assert "search_policy" in score.detail

    def test_呼叫順序不影響通過與否(self):
        assert score_tools(
            used=["search_policy", "calculate"],
            expected=["calculate", "search_policy"],
            forbidden=[],
        ).passed

    def test_多呼叫額外工具不算錯除非被禁用(self):
        assert score_tools(["calculate", "get_current_time"], ["calculate"], []).passed
        assert not score_tools(["calculate", "search_policy"], ["calculate"], ["search_policy"]).passed

    def test_沒有任何工具期望時不計分(self):
        assert score_tools([], [], []) is None


class TestScoreBudget:
    def test_超出呼叫次數上限判不通過(self):
        assert score_budget(llm_calls=2, max_llm_calls=3).passed
        assert not score_budget(llm_calls=5, max_llm_calls=3).passed

    def test_剛好等於上限算通過(self):
        assert score_budget(llm_calls=3, max_llm_calls=3).passed
