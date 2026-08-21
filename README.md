# Qwen3.8 27B × Ollama × LangChain Agent（含 Token 可觀測性）

在本機用 Ollama 跑 **Qwen3.8 27B**，並用 LangChain / LangGraph 做一個會呼叫工具的 Agent。
重點在於：**每一步花了多少 token、跑了多久、呼叫了哪些工具，全部看得見。**

```
--- Agent 執行軌跡 ---
  [LLM #1] in=525 out=94 total=619 (17.9s, 5.2 tok/s) → 準備呼叫 calculate, search_policy, get_current_time
  [Tool calculate] {'expression': '1280 * 3 * 0.85'} → 3264.0 (0.00s)
  [Tool search_policy] {'keyword': '免運'} → 知識庫中查無「免運」，可用主題：退貨、保固、運費、付款 (0.00s)
  [Tool get_current_time] {'utc_offset_hours': 8} → 2026-08-21 14:26:51 (UTC+0800) (0.00s)
  [LLM #2] in=700 out=26 total=726 (3.0s, 8.7 tok/s) → 準備呼叫 search_policy
  [Tool search_policy] {'keyword': '運費'} → 單筆訂單滿 1000 元免運，未滿收取 80 元運費。 (0.00s)
  [LLM #3] in=767 out=73 total=840 (7.9s, 9.3 tok/s)
```

---

## 目錄

- [1. 環境需求](#1-環境需求)
- [2. 安裝 Ollama](#2-安裝-ollama)
- [3. 下載 Qwen3.8 27B](#3-下載-qwen38-27b)
- [4. 啟用與測試模型](#4-啟用與測試模型)
- [5. 建立自訂上下文長度的變體](#5-建立自訂上下文長度的變體)
- [6. 專案安裝](#6-專案安裝)
- [7. 執行 Agent](#7-執行-agent)
- [8. Agent 是怎麼寫的](#8-agent-是怎麼寫的)
- [9. Token 可觀測性原理](#9-token-可觀測性原理)
- [10. 進階：接到正式的觀測系統](#10-進階接到正式的觀測系統)
- [11. 如何評估 Agent：單元測試 vs Eval](#11-如何評估-agent單元測試-vs-eval)
- [12. 疑難排解](#12-疑難排解)

---

## 1. 環境需求

| 項目 | 需求 | 本文驗證環境 |
|---|---|---|
| 作業系統 | macOS / Linux / Windows | macOS (Darwin 25.5) |
| 記憶體 | **建議 32 GB 以上**（模型本身 17 GB，需常駐 VRAM/統一記憶體） | Apple M5 / 32 GB |
| 磁碟 | 至少 20 GB 可用空間 | — |
| Ollama | >= 0.32.12（Qwen3.8 的 `qwen35` 架構需要） | 0.32.13 |
| Python | >= 3.9 | 3.9.6 |

> Qwen3.8 27B 是 Q4_K_M 量化、27.3B 參數的模型，並且同時具備
> **completion / tools / thinking / vision** 四種能力。工具呼叫（tools）正是本專案 Agent 的基礎。

---

## 2. 安裝 Ollama

macOS：

```bash
brew install ollama        # 或到 https://ollama.com/download 下載 App
brew services start ollama # 讓 Ollama 服務常駐於背景
```

Linux：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

確認版本與服務狀態：

```bash
ollama --version
curl -s http://localhost:11434/api/version
```

Ollama 預設監聽 `http://localhost:11434`，後面 LangChain 就是連這個位址。

---

## 3. 下載 Qwen3.8 27B

```bash
ollama pull qwen3.8:27b
```

下載約 17 GB，時間取決於網路。完成後確認：

```bash
ollama list
# NAME               ID              SIZE     MODIFIED
# qwen3.8:27b        22130167c4c2    17 GB    ...
```

查看模型規格（架構、上下文長度、能力）：

```bash
ollama show qwen3.8:27b
```

```
  architecture        qwen35
  parameters          27.3B
  context length      262144
  quantization        Q4_K_M
  Capabilities        completion, vision, tools, thinking
```

---

## 4. 啟用與測試模型

### 4.1 互動式對話

```bash
ollama run qwen3.8:27b
>>> 用一句話解釋什麼是 AI Agent
>>> /bye     # 離開
```

第一次載入需要把 17 GB 權重讀進記憶體，會等數十秒；之後就會常駐。

### 4.2 確認模型有沒有在跑

```bash
ollama ps
# NAME           SIZE     PROCESSOR    CONTEXT    UNTIL
# qwen3.8:27b    17 GB    100% GPU     8192       29 minutes from now
```

- `PROCESSOR` 顯示 `100% GPU` 代表完全跑在 GPU 上（最快）。若出現 `CPU` 比例，表示記憶體不足被迫分流，速度會明顯下降。
- `UNTIL` 是模型卸載倒數，預設閒置 5 分鐘後釋放記憶體。想讓它一直待命：

```bash
OLLAMA_KEEP_ALIVE=-1 ollama serve      # 或在單次請求中帶 "keep_alive": -1
```

### 4.3 用 REST API 測試（Agent 走的就是這條路）

```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "qwen3.8:27b",
  "messages": [{"role": "user", "content": "1+1 等於多少？只回答數字"}],
  "think": false,
  "stream": false
}' | python3 -m json.tool
```

回應中這兩個欄位就是**本專案 token 統計的原始來源**：

```json
{
  "prompt_eval_count": 22,      // 輸入 token 數
  "eval_count": 2,              // 輸出 token 數
  "total_duration": 7842088875  // 奈秒
}
```

### 4.4 關於 thinking 模式

Qwen3.8 支援思考模式。思考內容也會計入 `eval_count`，所以**開啟 thinking 會讓輸出 token 數大幅上升、Agent 變慢**。
本專案預設關閉，需要時用 `--think` 開啟。同一個問題（「保固多久？」）在本機實測的差異：

| 模式 | 輸出 token | 總 token | 總耗時 |
|---|---|---|---|
| `--think` 關閉（預設） | 48 | 1086 | 7.6s |
| `--think` 開啟 | 108（+125%） | 1142 | 19.9s（+162%） |

這正是可觀測性的價值：**「開 thinking 比較貴」是常識，貴多少則要量出來。**
單純查資料的任務關掉 thinking 即可；需要多步推理或數學時再開。

---

## 5. 建立自訂上下文長度的變體

Qwen3.8 原生支援到 262144 token 的上下文，但**上下文開越大，KV cache 佔用的記憶體越多**。
實務上會建立一個固定 `num_ctx` 的變體，避免每次呼叫都要傳參數：

```bash
cat > Modelfile <<'EOF'
FROM qwen3.8:27b
PARAMETER num_ctx 32768
EOF

ollama create qwen3.8:27b-32k -f Modelfile
ollama list   # 會多出 qwen3.8:27b-32k
```

要用哪個模型，透過環境變數切換即可：

```bash
OLLAMA_MODEL=qwen3.8:27b-32k python agent.py "..."
```

---

## 6. 專案安裝

```bash
cd ~/workspace/qwen3.8-27b-tutorial

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`：

```
langchain-ollama==0.3.10
langgraph==0.6.11
```

專案結構：

```
qwen3.8-27b-tutorial/
├── README.md
├── requirements.txt          # 執行 Agent 所需
├── requirements-dev.txt      # 額外加上 pytest
├── agent.py                  # Agent 本體：模型設定、工具、執行進入點
├── token_usage.py            # 可觀測性：token / 延遲 / 工具呼叫的 callback handler
├── tests/                    # 第 1、2 層：單元測試與接線測試（不需要 Ollama）
│   ├── test_tools.py
│   ├── test_token_usage.py
│   ├── test_scorers.py
│   └── test_agent_wiring.py
├── evals/                    # 第 3 層：真的呼叫模型的評測
│   ├── cases.py              # 評測資料集
│   ├── scorers.py            # 評分函式
│   ├── run_eval.py           # 執行器：跑資料集、算分、比對基準線
│   └── baseline.json         # 基準線（納入版控，方便 review 分數變動）
└── runs/usage.jsonl          # 執行後自動產生的事件紀錄（已加入 .gitignore）
```

若只是要跑 Agent，`requirements.txt` 就夠了；要跑測試與評測請改裝 `requirements-dev.txt`。
詳見 [第 11 節：如何評估 Agent](#11-如何評估-agent單元測試-vs-eval)。

可用的環境變數：

| 變數 | 預設值 | 說明 |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3.8:27b` | 要使用的模型 tag |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 服務位址 |

---

## 7. 執行 Agent

```bash
python agent.py                       # 跑內建示範問題
python agent.py "保固多久？"            # 問自己的問題
python agent.py --think "..."          # 開啟 thinking 模式
python agent.py --num-ctx 32768 "..."  # 指定上下文長度
```

實際輸出（M5 / 32GB，thinking 關閉）：

```
模型：qwen3.8:27b　thinking：off　num_ctx：8192
問題：保固多久？

--- Agent 執行軌跡 ---
  [LLM #1] in=487 out=27 total=514 (5.4s, 5.0 tok/s) → 準備呼叫 search_policy
  [Tool search_policy] {'keyword': '保固'} → 所有硬體享 2 年原廠保固，人為損壞不在保固範圍內。 (0.00s)
  [LLM #2] in=551 out=21 total=572 (2.3s, 9.3 tok/s)

--- 最終回答 ---
所有硬體享 2 年原廠保固，人為損壞不在保固範圍內。

==========================================================================
Token 使用量報告
==========================================================================
step     input  output   total     sec    tok/s  tools
--------------------------------------------------------------------------
1          487      27     514     5.4      5.0  search_policy
2          551      21     572     2.3      9.3  -
--------------------------------------------------------------------------
TOTAL     1038      48    1086     7.6      6.3

LLM 呼叫次數：2　工具呼叫次數：1　工具耗時：0.0s
事件明細已寫入：runs/usage.jsonl
==========================================================================
```

### 從這份報告可以讀出什麼

1. **一個問題 = 多次 LLM 呼叫。** ReAct Agent 每呼叫一次工具就要再問模型一次，token 是「累加」的，不是問一次就結束。
2. **輸入 token 一路變大**（487 → 551）。因為每一步都會把先前的對話與工具結果重新送進模型 —— 這就是 Agent 成本失控的主因。
3. **工具本身幾乎不花時間**（0.00s），時間全部花在模型推論上。要優化 Agent 延遲，該調的是模型與提示詞，不是工具。
4. 在上面較複雜的示範中，模型第一次查 `免運` 查無結果，**自己改用 `運費` 重查** —— 這次失敗的重試多花了 726 個 token。這種浪費只有在有觀測數據時才看得見。

---

## 8. Agent 是怎麼寫的

### 8.1 接上本地模型

`ChatOllama` 直接指向本機服務，不需要任何 API key：

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3.8:27b",
    base_url="http://localhost:11434",
    temperature=0,
    num_ctx=8192,      # 上下文視窗
    reasoning=False,   # Qwen3.8 的 thinking 開關
)
```

### 8.2 定義工具

用 `@tool` 裝飾器把普通函式變成工具，**docstring 就是模型看到的工具說明**，寫得越清楚，模型選錯工具的機率越低：

```python
from langchain_core.tools import tool

@tool
def search_policy(keyword: str) -> str:
    """查詢客服政策知識庫（退貨、保固、運費、付款）。傳入一個關鍵字。"""
    ...
```

本專案內建三個工具：

| 工具 | 用途 |
|---|---|
| `calculate` | 數學運算（用 AST 解析，不使用 `eval()`） |
| `get_current_time` | 取得指定時區的目前時間 |
| `search_policy` | 查詢內建的客服政策知識庫 |

工具出錯時要**把錯誤訊息回傳給模型**而不是拋出例外，模型才有機會自己修正：

```python
except Exception as exc:
    return f"計算失敗：{exc}"
```

### 8.3 組出 ReAct Agent

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, [calculate, get_current_time, search_policy],
                           prompt=SYSTEM_PROMPT)

result = agent.invoke(
    {"messages": [("user", question)]},
    config={"callbacks": [tracker], "recursion_limit": 12},
)
print(result["messages"][-1].content)
```

`create_react_agent` 幫你處理掉整個迴圈：**問模型 → 模型要求呼叫工具 → 執行工具 → 把結果送回模型 → 直到模型給出最終答案**。
`recursion_limit` 是保險絲，防止模型卡在無限工具迴圈裡把 token 燒光。

---

## 9. Token 可觀測性原理

### 9.1 資料從哪裡來

```
Ollama 回應                LangChain 正規化              本專案
prompt_eval_count   ──▶   usage_metadata.input_tokens   ──▶  LLMCall.input_tokens
eval_count          ──▶   usage_metadata.output_tokens  ──▶  LLMCall.output_tokens
done_reason / model ──▶   response_metadata             ──▶  finish_reason / model
```

即使是本地模型，`langchain-ollama` 一樣會填好 `usage_metadata`，格式與 OpenAI / Anthropic 完全一致。
換句話說，**這套統計程式碼換到雲端模型也不用改**。

### 9.2 用 callback 攔截，不污染業務邏輯

關鍵在於「不要為了統計去改 Agent 的程式碼」。LangChain 的 callback 機制讓觀測完全外掛：

```python
class TokenUsageCallbackHandler(BaseCallbackHandler):
    def on_llm_end(self, response, *, run_id=None, **kwargs):
        message = response.generations[0][0].message
        usage = dict(message.usage_metadata or {})
        meta = dict(message.response_metadata or {})

        input_tokens = usage.get("input_tokens") or meta.get("prompt_eval_count") or 0
        output_tokens = usage.get("output_tokens") or meta.get("eval_count") or 0
        ...
```

掛上去只要一行：

```python
tracker = TokenUsageCallbackHandler(jsonl_path="runs/usage.jsonl")
agent.invoke(payload, config={"callbacks": [tracker]})
print(tracker.report())
```

`token_usage.py` 監聽的事件：

| Callback | 收集的資訊 |
|---|---|
| `on_chat_model_start` / `on_llm_start` | 開始計時 |
| `on_llm_end` | 輸入/輸出/合計 token、延遲、tok/s、`done_reason`、這一步決定呼叫哪些工具 |
| `on_llm_error` | 模型呼叫失敗 |
| `on_tool_start` / `on_tool_end` | 工具名稱、輸入、輸出、耗時 |
| `on_tool_error` | 工具例外 |

### 9.3 程式化取用統計

`tracker.summary()` 回傳結構化資料，可以直接寫進資料庫或當作測試斷言：

```python
{'llm_calls': 2, 'tool_calls': 1, 'input_tokens': 1038, 'output_tokens': 48,
 'total_tokens': 1086, 'llm_seconds': 7.64, 'tool_seconds': 0.0, 'output_tps': 6.3}
```

例如在 CI 裡防止提示詞改壞導致成本暴增：

```python
assert tracker.summary()["total_tokens"] < 3000
assert tracker.summary()["llm_calls"] <= 5
```

### 9.4 JSONL 事件流

每一筆事件都會即時追加到 `runs/usage.jsonl`，一行一個 JSON：

```json
{"type": "LLMCall", "ts": 1787293650.6, "step": 1, "model": "qwen3.8:27b", "input_tokens": 487, "output_tokens": 27, "total_tokens": 514, "latency_s": 5.38, "finish_reason": "stop", "tool_calls": ["search_policy"]}
{"type": "ToolCall", "ts": 1787293650.6, "name": "search_policy", "input": "{'keyword': '保固'}", "output": "所有硬體享 2 年原廠保固...", "latency_s": 0.0002, "error": null}
```

快速統計歷史累計用量：

```bash
python3 - <<'EOF'
import json
rows = [json.loads(l) for l in open("runs/usage.jsonl")]
llm = [r for r in rows if r["type"] == "LLMCall"]
print("累計 LLM 呼叫：", len(llm))
print("累計 token：", sum(r["total_tokens"] for r in llm))
EOF
```

### 9.5 順帶談成本

本地推論沒有 API 費用，但 token 數依然是**規劃成本的核心指標**。
把 token 數乘上任一雲端模型的單價，就能估算「同一個 Agent 改用雲端模型要花多少錢」，
也能反過來估算本地部署的回本點：

```python
s = tracker.summary()
usd = s["input_tokens"] / 1e6 * PRICE_IN + s["output_tokens"] / 1e6 * PRICE_OUT
```

---

## 10. 進階：接到正式的觀測系統

`TokenUsageCallbackHandler` 是刻意寫成最小可讀版本，方便理解原理。實務上可以：

- **LangSmith**：設定 `LANGSMITH_TRACING=true` 與 `LANGSMITH_API_KEY`，自動取得完整 trace 樹狀圖。
- **OpenTelemetry**：在 `on_llm_end` 中把 token 數寫成 metric，送到 Prometheus / Grafana。
- **多 handler 併用**：`config={"callbacks": [tracker, my_otel_handler, LangSmithTracer()]}`，彼此不衝突。
- **官方彙總 handler**：`from langchain_core.callbacks import UsageMetadataCallbackHandler` 提供不含延遲與工具資訊的極簡版總計。

---

## 11. 如何評估 Agent：單元測試 vs Eval

### 11.1 先分清楚：什麼能測、什麼只能評

最容易踩的坑，是想用單元測試去測模型的回答。

> **模型的輸出是不確定的，沒辦法斷言「等於」。**
> 硬要這樣做，只會得到一個時好時壞的 flaky test，最後被整個團隊 skip 掉。

正確的做法是把系統切成兩半：**確定性的部分用測試釘死，不確定的部分用 eval 量分數。**

| 層次 | 問的問題 | 測的對象 | 需要模型？ | 速度 | 結果形式 | 何時跑 |
|---|---|---|---|---|---|---|
| **1. 單元測試** | 壞了沒？ | 工具、評分器、統計程式碼 | 否 | 毫秒 | pass / fail | 每次 commit |
| **2. 接線測試** | 串對了沒？ | Agent 骨架（用假模型） | 假模型 | 毫秒 | pass / fail | 每次 commit |
| **3. Eval 評測** | 現在多好？比上次好還壞？ | 模型的實際行為 | 是 | 分鐘 | 分數 / 通過率 | 改 prompt、換模型、發版前 |

所以「這屬於單元測試嗎？」的答案是 **一半一半**：

- 工具函式、token 統計、評分器、Agent 接線 → **是**，而且應該用最傳統的單元測試
- 模型答得對不對、會不會亂用工具、會不會編造 → **不是測試，是 eval**

三層在這個專案裡分別對應：

```
tests/test_tools.py         ← 第 1 層：工具是純函式，可斷言「等於」
tests/test_token_usage.py   ← 第 1 層：統計元件（用假的 LLMResult）
tests/test_scorers.py       ← 第 1 層：評分器本身
tests/test_agent_wiring.py  ← 第 2 層：Agent 接線（用假模型）
evals/                      ← 第 3 層：真的呼叫模型，算分數
```

```bash
pip install -r requirements-dev.txt
pytest                       # 第 1、2 層：50 個測試 0.24 秒跑完，不需要 Ollama
python -m evals.run_eval     # 第 3 層：真的跑模型，約 1 分鐘
```

---

### 11.2 第 1 層：單元測試 —— 測確定性的程式碼

工具是純函式，可以用最傳統的方式測：

```python
@pytest.mark.parametrize("expression,expected", [
    ("1 + 1", "2"),
    ("1280 * 3 * 0.85", "3264.0"),
])
def test_算式正確求值(self, expression, expected):
    assert calculate.invoke({"expression": expression}) == expected
```

有三類案例特別值得寫，因為它們對應的都是真實事故：

**1) 工具的安全性 —— 模型的輸出等同不可信輸入**

```python
@pytest.mark.parametrize("expression", [
    "__import__('os').system('ls')",
    "open('/etc/passwd').read()",
])
def test_拒絕執行非算式內容(self, expression):
    assert calculate.invoke({"expression": expression}).startswith("計算失敗")
```

**2) 錯誤要「講給模型聽」而不是拋出例外** —— 這是 Agent 能自我修正的前提：

```python
def test_除以零回傳錯誤訊息而非拋出例外(self):
    assert calculate.invoke({"expression": "1 / 0"}).startswith("計算失敗")
```

**3) 統計元件本身** —— 統計出錯比功能出錯更危險，因為你會拿著錯的數字做決策而不自知：

```python
def test_多次呼叫會累加而非覆蓋(self):
    h = TokenUsageCallbackHandler(verbose=False)
    feed(h, make_result(input_tokens=100, output_tokens=20))
    feed(h, make_result(input_tokens=150, output_tokens=30))
    assert h.summary()["total_tokens"] == 300
```

`make_result()` 是手工組出來的假 `LLMResult`，完全不需要模型 —— 連
「舊版沒有 `usage_metadata` 時退回讀 `prompt_eval_count`」這種平常走不到的防禦分支都測得到。

---

### 11.3 第 2 層：用假模型測 Agent 的「接線」

這一層最容易被忽略，卻抓得到**最多**的 bug。

模型本身不確定，但 **Agent 的骨架是確定性的**：工具有沒有註冊給模型、模型要求呼叫工具時
有沒有真的被執行、工具結果有沒有送回模型、callback 有沒有被觸發、保險絲有沒有生效 ——
這些跟模型聰不聰明完全無關，可以用一個按腳本回覆的假模型測到毫秒級。

```python
class FakeToolCallingModel(BaseChatModel):
    """按照腳本依序回覆的假模型，用來精準控制 Agent 要走的路徑。"""
    responses: List[AIMessage]
    index: int = 0
    bound_tools: List[str] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        template = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        response = template.model_copy(deep=True)
        response.id = None       # ← 這行很關鍵，原因見下
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [t.name for t in tools]
        return self
```

有了它就能寫出 100% 穩定的斷言：

```python
def test_模型要求呼叫工具時工具真的被執行且結果回饋給模型(self):
    model, agent = build([
        ai_tool_call("calculate", {"expression": "1280 * 3 * 0.85"}),
        ai_answer("總金額是 3264 元"),
    ])
    result = agent.invoke({"messages": [("user", "算一下")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.content for m in tool_messages] == ["3264.0"]          # 工具真的跑了
    assert any(isinstance(m, ToolMessage) for m in model.seen_messages[1])  # 結果回饋了
    assert result["messages"][-1].content == "總金額是 3264 元"
```

> **這一層在本專案實際抓到兩個坑，兩個都不是靠讀文件會發現的：**
>
> 1. **LangGraph 以 message id 判斷同一則訊息。** 假模型若每次回傳*同一個* `AIMessage`
>    物件，第二次會變成「取代」而非「附加」，工具迴圈提早結束，測試假性通過。
>    必須 `model_copy(deep=True)` 並把 `id` 清成 `None`。
> 2. **`create_react_agent` 步數用盡時不會拋 `GraphRecursionError`**，而是優雅降級回
>    一則普通訊息（"Sorry, need more steps..."）。如果你的程式碼在等一個例外來處理逾時，
>    永遠等不到 —— 這種「不是錯誤但也不是正常結果」的狀況，只有寫測試才會發現。

---

### 11.4 第 3 層：Eval —— 量模型到底有多好

**Eval 不是 pass/fail，是基準線。** 它的價值不在於某一次跑了幾分，
而在於你改 prompt、換模型、調 `num_ctx` 之後，**跟上次比是進步還是退步、代價是多少**。

#### 資料集

Eval 的品質等於資料集的品質。每個案例都要有可自動判定的期望（`evals/cases.py`）：

```python
EvalCase(
    id="multi-step",
    question="3 台單價 1280 元的設備打 85 折後總共多少？這個金額有免運嗎？",
    expect_contains=["3264", "免運"],          # 結果：答案該長什麼樣
    expect_tools=["calculate", "search_policy"],  # 軌跡：過程該怎麼走
    max_llm_calls=5,                           # 成本護欄
),
```

挑案例的四個原則：

1. 涵蓋**每一個工具**，以及**多工具組合**的情境
2. 一定要有**不該用工具**的案例（`no-tool-needed`）—— Agent 最常見的毛病是濫用工具
3. 一定要有**知識庫查不到**的案例（`not-in-kb`）—— 檢驗模型會不會編造答案
4. 時間之類的動態答案，期望值要在執行時計算，不能寫死

#### 三種評分方式，優先順序很明確

> **能用程式判定的，絕不交給模型判定。** 規則式評分免費、瞬間完成、百分之百可重現；
> LLM-as-judge 貴、慢，而且自己也會出錯。

| 方式 | 適用 | 代價 |
|---|---|---|
| **規則比對** `score_contains` | 有標準答案（數字、日期、關鍵字） | 免費、瞬間 |
| **軌跡評測** `score_tools` | Agent 專屬：該呼叫哪些工具 | 免費、瞬間 |
| **成本護欄** `score_budget` | LLM 呼叫次數上限 | 免費、瞬間 |
| **LLM-as-judge** `score_llm_judge` | 開放式判斷（有沒有編造事實） | 一次額外推論 |

**軌跡評測是 Agent 評測與一般 LLM 評測最大的差別**，值得特別強調：

```python
def score_tools(used, expected, forbidden):
    """「答案對但路徑錯」（例如沒查知識庫、憑記憶硬答）是最危險的假陽性 ——
    這次矇對了，換個問題就會錯，而且你完全不知道為什麼。"""
```

用 judge 時只給它**二元判斷 + 明確準則**，不要叫它打 1~10 分（分數極不穩定），
並且要把 judge 本身當成另一個需要被抽驗的元件。

#### 執行與報告

```bash
python -m evals.run_eval                       # 跑全部
python -m evals.run_eval --repeat 3            # 每題跑 3 次，量穩定度
python -m evals.run_eval --judge               # 啟用 LLM-as-judge
python -m evals.run_eval --save evals/baseline.json
python -m evals.run_eval --compare evals/baseline.json
```

```
==============================================================================
評測報告　模型：qwen3.8:27b　每題執行次數：1
==============================================================================
case                 pass   tokens    sec  calls  失敗原因
------------------------------------------------------------------------------
policy-warranty     100%     1086    7.6    2.0
calc-discount       100%     1155    9.6    2.0
multi-step          100%     1980   15.7    3.0
current-time        100%     1104    8.3    2.0
no-tool-needed      100%      525    7.2    1.0
not-in-kb           100%     1123   11.3    2.0
------------------------------------------------------------------------------
總計                100%     6973   59.7
==============================================================================
```

注意報告把**正確率與成本並列** —— 因為 `token_usage.py` 的 tracker 直接被 eval 重複使用。
一個「準確率不變但 token 翻倍」的改動，不是進步。

#### 真實踩坑：第一次跑出 67%，但模型其實全對

這份評測第一次執行的結果是這樣：

```
calc-discount         0%     1155   12.5    2.0  內容:缺少 ['3264']
current-time          0%     1104    9.3    2.0  內容:缺少 ['2026-08-21']
```

實際去看模型的回答：

```
[calc-discount]  '3 台單價 1280 元、打 85 折後，總金額為 **3,264 元**。'
[current-time]   '現在台北時間是 2026 年 8 月 21 日。'
```

**兩題都答對了**，是評分函式太死板：千分位逗號、中文日期寫法。
這正是 eval 開發的日常 —— **評測跑出紅字時，第一件事是確認評分器有沒有問題，而不是急著怪模型。**

解法是比對前先正規化，而且**兩邊套用同一個正規化**：

```python
def normalize(text):
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)   # 3,264 → 3264
    t = re.sub(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", ..., t)
    return re.sub(r"[\s*_`]", "", t)          # 空白與 markdown 記號
```

但**正規化要保守**：拿掉太多東西會製造假陽性，讓錯的答案也通過。
所以 `normalize()` 自己也有單元測試盯著（`tests/test_scorers.py`）：

```python
def test_不改動語意(self):
    assert normalize("3265") != normalize("3264")
```

#### 回歸比對：這才是 eval 真正的用途

有了基準線，就能量化任何改動的代價。實測「開啟 thinking 模式」：

```
與基準線比對：evals/baseline.json（基準模型：qwen3.8:27b）
------------------------------------------------------------------------------
policy-warranty   ! 變貴　通過率 100% → 100%　token 1086 → 1142 (+56)
calc-discount     ! 變貴　通過率 100% → 100%　token 1155 → 1239 (+84)
multi-step        ! 變貴　通過率 100% → 100%　token 1980 → 2107 (+127)
current-time      ! 變貴　通過率 100% → 100%　token 1104 → 1209 (+105)
no-tool-needed    ! 變貴　通過率 100% → 100%　token 525 → 572 (+47)
not-in-kb         ! 變貴　通過率 100% → 100%　token 1123 → 1222 (+99)
------------------------------------------------------------------------------
```

結論非常明確：**在這份資料集上，thinking 模式多花 7.4% token、多花 89% 時間
（59.7s → 112.8s），正確率完全沒有提升。** 這類任務就該關掉 thinking。

沒有 eval，這個決定只能靠感覺；有了 eval，它是一個有數字支撐的工程結論。
（反過來說，這也代表資料集需要再加入真正需要多步推理的難題，才能測出 thinking 的價值 ——
**eval 跑滿分通常不是好消息，而是代表題目太簡單。**）

---

### 11.5 該量哪些指標

| 指標 | 為什麼重要 | 本專案來源 |
|---|---|---|
| **正確率 / 通過率** | 基本盤，但單看它會漏掉下面所有問題 | `pass_rate` |
| **工具選擇準確率** | 答對但路徑錯 = 下次一定錯 | `score_tools` |
| **平均 LLM 呼叫次數** | Agent 成本與延遲的主要驅動因子 | `avg_llm_calls` |
| **平均 token** | 直接對應成本；換雲端模型時就是錢 | `avg_tokens` |
| **延遲** | 使用者體感；本地模型尤其明顯 | `avg_seconds` |
| **穩定度** | 同一題跑 N 次的通過率離散程度 | `--repeat` |

### 11.6 常見陷阱

| 陷阱 | 後果 | 做法 |
|---|---|---|
| 用單元測試斷言模型輸出 | flaky test，最後被整個 skip | 確定性部分才寫測試，其餘進 eval |
| 只看最終答案，不看軌跡 | 放過「矇對」的假陽性 | 加上 `expect_tools` |
| 叫 judge 打 1~10 分 | 分數飄移，無法比較 | 只做二元判斷 + 明確準則 |
| 資料集只有 happy path | 上線才發現濫用工具、編造答案 | 一定要有「不該用工具」與「查不到」案例 |
| 正規化過度 | 假陽性，錯的答案也通過 | 只處理格式，不碰語意，並為 `normalize()` 寫測試 |
| eval 綁死 CI 要求 100% | 模型天生會抖，CI 永遠紅 | 用通過率門檻（如 `>= 90%`）與 `--repeat` |
| 只比正確率不比成本 | 準確率持平但成本翻倍也看不出來 | 報告把 token 與延遲並列 |

### 11.7 接進 CI

`run_eval.py` 會以 exit code 回報結果，可以直接串進 pipeline：

```yaml
- run: pytest                              # 每次 push：快、不需要 Ollama
- run: python -m evals.run_eval --repeat 3 # 每晚或發版前：需要 Ollama
```

實務建議是**兩段式**：第 1、2 層擋在每次 commit（0.24 秒，沒有理由不跑）；
第 3 層跑得慢又需要模型，放在 nightly 或發版前，並且把基準線 JSON 納入版控，
讓每次分數變動都能在 code review 中被看見。

---

## 12. 疑難排解

| 問題 | 原因與處理 |
|---|---|
| `ConnectionError: [Errno 61] Connection refused` | Ollama 沒啟動。執行 `brew services start ollama` 或 `ollama serve`。 |
| `model requires version 0.32.12` | Ollama 版本太舊。`brew upgrade ollama` 後重啟服務。 |
| 回應極慢、`ollama ps` 顯示部分 CPU | 記憶體不足導致權重被分流到 CPU。關閉其他大型程式，或把 `num_ctx` 調小（KV cache 也吃記憶體）。 |
| Agent 一直重複呼叫工具 | 觀測報告會顯示 LLM 呼叫次數暴增。加強 system prompt、把工具 docstring 寫清楚，並保留 `recursion_limit` 當保險絲。 |
| token 數全是 0 | 檢查 `langchain-ollama` 版本；舊版不會填 `usage_metadata`。本專案已內建退回讀取 `prompt_eval_count` / `eval_count` 的保護。 |
| 第一次呼叫特別慢 | 模型載入時間。用 `OLLAMA_KEEP_ALIVE=-1` 讓模型常駐。 |
| 想釋放記憶體 | `ollama stop qwen3.8:27b` |

---

## 授權

Qwen3.8 採用 Apache License 2.0。本教學程式碼可自由使用。
