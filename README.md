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
- [11. 疑難排解](#11-疑難排解)

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
├── requirements.txt
├── agent.py           # Agent 本體：模型設定、工具、執行進入點
├── token_usage.py     # 可觀測性：token / 延遲 / 工具呼叫的 callback handler
└── runs/usage.jsonl   # 執行後自動產生的事件紀錄（已加入 .gitignore）
```

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

## 11. 疑難排解

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
