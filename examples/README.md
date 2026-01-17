# TranslateGemma 範例與測試工具

這個目錄包含 TranslateGemma 的使用範例和測試工具。

## 📁 目錄結構

```
examples/
├── translate.py              # ⭐ 統一翻譯工具（全新！）
├── backends/                 # 翻譯後端實作
│   ├── __init__.py
│   ├── base.py              # 抽象基礎類別
│   ├── transformers_backend.py  # Hugging Face Transformers
│   ├── ollama_backend.py     # Ollama (Metal 優化)
│   └── mlx_backend.py        # MLX (Apple Silicon 優化)
├── verify-hf-token.py        # 驗證 Hugging Face token
├── simple-translation.py     # Cloud Run API 客戶端
├── local-test.py             # 舊版：完整 transformers 測試
└── translategemma-fix.py     # 舊版：TranslateGemma 修正測試
```

## 🚀 統一翻譯工具 (translate.py)

**推薦**的本地使用 TranslateGemma 方式。

### 特色

- **多種後端**：可選擇 transformers、ollama 或 mlx
- **兩種模式**：單次翻譯或互動式 REPL
- **易於使用**：簡單的命令列介面
- **優化**：每個後端都針對其目標平台優化

### 快速開始

```bash
# 單次翻譯（Ollama - M1 上最快）
./run-examples.sh translate --text "Hello, world!" --backend ollama

# 互動模式
./run-examples.sh translate --mode interactive --backend ollama

# 使用 transformers 後端
./run-examples.sh translate --text "Hello!" --backend transformers --target ja

# 使用 MLX 後端（Apple Silicon 優化）
./run-examples.sh translate --text "Hello!" --backend mlx
```

### 後端比較

| 後端 | 速度 | 安裝 | 狀態 | 最適合 |
|------|------|------|------|--------|
| **ollama** | ⚡⚡⚡ 快 | 一行指令 | ✅ 推薦 | M1/M2/M3 Mac，所有用途 |
| **transformers** | ⚠️ 慢 | 預設 | ✅ 可用 | Colab/CUDA GPU，研究用途 |
| **mlx** | - | - | ❌ 不可用 | 等待 MLX 版本模型 |

**注意**: MLX 後端暫時不可用，因為 TranslateGemma 尚未有 MLX 優化版本。

### 模式

#### 1. 單次模式（One-shot，預設）

翻譯單一文字後退出。

```bash
# 基本使用
./run-examples.sh translate --text "早安！"

# 指定目標語言
./run-examples.sh translate --text "Hello!" --target ja

# 指定來源語言
./run-examples.sh translate --text "Bonjour" --source fr --target en
```

#### 2. 互動模式（Interactive）

REPL 模式持續翻譯。

```bash
./run-examples.sh translate --mode interactive
```

**互動命令：**

- `:target <code>` - 更改目標語言
- `:source <code>` - 更改來源語言
- `:info` - 顯示後端資訊
- `:quit`, `:exit` - 退出

**範例會話：**

```
[en → zh-TW] Hello, world!
→ 你好，世界！
  (0.82s, 3.7 tok/s)

[en → zh-TW] :target ja
ℹ️  Target language changed to: ja

[en → ja] Good morning!
→ おはようございます！
  (1.2s, 5.1 tok/s)

[en → ja] :quit

Statistics:
  Total translations: 2
  Average time: 1.01s

✅ Goodbye!
```

### 環境變數

- `BACKEND`: 預設後端 (`transformers`, `ollama`, `mlx`)
- `FORCE_DEVICE`: transformers 的設備 (`cpu`, `mps`, `auto`)
- `NO_MEM_LIMIT`: 停用 transformers 記憶體限制 (`0`, `1`)

### 範例

```bash
# Ollama（推薦給 M1/M2/M3 Mac）
./run-examples.sh translate --text "Hello!" --backend ollama

# Transformers 使用 CPU
FORCE_DEVICE=cpu ./run-examples.sh translate --text "Hello!" --backend transformers

# 互動模式
./run-examples.sh translate --mode interactive --backend ollama
```

## 🆚 效能比較（M1 Mac）

基於實際測試：

| 後端 | 模型載入 | 首次翻譯 | 記憶體 | 狀態 |
|------|----------|----------|--------|------|
| Ollama | 0.04s | 0.8s | 3.3 GB | ✅ 推薦 |
| Transformers (MPS 8GB) | 8.8s | 94.8s ⚠️ | 8.7 GB | ⚠️ 太慢 |
| Transformers (CPU) | ~15s | ~5min ⚠️ | 10 GB | ⚠️ 非常慢 |

**結論**：**在 M1 Mac 上使用 Ollama**。Transformers 太慢不實用。

## 📝 其他範例

### verify-hf-token.py

驗證你的 Hugging Face token 和模型存取權限。

```bash
./run-examples.sh verify-hf-token
```

### simple-translation.py

Cloud Run API 客戶端範例（需要已部署的服務）。

```bash
./run-examples.sh simple-translation
```

### 舊版腳本

**local-test.py** 和 **translategemma-fix.py** 是測試 transformers 後端的舊版腳本。新開發請使用 `translate.py`。

## 🏗️ 後端架構

所有後端都實作 `TranslationBackend` 介面：

```python
class TranslationBackend(ABC):
    def load_model(self, **kwargs) -> Dict[str, Any]:
        """載入模型並回傳元資料"""
        pass

    def translate(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """翻譯文字"""
        pass

    def get_backend_info(self) -> Dict[str, str]:
        """取得後端資訊"""
        pass

    def cleanup(self):
        """選擇性的清理方法"""
        pass
```

### 新增自訂後端

1. 建立 `backends/your_backend.py`
2. 實作 `TranslationBackend` 介面
3. 在 `backends/__init__.py` 中註冊
4. 使用 `translate.py --backend your_backend` 測試

## 🚨 重要：TranslateGemma Chat Template 格式

TranslateGemma 使用**特殊的 chat template 格式**，與標準格式不同。

### ❌ 錯誤格式（會導致 TemplateError）：
```python
messages = [
    {
        "role": "user",
        "content": "Translate this to Chinese: Hello"
    }
]
```

### ✅ 正確格式：
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Hello",
                "source_lang_code": "en",
                "target_lang_code": "zh-TW"
            }
        ]
    }
]
```

## 🎯 語言代碼對照表

使用 **ISO 639-1（兩碼）** 標準：

- `en` - English
- `zh-TW` - Traditional Chinese (Taiwan) 繁體中文（台灣）
- `zh-CN` - Simplified Chinese 簡體中文
- `ja` - Japanese 日文
- `ko` - Korean 韓文
- `es` - Spanish 西班牙文
- `fr` - French 法文
- `de` - German 德文
- 等等

**重要**：使用兩碼格式（如 `en`），不是三碼格式（如 `eng`）

## 🐛 疑難排解

### Ollama: Model not found

```bash
ollama pull translategemma
```

### MLX: Backend not available

MLX 後端目前不可用，因為 TranslateGemma 尚未有 MLX 優化版本。請使用 Ollama 後端：

```bash
./run-examples.sh translate --text "Hello!" --backend ollama
```

### Transformers: Invalid buffer size (M1 Mac)

在 M1 Mac 上請改用 Ollama。Transformers 在 MPS 上有記憶體管理問題，效能也較差。

## 📝 .env 檔案設定

複製 `.env.example` 到 `.env` 並填入你的配置：

```bash
# Hugging Face Access Token
HF_TOKEN=hf_xxxxx

# GCP Project (for Cloud Run deployment)
PROJECT_ID=your-gcp-project-id
REGION=us-central1
SERVICE_NAME=translategemma-4b

# Model Configuration
MODEL_ID=google/translategemma-4b-it
```

**重要**：
- ⚠️ `.env` 檔案包含敏感資訊，**不要** commit 到 Git
- ✅ `.env` 已經在 `.gitignore` 中
- ✅ 使用 `.env.example` 作為範本

## 🔗 相關資源

- [TranslateGemma 官方頁面](https://huggingface.co/google/translategemma-4b-it)
- [Ollama 官方網站](https://ollama.ai/)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [專案 GitHub Repository](https://github.com/jimmyliao/trans-gemma)

## 🆘 常見問題

### Q: 應該使用哪個後端？

A:
- **M1/M2/M3 Mac**: 使用 Ollama（推薦）
- **Google Colab / NVIDIA GPU**: 使用 Transformers
- **CPU only**: 使用 Ollama（如已安裝）或 Transformers CPU 模式

### Q: Ollama 會使用 GPU 嗎？

A: 是的，Ollama 在 M1 上自動使用 Metal (GPU) 加速。

### Q: 為什麼 Transformers 這麼慢？

A: Transformers 在 M1 的 MPS 上支援不佳，有記憶體管理問題。推薦使用針對 M1 優化的 Ollama 或 MLX。

### Q: 如何取得 HF_TOKEN？

A: 前往 https://huggingface.co/settings/tokens 建立新 token，選擇 Read 權限即可。
