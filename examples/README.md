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
| **ollama** | ⚡⚡⚡ 快 (~30 tok/s) | 一行指令 | ✅ 推薦 | M1/M2/M3 Mac，所有用途 |
| **transformers** | ⚠️ 慢 (~1.5 tok/s MPS) | 預設 | ✅ 可用 | Colab/CUDA GPU，研究用途 |
| **mlx** | ⚠️ 很慢 (~0.2 tok/s) | `uv pip install mlx-lm` | ⚠️ 實驗性 | 測試用途 |

**注意**:
- **Ollama** 是 M1 Mac 上的最佳選擇（速度快、穩定、易用）
- **MLX** 後端使用 4-bit 量化模型，翻譯品質良好但速度較慢，仍在實驗階段

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

#### 3. PDF 模式（PDF）

翻譯 PDF 文件，支援兩種模式：**文字模式**（預設）和**圖片模式**（實驗性）。

##### 3A. 文字模式（預設）

提取 PDF 文字內容並翻譯。

```bash
# 翻譯整份 PDF（TranslateGemma 技術報告）
./run-examples.sh translate --mode pdf --file examples/2601.09012v2.pdf

# 翻譯特定頁面範圍
./run-examples.sh translate --mode pdf --file examples/2601.09012v2.pdf --start-page 1 --end-page 3

# 使用不同後端翻譯 PDF
./run-examples.sh translate --mode pdf --file examples/2601.09012v2.pdf --backend transformers
```

**功能：**
- ✅ 快速（使用現有 backend）
- ✅ 自動提取 PDF 文字內容
- ✅ 逐頁翻譯並顯示進度
- ✅ 支援指定頁碼範圍
- ⚠️ 失去格式資訊（僅純文字）

##### 3B. 圖片模式（實驗性 - 多模態 TranslateGemma）

將 PDF 頁面轉換為圖片，使用 TranslateGemma 的多模態能力翻譯。

```bash
# 使用圖片模式翻譯 PDF（預設 DPI=96，速度優化）
./run-examples.sh translate --mode pdf --file examples/2601.09012v2.pdf --pdf-as-image

# 僅翻譯特定頁面（圖片模式）
./run-examples.sh translate --mode pdf --file examples/2601.09012v2.pdf --start-page 1 --end-page 1 --pdf-as-image

# 調整 DPI（更低 = 更快，更高 = 更清晰）
./run-examples.sh translate --mode pdf --file examples/2601.09012v2.pdf --start-page 1 --end-page 1 --pdf-as-image --dpi 72
```

**功能：**
- ✅ 保留視覺上下文（佈局、表格、圖表）
- ✅ 使用 TranslateGemma 多模態能力（image-text-to-text）
- ✅ 模型能"看到"整個頁面
- ✅ **自動圖片縮放** - 自動縮放到 896×896（TranslateGemma 最佳輸入）
- ✅ **Streaming 生成** - 即時顯示翻譯進度
- ✅ **Early Stopping** - 自動偵測重複並提早停止
- ✅ **可調 DPI** - 平衡速度與品質（預設 96）
- ✅ **強化語言約束** - 防止語言混雜（繁簡中文、韓文等）
- ✅ **優化生成參數** - temperature=0.3, top_p=0.85（更確定性輸出）
- ⚠️ 較慢（需載入多模態模型）
- ⚠️ 實驗性功能

**模式比較：**

| 特性 | 文字模式 | 圖片模式 |
|------|----------|----------|
| 速度 | ⚡ 快 | ⚠️ 慢（但已優化） |
| 格式保留 | ❌ 失去 | ✅ 保留上下文 |
| Backend | 任意 | transformers-multimodal |
| 記憶體需求 | 依 backend | ~10 GB |
| 圖片處理 | N/A | 自動縮放到 896×896 |
| DPI 設定 | N/A | 可調（預設 96） |
| 狀態 | ✅ 穩定 | ⚠️ 實驗性 |

**建議：**
- 一般翻譯：使用**文字模式** + Ollama（最快）
- 保留格式上下文：使用**圖片模式**（實驗性）
- 速度優化：使用較低 DPI（`--dpi 72` 或 `--dpi 96`）

**輸出範例：**

```
TranslateGemma - PDF Translation
Backend: ollama
File: examples/2601.09012v2.pdf

Extracting text from PDF...
✅ Extracted text from 12 page(s)

Loading ollama backend...
✅ Model loaded
   Device: metal

Page 1:
Translating 2143 characters...
[翻譯內容...]

Time: 3.2s, Tokens: 456, Speed: 142.5 tok/s
────────────────────────────────────────────────────────────────────────────────

[... 更多頁面 ...]

Summary:
  Pages translated: 12
  Total time: 38.4s
  Total tokens: 5234
  Average speed: 136.3 tok/s
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
| Ollama | 0.04s | 0.8s (~30 tok/s) | 3.3 GB | ✅ 推薦 |
| MLX (4-bit) | 3.5s | 15s (~0.2 tok/s) | ~6 GB | ⚠️ 實驗性 |
| Transformers (MPS 8GB) | 8.8s | 94.8s (~1.5 tok/s) ⚠️ | 8.7 GB | ⚠️ 太慢 |
| Transformers (CPU) | ~15s | ~5min ⚠️ | 10 GB | ⚠️ 非常慢 |

**結論**：
- **推薦**：**在 M1 Mac 上使用 Ollama**（速度快、穩定、易用）
- **實驗性**：MLX 可用但速度慢，翻譯品質良好
- **不推薦**：Transformers 太慢不實用

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
