# TranslateGemma 範例與測試工具

這個目錄包含 TranslateGemma 的使用範例和測試工具。

## 📋 檔案說明

### 1. verify-hf-token.py
驗證 Hugging Face token 和模型存取權限的獨立腳本。

**使用方式**：
```bash
# 方式 A: 使用環境變數
export HF_TOKEN="hf_xxxxx"
python examples/verify-hf-token.py

# 方式 B: 作為參數傳入
python examples/verify-hf-token.py hf_xxxxx
```

**測試項目**：
- ✅ HF 認證
- ✅ API 存取
- ✅ TranslateGemma 模型存取權限
- ✅ Tokenizer 載入

### 2. local-test.py
完整的本地測試流程，使用 `.env` 檔案管理配置。

**使用方式**：
```bash
# 1. 建立 .env 檔案
cp .env.example .env

# 2. 編輯 .env 並填入你的 HF_TOKEN
# HF_TOKEN=hf_xxxxx

# 3. 執行測試
python examples/local-test.py
```

**測試項目**：
- ✅ .env 檔案載入
- ✅ HF_TOKEN 驗證
- ✅ 模型存取測試
- ✅ 翻譯功能測試（可選）

### 3. translategemma-fix.py
TranslateGemma 正確使用範例，展示正確的 chat template 格式。

**使用方式**：
```bash
export HF_TOKEN="hf_xxxxx"
python examples/translategemma-fix.py
```

**特色**：
- ✅ 正確的 TranslateGemma message 格式
- ✅ 語言代碼映射（ISO 639-3）
- ✅ 包含 source_lang_code 和 target_lang_code
- ✅ 多種語言翻譯範例

### 4. simple-translation.py
Cloud Run API 客戶端範例（需要先部署 API）。

**使用方式**：
```bash
# 更新 API_URL 為你的 Cloud Run 服務 URL
python examples/simple-translation.py
```

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
                "source_lang_code": "eng",
                "target_lang_code": "zho_Hant"
            }
        ]
    }
]
```

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

## 🔐 安全最佳實踐

1. **不要**將 HF_TOKEN 硬編碼在程式碼中
2. **不要**將 `.env` 檔案 commit 到 Git
3. **不要**在公開場所分享你的 token
4. **使用**環境變數或 Colab Secrets 儲存 token
5. **定期**更新和輪換你的 tokens

## 🎯 語言代碼對照表

TranslateGemma 使用 **ISO 639-1（兩碼）** 標準，中文使用 CLDR 格式：

| 語言名稱 | 代碼 | 說明 |
|---------|------|------|
| English | en | |
| Traditional Chinese (Taiwan) | zh-TW | 繁體中文（台灣） |
| Simplified Chinese (China) | zh-CN | 簡體中文（中國） |
| Japanese | ja | |
| Korean | ko | |
| French | fr | |
| German | de | |
| Spanish | es | |
| Italian | it | |
| Portuguese | pt | |
| Russian | ru | |
| Arabic | ar | |
| Hindi | hi | |
| Vietnamese | vi | |
| Thai | th | |
| Indonesian | id | |
| Hebrew | he | |
| Persian | fa | |

**重要**：使用兩碼格式（如 `en`），不是三碼格式（如 `eng`）

## 🔗 相關資源

- [TranslateGemma 官方頁面](https://huggingface.co/google/translategemma-4b-it)
- [Hugging Face Token 設定](https://huggingface.co/settings/tokens)
- [專案 GitHub Repository](https://github.com/jimmyliao/trans-gemma)
- [Hugging Face 存取設定指南](../docs/huggingface-access.md)

## 🆘 常見問題

### Q: 為什麼會出現 TemplateError？

A: TranslateGemma 需要特殊的 message 格式，包含 `source_lang_code` 和 `target_lang_code`。請參考 `translategemma-fix.py` 中的正確格式。

### Q: 如何取得 HF_TOKEN？

A: 前往 https://huggingface.co/settings/tokens 建立新 token，選擇 Read 權限即可。

### Q: Token 驗證通過但無法載入模型？

A: 確認你已經在 Hugging Face 申請 TranslateGemma 的存取權限：
https://huggingface.co/google/translategemma-4b-it

### Q: 本地測試需要 GPU 嗎？

A:
- `verify-hf-token.py`: 不需要（只測試 tokenizer）
- `local-test.py`: 翻譯測試需要 GPU，但可以跳過
- `translategemma-fix.py`: 需要 GPU 來執行完整測試

建議在 Google Colab（免費 T4 GPU）或有 GPU 的環境中進行完整測試。
