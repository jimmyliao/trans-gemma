# TranslateGemma Notebooks 使用指南

## 📓 可用的 Notebooks

### 1. `arxiv-reader.ipynb` ⭐ **推薦**

**功能**: 翻譯 arXiv 論文，雙語對照閱讀

**支援環境**:
- ✅ Google Colab (Free T4 GPU)
- ✅ GCP Custom Runtime (T4 GPU)
- ✅ 本地 Jupyter (CPU/GPU)

**特色**:
- 🤖 **自動偵測環境**：Colab / GCP / Local
- ⚡ **一個 Notebook 通吃**：不需要多個版本
- 📖 **雙語對照**：原文與翻譯並列
- 💾 **互動式 HTML**：可下載離線瀏覽

**使用方式**:

#### 在 Google Colab (推薦新手)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimmyliao/trans-gemma/blob/main/arxiv-reader.ipynb)

1. 點擊上方按鈕開啟 Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run All

#### 在 GCP Custom Runtime

1. 連線到你的 Jupyter: `http://your-ip:8888/?token=xxx`
2. 開啟 `arxiv-reader.ipynb`
3. **選擇 Kernel**: Python 3.10 (trans-gemma)
4. Run All

#### 在本地 Jupyter

1. 安裝 PyTorch: `pip install torch`
2. 開啟 `arxiv-reader.ipynb`
3. Run All

---

### 2. `document-translator-colab.ipynb`

**功能**: 通用文件翻譯（支援多種格式）

**支援格式**: PDF, DOCX, TXT, Markdown

**環境**: Google Colab only

---

## 🚀 快速開始（推薦路徑）

### 第一次使用 TranslateGemma？

1. **使用 Google Colab** + `arxiv-reader.ipynb`
   - 完全免費
   - T4 GPU 加速
   - 無需設定環境

2. **取得 HuggingFace Token**
   - https://huggingface.co/settings/tokens
   - 接受模型: https://huggingface.co/google/gemma-2-2b-it

3. **Run All**
   - 自動下載模型（首次 ~4GB）
   - 翻譯完成後下載 HTML

### 已經有 GCP T4 Custom Runtime？

1. **使用 VSCode 連線**到你的 Jupyter
2. 開啟 `arxiv-reader.ipynb`
3. **重要**: 選擇 "Python 3.10 (trans-gemma)" kernel
4. Run All

---

## 📊 效能比較

| 環境 | 每頁翻譯時間 | 成本 |
|------|-------------|------|
| **Colab T4 (Free)** | 20-25 秒 | 免費 |
| **GCP T4 Custom** | 20-25 秒 | ~$0.08/45分鐘 |
| **M1 Mac** | 30-40 秒 | 本地 |
| **CPU only** | 15-20 分鐘 | - |

---

## ❓ 常見問題

### Q: 三個 notebook 有什麼差別？

A: **現在只有一個 `arxiv-reader.ipynb`**！
   - 舊版有 3 個分別給 Colab/GCP/Local
   - 新版自動偵測環境，一個 notebook 通吃

### Q: Colab 和 GCP Custom Runtime 哪個好？

A: 看需求：
   - **Colab**: 免費、簡單、適合新手
   - **GCP**: 持續運行、自訂環境、適合進階使用

### Q: 為什麼 GCP 要選 Python 3.10 kernel？

A: trans-gemma 需要 Python ≥3.10，但 GCP Deep Learning VM 預設是 3.9。
   我們在 startup script 中已建立 py310 環境。

### Q: 第一次執行很慢？

A: 正常！首次需下載 Gemma 2-2B 模型（~4GB）。
   下載後會快取，之後執行就快了。

### Q: 可以翻譯中文論文嗎？

A: 可以！修改 Step 5:
   ```python
   SOURCE_LANG = "zh-TW"  # 或 "zh-CN"
   TARGET_LANG = "en"
   ```

---

## 🛠️ 進階使用

### 自訂翻譯頁碼

在 Step 5 修改：

```python
SECTIONS = {
    "abstract": (1, 1),      # 摘要：第 1 頁
    "introduction": (2, 4),  # 介紹：2-4 頁
    "method": (5, 10),       # 方法：5-10 頁
}
```

### 批次翻譯多篇論文

```python
# 在最後一個 cell 加入
papers = [
    "2403.08295",  # Gemma
    "2312.11805",  # Gemini
    "2601.09012",  # TranslateGemma
]

for arxiv_id in papers:
    ARXIV_ID = arxiv_id
    # ... 執行翻譯邏輯
```

### 整合到你的專案

```python
from trans_gemma import TranslateGemma

translator = TranslateGemma(model_id="google/gemma-2-2b-it")
result = translator.translate("Hello world", target_lang="zh-TW")
print(result)
```

---

## 📝 回報問題

遇到問題？請到 [GitHub Issues](https://github.com/jimmyliao/trans-gemma/issues) 回報

---

**Made with ❤️ by Jimmy Liao**
