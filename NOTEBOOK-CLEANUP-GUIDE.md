# arxiv-reader.ipynb 清理指南

**目標**: 保留核心功能，標記/註解測試用 cells

---

## 📋 Cells 分類

### ✅ 保留（核心功能）

| Cell | 類型 | 說明 |
|------|------|------|
| 0 | Markdown | 標題、Open in Colab、作者介紹 |
| 1 | Markdown | Step 0 說明 |
| 2 | Code | 環境偵測 |
| 5 | Markdown | Step 1 說明 |
| 6 | Code | 套件安裝（根據環境） |
| 7 | Markdown | Step 2 說明 |
| 8 | Code | Clone trans-gemma |
| 9 | Markdown | Step 3 說明（HF 認證） |
| 15 | Code | HF 認證（多方法） |
| 16 | Markdown | Step 4 說明 |
| 17 | Code | GPU 檢查 |
| 18 | Markdown | Step 5 說明 |
| 19 | Code | 翻譯參數設定 |
| 20 | Markdown | Step 6 說明 |
| 22 | Code | 載入 TranslateGemma 模型 |
| 23 | Markdown | Step 7 說明 |
| 24 | Code | 下載 PDF & 翻譯 |
| 25 | Markdown | Step 8 說明 |
| 26 | Code | 顯示翻譯結果 |
| 27 | Markdown | Step 9 說明 |
| 28 | Code | 生成 HTML |
| 29 | Markdown | 完成頁面 |

---

### ⚠️ 需要處理（測試/驗證用）

| Cell | 類型 | 內容 | 建議處理 |
|------|------|------|---------|
| **3** | Code | `import sys; !{sys.executable} -m pip install...` | 🔄 改為 Markdown，標題「驗證：直接安裝到當前環境」 |
| **4** | Code | `from huggingface_hub import login...` | ❌ 刪除（重複，Cell 15 已有） |
| **10** | Code | `!echo "=== 安裝必要套件 ==="; !pip install...` | ❌ 刪除（測試遺留） |
| **11** | Code | `!echo ""; !python3 -c "import huggingface_hub..."` | ❌ 刪除（測試遺留） |
| **12** | Code | `!python3 -m pip install...` | ❌ 刪除（測試遺留） |
| **13** | Code | `!echo ""; !python3 -c "import..."` | ❌ 刪除（測試遺留） |
| **14** | Code | `!which python3; !which pip...` | 🔄 改為 Markdown，標題「驗證：檢查 Python 路徑」 |
| **21** | Code | `inspect.signature(TransformersBackend.__init__)` | 🔄 改為 Markdown，標題「驗證：檢查 Backend 參數」 |

---

## 🔧 具體清理步驟

### Step 1: 保留 Cell 3（改為可選驗證）

**原內容**（Code cell）:
```python
import sys
print(f"當前 Python: {sys.executable}")
!{sys.executable} -m pip install huggingface_hub transformers accelerate sentencepiece protobuf pymupdf pillow tqdm ipywidgets -q
print("\n✅ 安裝完成！")
```

**改為**（Markdown cell）:
```markdown
### 🔍 驗證：直接安裝到當前環境（可選）

如果 Step 1 安裝失敗，可以執行以下 cell 直接安裝到當前 Python 環境：

\`\`\`python
import sys
print(f"當前 Python: {sys.executable}")
!{sys.executable} -m pip install huggingface_hub transformers accelerate sentencepiece protobuf pymupdf pillow tqdm ipywidgets -q
print("\n✅ 安裝完成！")
\`\`\`

> **注意**: 正常情況下 Step 1 即可，此為備用方案。
```

---

### Step 2: 刪除 Cell 4

**理由**: 與 Cell 15 重複，Cell 15 更完整。

---

### Step 3: 刪除 Cells 10-13

**理由**: 測試遺留的重複安裝指令。

---

### Step 4: 保留 Cell 14（改為可選驗證）

**原內容**（Code cell）:
```python
!which python3
!which pip
!python3 -c "import sys; print(sys.executable)"
```

**改為**（Markdown cell）:
```markdown
### 🔍 驗證：檢查 Python 和 pip 路徑（可選）

如果懷疑環境不一致，可執行以下 cell 檢查：

\`\`\`python
!which python3
!which pip
!python3 -c "import sys; print(sys.executable)"
\`\`\`

**預期輸出**（GCP py310 環境）:
\`\`\`
/opt/conda/envs/py310/bin/python3
/opt/conda/envs/py310/bin/pip
/opt/conda/envs/py310/bin/python
\`\`\`
```

---

### Step 5: 保留 Cell 21（改為可選驗證）

**原內容**（Code cell）:
```python
import sys
sys.path.insert(0, '/root/trans-gemma/examples')
sys.path.insert(0, '/root/trans-gemma/examples/backends')

from transformers_backend import TransformersBackend
import inspect

print("TransformersBackend.__init__ 參數：")
print(inspect.signature(TransformersBackend.__init__))
```

**改為**（Markdown cell）:
```markdown
### 🔍 驗證：檢查 TransformersBackend 初始化參數（可選）

如果想了解 Backend 的正確用法，可執行：

\`\`\`python
import sys
sys.path.insert(0, '/root/trans-gemma/examples')
sys.path.insert(0, '/root/trans-gemma/examples/backends')

from transformers_backend import TransformersBackend
import inspect

print("TransformersBackend.__init__ 參數：")
print(inspect.signature(TransformersBackend.__init__))
\`\`\`

**預期輸出**:
\`\`\`
TransformersBackend.__init__ 參數：
(self)
\`\`\`

> **提示**: `__init__()` 不接受參數，需先創建實例再 `load_model()`。
```

---

## 📝 改進建議

### 1. 改善 Cell 0（作者介紹）

**參考原版 arxiv-reader-colab.ipynb 的風格**，加強：

- ✅ 功能特色（雙語對照、術語表、互動 HTML）
- ✅ 適合對象（研究生、工程師、英文學習者）
- ✅ 作者資訊（Jimmy Liao, AI GDE/MVP, CTO）

**建議新內容**:
```markdown
# arXiv Bilingual Reader - TranslateGemma

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimmyliao/trans-gemma/blob/main/arxiv-reader.ipynb)

**📖 雙語對照閱讀 arXiv 論文，提升英文學術寫作能力**

---

## ✨ 功能特色

- 🎯 **雙語並排**: 原文與翻譯並列，方便對照學習
- 📚 **章節分組**: 依 Abstract、Methods、Results 等結構化組織
- 💾 **互動式 HTML**: 生成可離線閱讀的網頁，支援鍵盤導航
- 🔤 **術語表**: 自動提取專業術語及其翻譯
- 🇹🇼 **繁體優化**: 針對台灣繁體中文優化

---

## 🎯 適合對象

- ✅ **研究生**: 閱讀文獻、準備論文寫作
- ✅ **工程師**: 追蹤最新技術、理解前沿研究
- ✅ **英文學習者**: 學習學術英文表達方式

---

## 🚀 支援環境

此 notebook 會**自動偵測**執行環境並調整設定：

- ✅ **Google Colab** (Free T4 GPU) - 推薦新手
- ✅ **GCP Custom Runtime** (T4 GPU) - 進階用戶
- ✅ **本地 Jupyter** (CPU/GPU) - 有 GPU 設備

---

## ⚡ 快速開始

### Google Colab (推薦)
1. 點擊上方 "Open In Colab" 按鈕
2. Runtime → Change runtime type → T4 GPU
3. 按順序執行所有 cells

### 預期時間
- 首次執行：~10 分鐘（含下載模型 8GB）
- 之後執行：~5 分鐘（模型已快取）
- 翻譯速度：~3 分鐘/頁

---

## 👤 作者

**Jimmy Liao** ([GitHub](https://github.com/jimmyliao))
- Google AI GDE (Generative AI)
- Microsoft MVP (AI)
- AI Startup CTO
- Blog: https://jimmyliao.dev

---

**License**: MIT | **Model**: TranslateGemma 4B | **Source**: [GitHub](https://github.com/jimmyliao/trans-gemma)
```

---

### 2. 改善 Cell 28（下載功能）

**確保 Colab 自動下載 HTML**:

```python
if SAVE_HTML:
    # ... (生成 HTML 的程式碼) ...

    filename = f"arxiv_{ARXIV_ID}_{SOURCE_LANG}-{TARGET_LANG}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n💾 HTML saved: {filename}")

    # Auto-download in Colab
    if ENV == 'colab':
        from google.colab import files
        print(f"📥 Downloading {filename}...")
        files.download(filename)
        print("✅ Downloaded! Check your Downloads folder.")
    else:
        print(f"📂 File location: {os.path.abspath(filename)}")
```

---

### 3. 新增 Cell 29（完成頁面）

**參考原版的結束頁面**:

```markdown
## 🎉 翻譯完成！

### 下一步

1. **翻譯其他論文**:
   - 修改 Step 5 的 `ARXIV_ID`
   - 例如: `"2312.11805"` (Gemini Paper)

2. **翻譯更多章節**:
   \`\`\`python
   SECTIONS = {
       "abstract": (1, 1),
       "intro": (2, 4),
       "method": (5, 10),
   }
   \`\`\`

3. **查看 HTML 輸出**:
   - 在瀏覽器開啟下載的 HTML 檔案
   - 使用 ← → 鍵導航
   - 享受雙語對照閱讀！

---

### 📚 延伸資源

- [TranslateGemma Paper](https://arxiv.org/abs/2601.09012)
- [Gemma Model Card](https://huggingface.co/google/translategemma-4b-it)
- [GitHub Repository](https://github.com/jimmyliao/trans-gemma)
- [使用指南](https://github.com/jimmyliao/trans-gemma/blob/main/NOTEBOOK-GUIDE.md)

---

### 🤝 回饋與貢獻

遇到問題？有建議？

- 🐛 [回報 Issue](https://github.com/jimmyliao/trans-gemma/issues)
- ⭐ [給個 Star](https://github.com/jimmyliao/trans-gemma)
- 💬 [加入討論](https://github.com/jimmyliao/trans-gemma/discussions)

---

**Made with ❤️ by Jimmy Liao**
```

---

## ✅ 清理後的 Notebook 結構

```
[0] Markdown: 標題 + 作者介紹 + 功能特色
[1] Markdown: Step 0 說明
[2] Code: 環境偵測
[3] Markdown: 🔍 驗證：直接安裝（可選）
[5] Markdown: Step 1 說明
[6] Code: 套件安裝
[7] Markdown: Step 2 說明
[8] Code: Clone trans-gemma
[9] Markdown: Step 3 說明
[14] Markdown: 🔍 驗證：檢查路徑（可選）
[15] Code: HF 認證
[16] Markdown: Step 4 說明
[17] Code: GPU 檢查
[18] Markdown: Step 5 說明
[19] Code: 翻譯參數設定
[20] Markdown: Step 6 說明
[21] Markdown: 🔍 驗證：檢查 Backend（可選）
[22] Code: 載入模型
[23] Markdown: Step 7 說明
[24] Code: 下載 PDF & 翻譯
[25] Markdown: Step 8 說明
[26] Code: 顯示結果
[27] Markdown: Step 9 說明
[28] Code: 生成 HTML + 自動下載
[29] Markdown: 完成頁面 + 延伸資源
```

---

## 🎯 清理完成檢查表

- [ ] Cell 0: 更新為完整作者介紹
- [ ] Cell 3: 改為 Markdown 驗證說明
- [ ] Cell 4: 刪除
- [ ] Cells 10-13: 刪除
- [ ] Cell 14: 改為 Markdown 驗證說明
- [ ] Cell 21: 改為 Markdown 驗證說明
- [ ] Cell 28: 確保 Colab 自動下載
- [ ] Cell 29: 新增完成頁面
- [ ] 測試: 在 Colab 完整執行一次
- [ ] 文檔: 更新 README 連結到新 notebook

---

**下一步**: 根據此指南手動編輯 notebook 或建立清理腳本。
