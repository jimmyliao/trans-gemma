# arXiv 雙語閱讀器：用 TranslateGemma 打造學術論文的最佳閱讀體驗

**副標**：從看不懂到學得會 — 如何用 AI 翻譯同時提升英文學術寫作能力

---

## 痛點：讀英文論文的三大困境

作為研究者或工程師，您一定遇過這些情況：

### 困境 1：專業術語看不懂，翻譯軟體也幫不上忙

```
原文："We employ a two-stage fine-tuning process with
       MetricX-QE and AutoMQM for reinforcement learning..."

Google 翻譯："我們採用兩階段微調過程，並使用 MetricX-QE
            和 AutoMQM 進行強化學習..."
```

**問題**：翻譯是翻譯了，但 MetricX-QE、AutoMQM 是什麼？reinforcement learning 在這裡的脈絡是什麼意思？

### 困境 2：想學英文學術寫作，但缺乏對照範本

看完中文翻譯，理解了內容。但下次自己寫論文時：
- ❌ "evaluation results" 到底怎麼用？
- ❌ "demonstrate the effectiveness" 的句型記不起來
- ❌ 專業術語的英文表達方式想不出來

### 困境 3：PDF 工具翻譯破壞排版，圖表消失

市面上的 PDF 翻譯工具：
- ❌ 翻譯後版面跑掉
- ❌ 圖表和文字的關係斷裂
- ❌ 數學公式變亂碼

**我需要的是**：能保留原文、對照學習、又不破壞閱讀體驗的工具。

---

## 解決方案：TranslateGemma + arXiv 雙語閱讀器

### 什麼是 TranslateGemma？

Google 在 2026 年 1 月發布的開放式機器翻譯模型家族，基於 Gemma 3 架構：

- 🌍 **支援 38 種語言**（包含繁體中文）
- 🎯 **專門優化翻譯品質**：透過 MetricX-QE 和 AutoMQM 強化學習
- 📖 **開源可用**：可在 HuggingFace 下載，T4 GPU 即可運行
- 🏆 **SOTA 級表現**：在 WMT25 測試集上表現優異

**技術報告**：[arXiv:2601.09012](https://arxiv.org/abs/2601.09012)

### 為什麼不用 ChatGPT/Claude 翻譯就好？

TranslateGemma 的優勢：

| 特性 | TranslateGemma | ChatGPT/Claude |
|------|----------------|----------------|
| **專業優化** | ✅ 專門為翻譯訓練 | ⚠️ 通用模型 |
| **術語一致性** | ✅ 保持原文術語 | ❌ 可能過度意譯 |
| **成本** | ✅ 免費（自建）| 💰 API 付費 |
| **批次處理** | ✅ Colab GPU 免費 | 💰 大量翻譯成本高 |
| **離線使用** | ✅ 下載模型即可 | ❌ 需網路連線 |

---

## 實際成果：看看翻譯效果

### 翻譯品質範例

**論文**：TranslateGemma Technical Report (arXiv:2601.09012v2)

**原文 (Abstract)**：
```
We present TranslateGemma, a suite of open machine translation
models based on the Gemma 3 foundation models. To enhance the
inherent multilingual capabilities of Gemma 3 for the translation
task, we employ a two-stage fine-tuning process...
```

**TranslateGemma 翻譯**：
```
我們介紹 TranslateGemma，這套基於 Gemma 3 基礎模型的開
放式機器翻譯模型。為瞭增強 Gemma 3 在翻譯任務中的固有多
語言能力，我們採用兩階段的微調過程。首先，使用大量高質量
的大規模圖閱資料進行監督微調...
```

### 雙語對照介面

**特色**：
- 📖 **左右對照**：原文、翻譯並列，方便對比
- 🎯 **專業術語保留**：MetricX-QE、AutoMQM 等保持原文
- 🔍 **術語對照表**：自動提取中英對照
- ⌨️ **鍵盤導航**：← → 快速切換頁面

---

## 開發過程：從想法到實作的 4 輪迭代

### 第 1 輪：基本翻譯功能

**目標**：讓 TranslateGemma 跑起來

**遇到的問題**：
```python
# 官方範例
target_lang = "zh-TW"  # 設定繁體中文

# 結果：輸出簡體中文 😱
"我们介绍 TranslateGemma..."  # 简体！
```

**解決方案**：發現 TranslateGemma 的 `zh-TW` bug
```python
# 加入後處理
from hanziconv import HanziConv
translation = HanziConv.toTraditional(translation)
```

✅ **成果**：成功輸出繁體中文

---

### 第 2 輪：記憶體管理

**目標**：在 Colab T4 GPU (15GB) 上翻譯完整論文

**遇到的問題**：
```
OutOfMemoryError: CUDA out of memory.
Tried to allocate 2.34 GiB (GPU 0; 14.76 GiB total capacity)
```

**原因分析**：
- Text backend：8 GB
- Multimodal backend：7 GB
- **同時載入：15 GB** ❌ 超過限制

**解決方案**：Sequential loading
```python
# Phase 1: Text pages
text_backend = TransformersBackend()
text_backend.load_model()
# ... translate text pages ...
del text_backend.model
gc.collect()
torch.cuda.empty_cache()

# Phase 2: Image pages (if needed)
image_backend = TransformersMultimodalBackend()
image_backend.load_model()
# ... translate image pages ...
```

✅ **成果**：記憶體使用從 15GB → 8GB

---

### 第 3 輪：使用者體驗優化

**問題發現**：第一個測試用戶（就是我）用 CPU 跑了 30 分鐘... 😱

**效能對比**：

| 硬體 | 每頁翻譯時間 | 7 頁總時間 | 體驗 |
|------|-------------|-----------|------|
| CPU | 15-20 分鐘 | 2-3 小時 | ❌❌❌ 不可接受 |
| T4 GPU | 25 秒 | 3 分鐘 | ✅✅✅ 流暢 |

**改進措施**：

1. **設定預設 GPU Runtime**
```json
{
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "machine_shape": "hm"
    }
  }
}
```

2. **新增醒目警告**
```markdown
⚠️ 重要：必須使用 GPU 加速

TranslateGemma 需要 GPU 才能正常運作

如何確認：
1. 點擊右上角查看「連線至代管的執行階段」
2. 確認顯示「T4」而非「Python 3」
```

✅ **成果**：用戶不會再踩 CPU 的坑

---

### 第 4 輪：互動體驗

**目標**：讓翻譯結果像閱讀器，不只是文字檔

**實作功能**：

1. **進度條顯示**
```python
from tqdm.auto import tqdm

with tqdm(total=7, desc="📖 Translating", unit="page") as pbar:
    for page_num in pages:
        pbar.set_description(f"📖 Page {page_num}/{total_pages}")
        # ... translate ...
        pbar.update(1)
```

2. **Rich HTML 輸出**
   - 雙欄對照排版
   - 漸層色標題
   - 翻譯時間顯示

3. **互動式導航**
```javascript
// 鍵盤快捷鍵
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') prevPage();
    else if (e.key === 'ArrowRight') nextPage();
});
```

4. **一鍵下載**
```python
from google.colab import files
files.download(html_file)
```

✅ **成果**：從「翻譯工具」變成「閱讀器」

---

## 使用指南：5 分鐘開始使用

### Step 1: 開啟 Colab Notebook

點擊 Badge 直接開啟：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimmyliao/trans-gemma/blob/main/arxiv-reader-colab.ipynb)

**重要**：確認右上角顯示「連線至代管的執行階段: T4」

### Step 2: 設定 HuggingFace Token

TranslateGemma 是 gated model，需要：

1. 到 [HuggingFace Settings](https://huggingface.co/settings/tokens) 建立 Token
2. 到 [TranslateGemma 頁面](https://huggingface.co/google/translategemma-4b-it) 接受授權

**Web Colab 用戶**：
- 點擊左側 🔑 圖示
- 新增 Secret：`HF_TOKEN` = 你的 token

**VS Code Colab Extension 用戶**：
- 執行 Cell 4 時會提示輸入 token
- 自動建立 `.env` 檔案（session 內有效）

### Step 3: 配置論文 ID 和翻譯範圍

```python
# Cell 6: Configuration

# arXiv Paper ID
ARXIV_ID = "2601.09012v2"  # 改成你想翻譯的論文

# 翻譯章節（頁碼範圍）
SECTIONS = {
    "abstract": (1, 1),      # 摘要
    "method": (3, 5),        # 方法論
    "experiments": (7, 9),   # 實驗結果
}

# 目標語言
TARGET_LANG = "zh-TW"  # 繁體中文
```

### Step 4: 執行翻譯

執行 Cell 8，等待約 3 分鐘：

```
📥 Downloading from arXiv: 2601.09012v2
✅ Downloaded: 2601.09012v2.pdf (12 pages)

🔄 Loading text backend...
✅ Text backend ready

📖 Translating: 7 pages
  Abstract - Page 1/12: 25.9s
  Method - Page 3/12: 22.1s
  Method - Page 4/12: 23.4s
  ...

✅ Translation Complete!
💾 Interactive HTML saved to: translation_2601.09012v2_en-zh-TW.html
```

### Step 5: 下載 HTML 檔案

執行 Cell 11（Download cell）：

```python
files.download('translation_2601.09012v2_en-zh-TW.html')
```

在瀏覽器開啟 HTML：
- 💡 使用 ← → 方向鍵切換頁面
- 📖 左右對照原文和翻譯
- 📚 滾動到最下方查看術語對照表

---

## 技術細節：給進階讀者

### 架構設計

```
┌─────────────────────────────────────────────────┐
│  arxiv-reader-colab.ipynb (使用者介面)           │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼──────┐  ┌──────▼───────┐
│ transformers │  │ transformers │
│   _backend   │  │  _multimodal │
│  (Text only) │  │   _backend   │
│   8GB RAM    │  │  (with vision)│
└───────┬──────┘  └──────┬───────┘
        │                │
        └────────┬────────┘
                 │
        ┌────────▼─────────┐
        │ TranslateGemma   │
        │ 4B parameters    │
        └──────────────────┘
```

### Memory Management Strategy

**問題**：Colab T4 GPU 只有 15GB VRAM

**解決方案**：Sequential backend loading

```python
# 智慧記憶體管理
def translate_pages(sections):
    results = []

    # Phase 1: Text-only pages
    text_backend = TransformersBackend()
    text_backend.load_model()

    for page in text_pages:
        result = text_backend.translate(page)
        results.append(result)

    # 釋放記憶體
    del text_backend.model
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: Image pages (if needed)
    if has_image_pages:
        image_backend = TransformersMultimodalBackend()
        image_backend.load_model()
        # ...

    return results
```

**效果**：
- 理論記憶體需求：15 GB
- 實際使用：8 GB
- 節省：47%

### Authentication Flow

支援 4 種認證方式（優先順序）：

```python
def get_hf_token():
    # 1. .env 檔案（VS Code Colab）
    if Path('.env').exists():
        return read_from_env()

    # 2. 環境變數
    if os.getenv('HF_TOKEN'):
        return os.getenv('HF_TOKEN')

    # 3. Colab Secrets（Web Colab）
    try:
        from google.colab import userdata
        return userdata.get('HF_TOKEN')
    except:
        pass

    # 4. 手動輸入 + 建立 .env
    token = input("HuggingFace Token: ")
    save_to_env(token)
    return token
```

**重要發現**：VS Code Colab Extension 不支援 `userdata.get()`
- 解決方案：在 remote runtime 建立 `.env`
- 參考文檔：[VSCODE-COLAB-ANALYSIS.md](https://github.com/jimmyliao/trans-gemma/blob/main/VSCODE-COLAB-ANALYSIS.md)

---

## 實際應用場景

### 場景 1：研究生讀 Related Work

**需求**：快速瀏覽 50 篇論文的摘要，篩選相關文獻

**使用方式**：
```python
SECTIONS = {
    "abstract": (1, 1),  # 只翻譯摘要
}
```

**效果**：
- 每篇論文 ~25 秒
- 50 篇 ~20 分鐘
- 比逐字讀英文快 10 倍

### 場景 2：深入研讀重要論文

**需求**：理解核心方法論，學習英文寫作

**使用方式**：
```python
SECTIONS = {
    "introduction": (1, 2),
    "method": (3, 7),
    "experiments": (8, 12),
    "conclusion": (13, 14),
}
```

**使用技巧**：
1. 先看中文理解內容
2. 對照英文學習表達方式
3. 記錄專業術語對照表
4. 練習用英文複述重點

### 場景 3：準備論文寫作

**需求**：學習特定領域的學術英文寫作

**使用方式**：
1. 收集同領域頂會論文 5-10 篇
2. 翻譯 Introduction 和 Method sections
3. 整理常用句型和術語
4. 建立個人寫作參考庫

---

## FAQ：常見問題

### Q1: 為什麼不直接用 Google Translate？

**答**：
- Google Translate：通用翻譯，可能過度意譯
- TranslateGemma：保留學術術語，適合學習
- 雙語對照：同時吸收內容和語言

### Q2: 可以翻譯中文論文成英文嗎？

**答**：可以！只需調整參數：

```python
SOURCE_LANG = "zh-TW"  # 或 "zh-CN"
TARGET_LANG = "en"
```

### Q3: Colab 免費版夠用嗎？

**答**：
- ✅ **T4 GPU**：免費版可用，速度 ~25 秒/頁
- ⚠️ **使用時數限制**：免費版約 12-15 小時/週
- 💡 **單篇論文 (10-20 頁)**：5-10 分鐘，完全夠用

---

## 總結：從工具到方法論的轉變

### 不只是翻譯工具

這個專案的核心價值不在於「把英文變中文」，而在於：

1. **雙語學習法**：同時吸收內容和語言
2. **術語累積**：建立個人學術詞彙庫
3. **寫作參考**：從閱讀到寫作的橋樑

### 開源與社群

**GitHub Repository**: [jimmyliao/trans-gemma](https://github.com/jimmyliao/trans-gemma)

包含：
- ✅ arXiv Bilingual Reader（本文介紹）
- ✅ Document Translator（通用文件翻譯）
- ✅ VS Code Colab Extension 支援分析
- ✅ 完整技術文檔

---

## 立即開始使用

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimmyliao/trans-gemma/blob/main/arxiv-reader-colab.ipynb)

**5 分鐘內**，您就能：
1. 翻譯第一篇論文
2. 下載雙語 HTML
3. 開始學習之旅

**記得**：選擇 T4 GPU，否則會等很久 😉

---

## 關於作者

**Jimmy Liao** - AI Google Developer Expert (GDE), CTO/Co-Founder

- 🐦 Twitter: [@jimmyliao](https://twitter.com/jimmyliao)
- 💼 LinkedIn: [jimmyliao](https://linkedin.com/in/jimmyliao)
- 📝 Blog: [memo.jimmyliao.net](https://memo.jimmyliao.net)
- 🔗 GitHub: [jimmyliao](https://github.com/jimmyliao)

**如果覺得有幫助，歡迎**：
- ⭐ 在 GitHub 給個 Star
- 📢 分享給需要的朋友
- 💬 留言分享你的使用心得

---

*文章發布日期：2026-01-19*
*最後更新：2026-01-19*
