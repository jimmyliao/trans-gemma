# TranslateGemma 使用指南：免費翻譯 PDF、圖片、網頁到繁體中文

> 一鍵在 Google Colab 上運行 TranslateGemma，將日文/英文內容翻譯成繁體中文

**作者**：Jimmy Liao (AI GDE)
**專案連結**：[github.com/jimmyliao/trans-gemma](https://github.com/jimmyliao/trans-gemma)
**發布日期**：2026-01-17

---

## 🎯 這個專案能做什麼？

這個 repository 提供一個完整的 **TranslateGemma 翻譯工具**，讓你可以：

- 📄 **翻譯 arXiv 論文**：輸入論文 ID，自動下載並翻譯
- 🖼️ **翻譯圖片中的文字**：菜單、海報、截圖都能翻譯
- 📚 **翻譯 PDF 文件**：支援文字模式和圖片模式
- 🌐 **翻譯網頁文章**：抓取網頁內容並翻譯
- 📸 **翻譯網頁截圖**：保留視覺排版的翻譯
- 🇹🇼 **強制繁體中文輸出**：確保輸出台灣慣用的繁體中文

**最棒的是**：全部在 Google Colab 免費 GPU 上運行，不需要本地環境！

---

## 🚀 快速開始（3 分鐘上手）

### 步驟 1：開啟 Colab Notebook

點擊這個按鈕直接開啟：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimmyliao/trans-gemma/blob/main/document-translator-colab.ipynb)

### 步驟 2：執行環境設置

在 Colab 中依序執行以下 cells：

**Cell 1-2: Clone 專案並安裝依賴**
```python
# 自動下載專案程式碼
!rm -rf trans-gemma
!git clone https://github.com/jimmyliao/trans-gemma.git
%cd trans-gemma

# 安裝 Python 套件
!pip install uv -q
!uv pip install --system -e ".[examples]"
```

執行時間：約 1-2 分鐘

### 步驟 3：設定 HuggingFace 認證

TranslateGemma 需要 HuggingFace 授權才能使用。

**3.1 取得 Token**
1. 前往 [HuggingFace Tokens 頁面](https://huggingface.co/settings/tokens)
2. 建立新 token（選擇 `read` 權限）
3. 前往 [TranslateGemma 模型頁](https://huggingface.co/google/translategemma-4b-it)，點擊 "Agree and access repository"

**3.2 在 Colab 設定 Secret**
1. 點擊 Colab 左側欄的 🔑 圖示
2. 新增 secret：
   - **Name**: `HF_TOKEN`
   - **Value**: 貼上你的 token

**3.3 執行認證 Cell**
```python
from huggingface_hub import login
from google.colab import userdata

HF_TOKEN = userdata.get('HF_TOKEN')
login(token=HF_TOKEN)
```

看到 `✅ Authenticated with HuggingFace` 就成功了！

### 步驟 4：選擇你要的功能

現在可以使用任何翻譯功能了！往下看各種使用場景。

---

## 📖 使用場景 1：翻譯 arXiv 論文

**適合**：想快速閱讀最新研究論文

### 使用方法

**Cell 8: 設定目標語言**
```python
TARGET_LANG = "zh-TW"  # 繁體中文
BACKEND = "transformers"
```

**Cell 10: 輸入論文 ID 並翻譯**
```python
# 設定論文 ID（從 arXiv URL 取得）
# 例如：https://arxiv.org/abs/2601.09012v2
ARXIV_ID = "2601.09012v2"

# 選擇要翻譯的頁面
START_PAGE = 1
END_PAGE = 1  # 只翻譯第 1 頁

# 執行翻譯
!python examples/translate.py \
  --mode pdf \
  --arxiv {ARXIV_ID} \
  --backend {BACKEND} \
  --target {TARGET_LANG} \
  --start-page {START_PAGE} \
  --end-page {END_PAGE}
```

### 實際範例

翻譯 TranslateGemma 技術報告的第 1 頁：

```python
ARXIV_ID = "2601.09012v2"
START_PAGE = 1
END_PAGE = 1
```

執行後會看到：
```
📄 Downloading from arXiv: 2601.09012v2
✅ Downloaded: 2601.09012v2.pdf (24 pages)
📖 Translating page 1/1...
🔄 Translating...
✅ Translation:
[繁體中文翻譯內容]
```

### 小技巧

- 想翻譯前 5 頁：設定 `END_PAGE = 5`
- 想翻譯全部：設定 `END_PAGE = None`
- 翻譯特定區段：`START_PAGE = 3, END_PAGE = 5`

---

## 🖼️ 使用場景 2：翻譯圖片（如菜單、海報）

**適合**：翻譯日文菜單、旅遊景點介紹、社群媒體圖片

### 使用方法

**Cell 16: 翻譯圖片**

預設會翻譯日文菜單範例圖片：

```python
# 使用預設的日文菜單圖片
USE_DEFAULT_IMAGE = True
DEFAULT_IMAGE_URL = "https://cdn.odigo.net/f91b9c108a1e0cd1117e1c46ee36eeca.jpg"
SOURCE_LANG = "ja"  # 日文

# 執行這個 cell，會自動：
# 1. 下載圖片
# 2. 載入多模態翻譯模型
# 3. 翻譯圖片中的文字
```

### 上傳你自己的圖片

如果要翻譯自己的圖片：

```python
# 改成 False
USE_DEFAULT_IMAGE = False

# 執行 cell 時會提示上傳檔案
# 選擇你的圖片（JPG/PNG）
```

### 調整來源語言

翻譯英文圖片：
```python
SOURCE_LANG = "en"  # 英文
```

翻譯韓文圖片：
```python
SOURCE_LANG = "ko"  # 韓文
```

---

## 📚 使用場景 3：翻譯 PDF 文件

**適合**：翻譯技術手冊、研究報告、電子書

### 方法一：上傳 PDF

**Cell 12: 上傳並翻譯 PDF**

```python
# 執行這個 cell，會出現上傳按鈕
from google.colab import files
uploaded = files.upload()  # 選擇你的 PDF

# 設定翻譯範圍
START_PAGE = 1
END_PAGE = 3  # 翻譯前 3 頁

# 自動開始翻譯
```

### 方法二：PDF 圖片模式（保留排版）

如果 PDF 包含表格、圖表、複雜排版：

**Cell 14: 使用圖片模式**

```python
# 設定 PDF
ARXIV_ID = "2601.09012v2"
START_PAGE = 3  # 有圖表的頁面
END_PAGE = 3
DPI = 96  # 解析度

# 加上 --pdf-as-image 參數
!python examples/translate.py \
  --mode pdf \
  --arxiv {ARXIV_ID} \
  --backend transformers \
  --target zh-TW \
  --pdf-as-image \
  --dpi {DPI} \
  --start-page {START_PAGE} \
  --end-page {END_PAGE}
```

### 選擇建議

| PDF 類型 | 建議模式 | 說明 |
|---------|---------|------|
| 純文字 | 文字模式 (Cell 10/12) | 速度快 |
| 有圖表/公式 | 圖片模式 (Cell 14) | 保留視覺上下文 |
| 掃描版 PDF | 圖片模式 | 需要 OCR 識別 |

---

## 🌐 使用場景 4：翻譯網頁文章

**適合**：技術部落格、新聞文章、文檔網站

### 方法一：網頁抓取（推薦）⭐

**Cell 18: 抓取網頁文字並翻譯**

```python
# 設定要翻譯的網頁 URL
ARTICLE_URL = "https://aismiley.co.jp/ai_news/gemma3-rag-api-local-use/"
SOURCE_LANG = "ja"

# 執行這個 cell，會自動：
# 1. 抓取網頁內容
# 2. 提取文章段落
# 3. 翻譯成繁體中文
```

**優點**：
- ✅ 速度快（無需截圖）
- ✅ 準確度高（直接取得原文）
- ✅ 可抓取更多內容

### 翻譯其他網站

只需修改 URL：

```python
# 翻譯日文技術文章
ARTICLE_URL = "https://qiita.com/some-article"
SOURCE_LANG = "ja"

# 翻譯英文部落格
ARTICLE_URL = "https://example.com/blog/post"
SOURCE_LANG = "en"
```

### 方法二：網頁截圖

如果網頁是動態載入或需要保留視覺效果：

**Cell 20: 截圖並翻譯**

```python
WEBSITE_URL = "https://www.yomiuri.co.jp/national/20260117-GYT1T00119/"
SOURCE_LANG = "ja"

# 執行 cell，會自動：
# 1. 啟動瀏覽器截圖
# 2. 翻譯截圖中的文字
```

### 選擇建議

| 網頁類型 | 建議方法 | 原因 |
|---------|---------|------|
| 一般文章網站 | 網頁抓取 (Cell 18) | 速度快、準確 |
| 動態網頁 (SPA) | 截圖翻譯 (Cell 20) | 需要執行 JavaScript |
| 圖文混合 | 截圖翻譯 | 保留排版 |

---

## ⚙️ 進階設定

### 調整翻譯目標語言

在 **Cell 8** 修改：

```python
# 簡體中文
TARGET_LANG = "zh-CN"

# 日文
TARGET_LANG = "ja"

# 韓文
TARGET_LANG = "ko"

# 英文
TARGET_LANG = "en"
```

### 調整圖片模式解析度

**Cell 14** 中的 DPI 設定：

```python
# 快速（較低品質）
DPI = 72

# 平衡（推薦）
DPI = 96

# 高品質（較慢）
DPI = 150
```

---

## 🔧 常見問題

### 問題 1：認證失敗

```
huggingface_hub.errors.GatedRepoError: 401 Client Error
```

**解決方法**：
1. 確認已在 HuggingFace 接受模型授權
2. 檢查 Colab Secrets 中的 `HF_TOKEN` 是否正確
3. Token 權限需包含 `read`

### 問題 2：翻譯結果是簡體中文

**解決方法**：
確認 Cell 8 中設定為：
```python
TARGET_LANG = "zh-TW"  # 不是 zh-CN
```

### 問題 3：網頁抓取失敗（Paragraphs: 0）

**可能原因**：
- 網站需要登入
- 網站有反爬蟲機制
- 網頁是動態載入（SPA）

**解決方法**：
改用 **Cell 20 截圖模式**

### 問題 4：記憶體不足

```
torch.cuda.OutOfMemoryError
```

**解決方法**：
1. 重啟 Runtime：`Runtime > Restart runtime`
2. 減少翻譯頁數（如改成 1-2 頁）
3. 使用文字模式而非圖片模式

### 問題 5：翻譯結果被截斷

**症狀**：只看到一小段翻譯

**解決方法**：
已在 Cell 18 修正（使用 `max_new_tokens=1024`）。如果還有問題，減少輸入段落數量。

---

## 📊 效能參考

在 Colab T4 GPU 上的實測速度：

| 任務 | 時間 | 備註 |
|------|------|------|
| 翻譯 PDF 1 頁（文字模式） | ~20-25 秒 | A4 頁面 |
| 翻譯 PDF 1 頁（圖片模式） | ~40-50 秒 | 包含圖表 |
| 翻譯單張圖片 | ~15-20 秒 | 菜單、海報 |
| 翻譯網頁文章（抓取） | ~20-25 秒 | 10-20 段落 |
| 翻譯網頁截圖 | ~18-25 秒 | 1280×1024 |

**注意**：第一次執行會下載模型（~8GB），需要額外 5-10 分鐘。

---

## 🎓 進階用法

### 在本地電腦運行

如果想在自己電腦上運行（需要有 GPU）：

```bash
# Clone repository
git clone https://github.com/jimmyliao/trans-gemma.git
cd trans-gemma

# 安裝依賴
pip install -e ".[examples]"

# 執行翻譯
python examples/translate.py \
  --mode pdf \
  --file document.pdf \
  --backend transformers \
  --target zh-TW
```

### 使用 CLI 工具

專案包含命令列工具，可以直接使用：

```bash
# 翻譯文字
python examples/translate.py \
  --mode text \
  --text "Hello, how are you?" \
  --source en \
  --target zh-TW

# 翻譯 PDF
python examples/translate.py \
  --mode pdf \
  --file document.pdf \
  --target zh-TW \
  --start-page 1 \
  --end-page 5

# 翻譯圖片
python examples/translate.py \
  --mode image \
  --file menu.jpg \
  --source ja \
  --target zh-TW
```

### 批次處理

翻譯多個檔案：

```bash
# 在 Colab 中
for file in *.pdf; do
    python examples/translate.py \
      --mode pdf \
      --file "$file" \
      --target zh-TW
done
```

---

## 🌟 實用範例

### 範例 1：翻譯日文技術書籍

```python
# Cell 12: 上傳 PDF
# 選擇你的日文技術書 PDF

# 設定
START_PAGE = 1
END_PAGE = 10  # 翻譯前 10 頁

# 執行翻譯
```

### 範例 2：閱讀日本餐廳菜單

```python
# Cell 16: 上傳菜單照片
USE_DEFAULT_IMAGE = False  # 改成 False
SOURCE_LANG = "ja"

# 執行 cell，上傳你拍的菜單照片
```

### 範例 3：追蹤日本科技新聞

```python
# Cell 18: 網頁抓取
ARTICLE_URL = "https://www.itmedia.co.jp/news/articles/..."
SOURCE_LANG = "ja"

# 執行翻譯
```

### 範例 4：研讀英文研究論文

```python
# Cell 10: arXiv 論文
ARXIV_ID = "2312.xxxxx"  # 你要讀的論文 ID
START_PAGE = 1
END_PAGE = 5  # 先翻譯前 5 頁看看

# 執行翻譯
```

---

## 💡 使用技巧

### 技巧 1：分段翻譯長文件

不要一次翻譯整份 PDF，分段處理：

```python
# 先翻譯第 1-5 頁
START_PAGE = 1
END_PAGE = 5

# 執行完後再翻譯第 6-10 頁
START_PAGE = 6
END_PAGE = 10
```

### 技巧 2：先試單頁確認品質

翻譯前先測試一頁：

```python
END_PAGE = 1  # 只翻譯第 1 頁
```

確認翻譯品質滿意後，再增加頁數。

### 技巧 3：善用文字模式

優先使用文字模式（更快），只在需要時才用圖片模式。

### 技巧 4：儲存翻譯結果

在 Colab 中複製翻譯結果後貼到 Google Docs 或其他文字編輯器。

---

## 📦 專案結構

```
trans-gemma/
├── document-translator-colab.ipynb  ← 主要 Notebook（你會用到的）
├── examples/
│   ├── translate.py                 ← CLI 翻譯工具
│   └── backends/                    ← 支援多種翻譯後端
│       ├── transformers_backend.py          # 文字翻譯
│       ├── transformers_multimodal_backend.py  # 圖片翻譯
│       ├── ollama_backend.py        # 本地 Ollama
│       └── mlx_backend.py           # Apple Silicon
├── README.md                        ← 專案說明
└── blog-post-zh-tw.md              ← 本文件
```

**你只需要使用 `document-translator-colab.ipynb` 就能完成所有翻譯工作！**

---

## 🔗 相關連結

- 📦 **GitHub Repository**: [github.com/jimmyliao/trans-gemma](https://github.com/jimmyliao/trans-gemma)
- 📓 **Colab Notebook**: [直接開啟](https://colab.research.google.com/github/jimmyliao/trans-gemma/blob/main/document-translator-colab.ipynb)
- 🤗 **TranslateGemma 模型**: [HuggingFace](https://huggingface.co/google/translategemma-4b-it)
- 📄 **技術報告**: [arXiv:2601.09012](https://arxiv.org/abs/2601.09012)
- 🌐 **Google 官方介紹**: [Blog](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/)

---

## 👤 關於作者

**Jimmy Liao** - AI Google Developer Expert (GDE)，AI 新創公司 CTO/共同創辦人

- 🐦 Twitter: [@jimmyliao](https://twitter.com/jimmyliao)
- 💼 LinkedIn: [jimmyliao](https://linkedin.com/in/jimmyliao)
- 📝 Blog: [memo.jimmyliao.net](https://memo.jimmyliao.net)
- 🎤 Sessionize: [jimmy-liao](https://sessionize.com/jimmy-liao/)

---

## 🙏 致謝

- Google TranslateGemma 團隊提供優秀的開源模型
- HuggingFace 提供模型託管與 transformers 函式庫
- Google Colab 提供免費 GPU 資源

---

## 📄 授權

本專案採用 **MIT License**，可自由用於商業與非商業用途。

---

## 🎉 開始使用

現在就開啟 Colab Notebook，開始你的翻譯之旅：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimmyliao/trans-gemma/blob/main/document-translator-colab.ipynb)

**3 分鐘快速上手流程**：
1. 開啟 Notebook
2. 執行 Cell 1-2（環境設置）
3. 執行 Cell 5-8（認證與配置）
4. 選擇你要的翻譯功能並執行對應的 Cell

就是這麼簡單！🚀

---

**有問題或建議？** 歡迎到 [GitHub Issues](https://github.com/jimmyliao/trans-gemma/issues) 提出！
