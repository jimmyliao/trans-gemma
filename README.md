# TranslateGemma 實驗專案

> 使用 Google Colab + VS Code 開發 TranslateGemma 翻譯模型，並部署到 Cloud Run

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 📋 專案簡介

本專案實驗 Google 最新開源的 **TranslateGemma** 翻譯模型（4B 參數版本），採用 **Colab-First** 開發策略：

- ✅ 在 Google Colab 免費 T4 GPU 上驗證模型
- ✅ 使用 VS Code + Colab 整合進行雲端開發
- ✅ 設計可直接部署到 Cloud Run 的流程
- ✅ 支援 55 種語言的高品質翻譯

### 為什麼選擇 Colab-First？

- 💡 **零本地資源消耗**：不需要高階 GPU 或大量磁碟空間
- 🚀 **快速驗證**：免費 T4 GPU 足以運行 4B 模型
- 🔄 **無縫部署**：notebook 可直接轉換為生產環境
- 💰 **成本優化**：開發免費，部署按需計費

## 🚀 快速開始

### ⚠️ 前置需求：Hugging Face 模型存取

TranslateGemma 是 **gated repository**，使用前需要：

1. 前往 [Hugging Face TranslateGemma 頁面](https://huggingface.co/google/translategemma-4b-it)
2. 點擊「**Request access**」申請存取（通常立即批准）
3. 建立 [Hugging Face Access Token](https://huggingface.co/settings/tokens)

詳細步驟請參考：[Hugging Face 存取設定指南](docs/huggingface-access.md)

### 選項 1: Google Colab（推薦）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimmyliao/trans-gemma/blob/main/translategemma-colab.ipynb)

1. **完成上方的 Hugging Face 存取申請**
2. 點擊按鈕在 Colab 中開啟 notebook
3. 確認 GPU 已啟用（Runtime > Change runtime type > T4 GPU）
4. 在 Colab Secrets 中設定 `HF_TOKEN`（或手動輸入）
5. 執行所有 cells，體驗 TranslateGemma 翻譯功能

### 選項 2: VS Code + Colab 整合

1. 安裝 [VS Code Colab Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-jupyter-colab)
2. 在 VS Code 中開啟 `translategemma-colab.ipynb`
3. 連接到 Colab runtime 並執行

## 📦 專案結構

```
trans-gemma/
├── translategemma-colab.ipynb    # 主要 Colab notebook（核心檔案）
├── README.md                      # 本檔案
├── .gitignore                     # Git 忽略規則
│
├── cloudrun/                      # Cloud Run 部署配置
│   ├── Dockerfile                # 容器定義
│   ├── requirements.txt          # Python 依賴
│   ├── main.py                   # FastAPI 應用
│   └── deploy.sh                 # 部署腳本
│
├── docs/                          # 文檔
│   ├── colab-vscode-setup.md    # VS Code Colab 整合教學
│   └── deployment-guide.md      # 部署指南
│
├── examples/                      # 使用範例
│   └── simple-translation.py    # 簡單翻譯範例
│
└── tests/                         # 測試檔案
    └── test_translation.py      # 單元測試
```

## 🎯 功能特色

- ✅ **多語言翻譯**：支援 55 種語言（英↔中、英↔日等）
- ✅ **Colab 免費 GPU**：在 T4 GPU 上運行 4B 模型
- ✅ **FastAPI 服務**：RESTful API 設計
- ✅ **Cloud Run 部署**：一鍵部署到 GCP
- ✅ **效能基準測試**：完整的效能評估數據

## 📚 文檔

- [VS Code Colab 整合教學](docs/colab-vscode-setup.md)
- [Cloud Run 部署指南](docs/deployment-guide.md)
- [API 參考文件](docs/api-reference.md)
- [效能基準測試](docs/performance-benchmarks.md)

## 🔗 相關資源

### TranslateGemma 官方資源
- [Google Blog: TranslateGemma](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/)
- [Kaggle Models](https://www.kaggle.com/models/google/translategemma/)
- [Hugging Face: translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it)
- [Technical Report (arXiv)](https://arxiv.org/abs/2601.09012)

### 開發工具
- [Google Colab](https://colab.research.google.com/)
- [VS Code Colab Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-jupyter-colab)
- [Cloud Run GPU Documentation](https://cloud.google.com/run/docs/configuring/services/gpu)

## 📊 效能數據

| 平台 | GPU | 記憶體 | 推理速度 | 成本 |
|------|-----|--------|---------|------|
| Google Colab | T4 | 12GB | ~40 tok/s | 免費 |
| Cloud Run | L4 | 24GB | ~80 tok/s | $0.67/hr |

> 詳細效能測試結果請參考 [performance-benchmarks.md](docs/performance-benchmarks.md)

## 🚀 部署到 Cloud Run

### 使用 GitHub Actions（推薦）

1. Fork 本專案
2. 設定 GCP 認證（Workload Identity）
3. Push 到 main 分支，自動觸發部署

### 手動部署

```bash
cd cloudrun
./deploy.sh
```

詳細步驟請參考 [deployment-guide.md](docs/deployment-guide.md)

## 🤝 貢獻

歡迎提交 Issue 或 Pull Request！

## 📝 授權

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 👤 作者

**Jimmy Liao** ([@jimmyliao](https://github.com/jimmyliao))

---

⭐ 如果這個專案對你有幫助，請給個 Star！
