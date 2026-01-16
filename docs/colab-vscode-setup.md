# VS Code + Google Colab 整合設置指南

本指南說明如何設置 VS Code 與 Google Colab 的整合，讓你可以在 VS Code 中直接編輯和執行 Colab notebook。

## 前置需求

- VS Code 1.85 或更新版本
- Google 帳號
- Hugging Face 帳號（用於存取 TranslateGemma 模型）
- 穩定的網路連線

### ⚠️ 重要：先完成 Hugging Face 存取申請

TranslateGemma 是 gated repository，請先完成：
1. [申請模型存取](https://huggingface.co/google/translategemma-4b-it)
2. [建立 Access Token](https://huggingface.co/settings/tokens)

詳細步驟請參考：[Hugging Face 存取設定指南](huggingface-access.md)

## 安裝步驟

### 1. 安裝 VS Code 擴充套件

有兩種方式安裝：

#### 方式 A: 從 VS Code Marketplace 安裝

1. 開啟 VS Code
2. 點擊左側的擴充套件圖示（或按 `Cmd+Shift+X` / `Ctrl+Shift+X`）
3. 搜尋 "Jupyter" 和 "Google Colab"
4. 安裝以下擴充套件：
   - **Jupyter** (Microsoft)
   - **Google Colab** (Google)

#### 方式 B: 使用命令列安裝

```bash
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.vscode-jupyter-colab
```

### 2. 連接 Google 帳號

1. 在 VS Code 中開啟命令面板（`Cmd+Shift+P` / `Ctrl+Shift+P`）
2. 輸入 "Colab: Sign In"
3. 選擇你的 Google 帳號
4. 授權 VS Code 存取 Google Colab

### 3. 開啟 Colab Notebook

#### 方式 A: 開啟本專案的 notebook

1. 在 VS Code 中開啟 trans-gemma 專案資料夾
2. 開啟 `translategemma-colab.ipynb`
3. 點擊右上角的 kernel 選擇器
4. 選擇 "Google Colab"
5. 等待連接到 Colab runtime

#### 方式 B: 從 GitHub 開啟

1. 使用命令面板（`Cmd+Shift+P` / `Ctrl+Shift+P`）
2. 輸入 "Colab: Open from GitHub"
3. 輸入：`jimmyliao/trans-gemma`
4. 選擇 `translategemma-colab.ipynb`

## 選擇 GPU Runtime

1. 開啟 notebook 後，點擊右上角的 runtime 設定
2. 或使用命令面板：`Colab: Change Runtime Type`
3. 選擇 "T4 GPU"
4. 點擊 "Save"
5. 等待 runtime 重新連接

## 使用技巧

### 執行 Cells

- **執行單個 cell**: `Shift+Enter`
- **執行所有 cells**: 使用命令面板 → "Notebook: Execute All Cells"
- **執行到當前 cell**: 使用命令面板 → "Notebook: Execute Cells Above"

### 檢查 GPU 狀態

執行以下 cell 來確認 GPU 是否正常運作：

```python
!nvidia-smi
```

### 儲存變更

- **自動儲存**: VS Code 會自動儲存變更到本地檔案
- **同步到 Colab**: 變更會自動同步到 Colab runtime
- **Commit 到 Git**: 使用 VS Code 的 Git 整合提交變更

## 常見問題

### Q: 為什麼連接 Colab 失敗？

A: 請檢查：
1. 網路連線是否正常
2. 是否已登入 Google 帳號
3. 是否有過多的 Colab sessions 正在運行（免費版限制 1 個）

### Q: GPU 不可用怎麼辦？

A:
1. 確認已選擇 T4 GPU runtime
2. 檢查 Colab 免費配額是否用完（每天有限制）
3. 嘗試重新連接 runtime

### Q: 模型下載很慢？

A:
1. Colab 提供高速網路，通常下載很快
2. 如果速度慢，可能是 Hugging Face 伺服器繁忙
3. 可以使用 Kaggle 或其他模型源作為替代

### Q: 如何在本地和 Colab 之間切換？

A:
1. 點擊右上角的 kernel 選擇器
2. 選擇 "Local" 或 "Google Colab"
3. 等待 kernel 重新啟動

## 優勢

使用 VS Code + Colab 整合的優勢：

- ✅ **熟悉的 VS Code 介面**：使用你習慣的編輯器
- ✅ **免費 GPU**：使用 Colab 免費的 T4 GPU
- ✅ **本地檔案管理**：直接編輯本地 Git repository
- ✅ **版本控制**：輕鬆使用 Git 追蹤變更
- ✅ **擴充套件**：使用 VS Code 的所有擴充套件
- ✅ **快捷鍵**：使用你習慣的 VS Code 快捷鍵

## 相關資源

- [Google Colab is coming to VS Code](https://developers.googleblog.com/en/google-colab-is-coming-to-vs-code/)
- [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- [Google Colab Documentation](https://colab.research.google.com/)

## 下一步

設置完成後，你可以：

1. 執行 `translategemma-colab.ipynb` 中的所有 cells
2. 實驗不同的翻譯範例
3. 修改程式碼並測試
4. 準備部署到 Cloud Run

祝你使用愉快！🚀
