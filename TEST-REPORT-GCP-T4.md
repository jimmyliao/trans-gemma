# TranslateGemma GCP T4 測試報告

**測試日期**: 2026-01-19
**測試環境**: GCP T4 Custom Runtime
**測試者**: Jimmy Liao

---

## 🖥️ 環境配置

### 硬體
- **VM**: colab-t4-systemd (GCP Compute Engine)
- **GPU**: Tesla T4 (14.6 GB VRAM)
- **CPU**: n1-standard-4 (4 vCPUs)
- **RAM**: 15 GB
- **磁碟**: 100 GB (已擴展)
- **區域**: us-central1-a

### 軟體
- **OS**: Debian GNU/Linux (GCP Deep Learning VM)
- **Python**: 3.10.x (conda py310 環境)
- **PyTorch**: 2.5.1+cu121
- **Transformers**: Latest
- **CUDA**: 12.1
- **Jupyter**: 6.5.7 (systemd service)

---

## 📝 測試過程

### Step 0: 環境偵測
```
✅ Environment: GCP
✅ Python: 3.10
✅ Working dir: /root
```

### Step 1-3: 套件安裝
**問題遇到**:
1. ❌ 系統 Python 是 3.9.2（不符合 trans-gemma ≥3.10 要求）
2. ❌ 需要手動選擇 py310 kernel
3. ❌ py310 環境缺少套件

**解決方案**:
```python
# 在 notebook 中執行
import sys
!{sys.executable} -m pip install huggingface_hub transformers accelerate sentencepiece protobuf pymupdf pillow tqdm ipywidgets -q
```

✅ 成功在 py310 環境安裝所有依賴

### Step 4: GPU 檢查
```
✅ PyTorch: 2.5.1+cu121
✅ CUDA available: True
✅ GPU: Tesla T4
✅ VRAM: 14.6 GB
📊 nvidia-smi: Tesla T4, 15360 MiB
```

### Step 6: 模型載入
**模型**: `google/translategemma-4b-it`

```
🚀 Loading TranslateGemma (4B)...
   ⏳ Downloading model (~8GB) on first run...

✅ Model loaded!
📍 Device: cuda:0
📊 Load time: 37.8s
💾 Memory: 13.0 GB available
```

**下載大小**: ~8.6 GB (model-00001: 4.96GB, model-00002: 3.64GB)

### Step 7: 翻譯測試
**測試論文**: arXiv:2403.08295 (Gemma Technical Report)

**設定**:
- Source: English
- Target: Traditional Chinese (zh-TW)
- Pages: 1 (Abstract only)

**結果**:
```
📥 Downloading arXiv:2403.08295
✅ Downloaded: 2403.08295.pdf (17 pages)

🚀 Translation Started
📖 Translating: 100% | 1/1 [00:23<00:00]
✅ Page 1: 187.38s

✅ Translation Complete!
📊 Pages: 1
⏱️  Total: 187.4s
⚡ Avg: 187.4s/page
```

### Step 9: HTML 生成
```
✅ HTML saved: translation_2403.08295_en-zh-TW.html
```

---

## 📊 效能數據

| 指標 | 數值 |
|------|------|
| **模型載入時間（首次）** | 37.8 秒 |
| **模型下載大小** | ~8.6 GB |
| **翻譯速度（每頁）** | 187.4 秒 |
| **GPU 使用** | Tesla T4 (cuda:0) |
| **VRAM 可用** | 13.0 GB |
| **記憶體佔用** | ~1.6 GB (15.4GB - 13.0GB) |

---

## 🎯 翻譯品質評估

### 原文（節錄）
```
This work introduces Gemma, a family of lightweight, state-of-the art open models
built from the research and technology used to create Gemini models. Gemma models
demonstrate strong performance across...
```

### 翻譯（節錄）
```
論文摘要：
Gemma 是一系列基于 Gemini 的轻量级、先进的开源模型。这些模型在语言理解、推理和安全性等
方面的表现优异。我們發布了兩個不同大小的模型（70 亿和 20 亿参数）...
```

### 評估
- **專業術語**: ⭐⭐⭐⭐ (正確翻譯 "lightweight", "open models", "parameters")
- **語句通順**: ⭐⭐⭐⭐ (流暢易讀)
- **格式保留**: ⭐⭐⭐⭐⭐ (完整保留段落結構)
- **簡繁混用**: ⚠️ 注意到簡體字出現（"基于"、"轻量级"、"亿"）

**改善建議**: TranslateGemma 4B 似乎傾向輸出簡體中文，即使指定 `zh-TW`。可能需要後處理轉換。

---

## ⚠️ 問題與挑戰

### 1. 翻譯速度較慢
- **預期**: 20-25 秒/頁（基於 Gemma 2B）
- **實際**: 187.4 秒/頁
- **原因分析**:
  - TranslateGemma 4B 比 Gemma 2B 參數多 2 倍
  - 推理速度自然較慢
  - 首次執行可能有額外開銷

### 2. Kernel 選擇複雜
- 用戶需要手動選擇 `Python 3.10 (trans-gemma)` kernel
- 不夠直覺，容易選錯（預設是 Python 3.9）

### 3. 套件安裝重複
- notebook 中有多個套件安裝 cells（測試遺留）
- 需要清理

### 4. 簡繁體混用
- 指定 `zh-TW` 仍出現簡體字
- 需要後處理（OpenCC 或 HanziConv）

---

## ✅ 成功要點

1. **統一 Notebook**: 單一 notebook 自動偵測環境（Colab/GCP/Local）
2. **py310 環境**: 成功建立並使用 Python 3.10 環境
3. **GPU 加速**: T4 GPU 正常運作
4. **模型載入**: TranslateGemma 4B 成功載入
5. **完整流程**: 從 PDF 下載到 HTML 生成全部完成

---

## 🎯 下一步改善

### Notebook 優化
- [ ] 移除重複的測試 cells
- [ ] 新增簡繁轉換（使用 OpenCC）
- [ ] 改善 kernel 選擇提示
- [ ] 新增進度估算（基於實際速度 187s/頁）

### 文檔更新
- [ ] 更新 README 效能表（TranslateGemma 4B 數據）
- [ ] 更新 TESTING-CHECKLIST（187s/頁 預期時間）
- [ ] 新增 GCP T4 測試報告（本文件）

### 部落格準備
- [ ] 撰寫部落格大綱
- [ ] 準備截圖素材
- [ ] 記錄關鍵學習點

---

## 📸 測試截圖素材

1. ✅ 環境偵測結果（Environment: GCP, Python: 3.10）
2. ✅ GPU 檢查（Tesla T4, 14.6 GB VRAM）
3. ✅ 模型載入進度（~8.6GB 下載）
4. ✅ 翻譯進度條（187.4s）
5. ✅ 翻譯結果對照（原文 vs 翻譯）
6. ⏳ HTML 互動介面（待截圖）

---

## 🎉 結論

**測試狀態**: ✅ PASS

成功在 GCP T4 Custom Runtime 上運行 TranslateGemma，完成 arXiv 論文翻譯全流程。

**主要發現**:
1. TranslateGemma 4B 速度較慢（187s/頁），但翻譯品質優秀
2. 需要 Python 3.10 環境（GCP VM 預設 3.9）
3. T4 GPU (15GB VRAM) 足夠運行 4B 模型
4. 統一 notebook 自動偵測環境成功

**推薦使用場景**:
- ✅ 深度閱讀學術論文（品質優先）
- ✅ 學習專業術語英文表達
- ⚠️ 大量快速翻譯（速度較慢，可考慮 Gemma 2B）

---

**測試者**: Jimmy Liao
**完成時間**: 2026-01-19
**測試版本**: arxiv-reader.ipynb (unified)
