# GCP 設定腳本

TranslateGemma 的 GCP 相關部署腳本集合。

## 📁 腳本列表

### 1. `setup_gcp_t4_auto_shutdown.sh`
**用途**: 建立具備自動關機功能的 Preemptible T4 GPU VM

**特色**:
- 💰 Preemptible VM：成本降低 ~80% ($0.11/hour vs $0.54/hour)
- ⏱️ 閒置 1 小時自動關機
- 🎯 T4 GPU (與 Colab 免費版相同規格)
- 📦 預裝 PyTorch + CUDA 深度學習環境

**使用方式**:
```bash
# 編輯腳本中的專案 ID
vim setup_gcp_t4_auto_shutdown.sh
# 修改: PROJECT="your-gcp-project-id"

# 執行建立
bash setup_gcp_t4_auto_shutdown.sh
```

**估計成本**: ~$0.11/hour (us-central1 區域)

---

### 2. `setup_gcp_t4_colab.sh`
**用途**: 建立標準 T4 GPU VM 供 Colab 自訂執行環境使用

**特色**:
- 🖥️ n1-standard-4 + T4 GPU
- 🔧 PyTorch 深度學習映像檔
- 🌐 Jupyter 防火牆規則
- 📡 支援 Colab 自訂執行環境連線

**使用方式**:
```bash
# 編輯專案設定
vim setup_gcp_t4_colab.sh
# 修改: PROJECT="your-gcp-project-id"

# 執行建立
bash setup_gcp_t4_colab.sh
```

**後續步驟**:
1. SSH 進入 VM
2. 執行 `setup_colab_runtime.sh` 設定 Jupyter
3. 在 Colab 連接自訂執行環境

**估計成本**: ~$0.54/hour

---

### 3. `setup_colab_runtime.sh`
**用途**: 在 GCP VM 上設定 Jupyter 供 Colab 連線

**特色**:
- 📦 安裝 Jupyter + jupyter_http_over_ws
- 🔄 Clone trans-gemma repository
- 🔧 安裝專案相依套件
- 🚀 啟動 Jupyter server

**使用方式**:
```bash
# 在 GCP VM 內執行
curl -sSL https://raw.githubusercontent.com/jimmyliao/trans-gemma/main/scripts/gcp/setup_colab_runtime.sh | bash

# 或手動執行
bash setup_colab_runtime.sh
```

**連線到 Colab**:
1. 複製顯示的 Jupyter URL (含 token)
2. 將 `127.0.0.1` 改為 VM 的外部 IP
3. 在 Colab: 連線 → 連接到本機執行環境
4. 貼上: `http://YOUR_VM_IP:8888/?token=YOUR_TOKEN`

---

## 💡 使用情境

| 情境 | 推薦腳本 | 理由 |
|------|---------|------|
| 開發測試 | `setup_gcp_t4_auto_shutdown.sh` | 成本低、自動關機 |
| 生產環境 | `setup_gcp_t4_colab.sh` | 穩定、可長時間運行 |
| Colab 連線 | 兩者皆可 + `setup_colab_runtime.sh` | 看預算選擇 |

## 🛑 注意事項

**Preemptible VM 限制**:
- Google 可隨時終止 (最長 24 小時)
- 適合開發、測試、短期任務
- 不適合關鍵生產環境

**成本控管**:
```bash
# 停止 VM (不刪除)
gcloud compute instances stop VM_NAME --zone=ZONE --project=PROJECT

# 啟動 VM
gcloud compute instances start VM_NAME --zone=ZONE --project=PROJECT

# 刪除 VM (釋放所有資源)
gcloud compute instances delete VM_NAME --zone=ZONE --project=PROJECT
```

## 📚 相關文件

- [TranslateGemma 主專案](https://github.com/jimmyliao/trans-gemma)
- [GCP Compute Engine 定價](https://cloud.google.com/compute/pricing)
- [Colab 自訂執行環境](https://research.google.com/colaboratory/local-runtimes.html)

---

**維護者**: Jimmy Liao (@jimmyliao)
**最後更新**: 2026-01-19
