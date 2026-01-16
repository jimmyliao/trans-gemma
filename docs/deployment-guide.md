# Cloud Run 部署指南

本指南說明如何將 TranslateGemma API 部署到 Google Cloud Run。

## 前置需求

- Google Cloud 帳號（已啟用計費）
- GCP 專案（已啟用 GPU）
- `gcloud` CLI 工具
- Docker（用於本地部署）

## 方式一：手動部署（使用腳本）

### 1. 設定環境變數

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"  # 或其他支援 GPU 的區域
export SERVICE_NAME="translategemma-4b"
```

### 2. 認證到 GCP

```bash
gcloud auth login
gcloud config set project $PROJECT_ID
```

### 3. 啟用必要的 API

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### 4. 執行部署腳本

```bash
cd cloudrun
./deploy.sh
```

部署腳本會自動：
- 建立 Docker 映像
- 推送到 Google Container Registry
- 部署到 Cloud Run with L4 GPU
- 輸出服務 URL

### 5. 測試 API

部署完成後，使用輸出的 URL 測試：

```bash
SERVICE_URL="your-service-url"

# 健康檢查
curl $SERVICE_URL/health

# 測試翻譯
curl -X POST $SERVICE_URL/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "target_lang": "Traditional Chinese"}'
```

## 方式二：GitHub Actions 自動部署

### 1. Fork 或 Clone 專案

如果你還沒有，請先 fork 或 clone 這個專案到你的 GitHub 帳號。

### 2. 設定 GCP Service Account

#### 2.1 建立 Service Account

```bash
export PROJECT_ID="your-gcp-project-id"
export SA_NAME="github-actions-deploy"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# 建立 Service Account
gcloud iam service-accounts create $SA_NAME \
  --display-name="GitHub Actions Deployment"

# 授予必要權限
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountUser"

# 建立 key
gcloud iam service-accounts keys create key.json \
  --iam-account=$SA_EMAIL
```

#### 2.2 設定 GitHub Secrets

在 GitHub repository 中設定以下 secrets：

1. 前往 GitHub repository → Settings → Secrets and variables → Actions
2. 點擊 "New repository secret"
3. 新增以下 secrets：

| Secret Name | Value |
|------------|-------|
| `GCP_PROJECT_ID` | 你的 GCP 專案 ID |
| `GCP_SA_KEY` | `key.json` 的完整內容 |

**重要**：設定完成後，請刪除本地的 `key.json` 檔案：

```bash
rm key.json
```

### 3. 觸發部署

#### 自動部署
Push 到 `main` 分支會自動觸發部署：

```bash
git add .
git commit -m "Your commit message"
git push origin main
```

#### 手動觸發
1. 前往 GitHub repository → Actions
2. 選擇 "Deploy to Cloud Run" workflow
3. 點擊 "Run workflow"

### 4. 監控部署

1. 在 GitHub Actions 頁面查看部署進度
2. 部署完成後，在日誌中找到服務 URL
3. 或使用以下命令查詢：

```bash
gcloud run services describe translategemma-4b \
  --region=us-central1 \
  --format='value(status.url)'
```

## 配置說明

### Cloud Run 配置

當前配置（可在 `deploy.sh` 或 workflow 中調整）：

| 參數 | 值 | 說明 |
|------|-----|------|
| CPU | 4 | vCPU 數量 |
| Memory | 16Gi | 記憶體大小 |
| GPU | 1 | GPU 數量 |
| GPU Type | nvidia-l4 | GPU 類型 |
| Max Instances | 3 | 最大實例數 |
| Min Instances | 0 | 最小實例數（scale-to-zero）|
| Timeout | 300s | 請求超時時間 |

### 支援的 GPU 區域

目前支援 Cloud Run GPU 的區域：

- `us-central1` (推薦)
- `us-east1`
- `us-west1`
- `europe-west1`
- `asia-east1`

詳細清單請參考：[Cloud Run GPU 區域](https://cloud.google.com/run/docs/locations)

## 成本估算

### L4 GPU 定價（us-central1）

- **GPU**: ~$0.67/hour
- **vCPU**: ~$0.024/vCPU-hour
- **Memory**: ~$0.003/GiB-hour
- **Requests**: $0.40/million requests

### 使用 scale-to-zero 優化成本

當前配置使用 `min-instances=0`，代表沒有請求時不會產生費用。

### 預估月成本（假設場景）

假設：
- 每天 1000 次翻譯請求
- 平均每次請求 3 秒
- 總運行時間：1000 × 3 ÷ 3600 ≈ 0.83 小時/天

**每月成本**：
- GPU: $0.67 × 0.83 × 30 = ~$16.7
- vCPU: $0.024 × 4 × 0.83 × 30 = ~$2.4
- Memory: $0.003 × 16 × 0.83 × 30 = ~$1.2
- Requests: $0.40 × 30 ÷ 1000 = ~$0.01

**總計**: ~$20/月

## 疑難排解

### 問題：部署失敗，顯示 GPU quota 不足

**解決方案**：
1. 前往 [GCP Quotas 頁面](https://console.cloud.google.com/iam-admin/quotas)
2. 搜尋 "Cloud Run GPU"
3. 請求增加配額

### 問題：服務冷啟動太慢

**原因**：模型載入需要時間（約 30-60 秒）

**解決方案**：
- 使用 `min-instances=1` 保持至少一個實例運行
- 或接受較長的首次請求時間

### 問題：記憶體不足錯誤

**解決方案**：
- 增加 memory 設定（例如改為 24Gi）
- 或使用較小的模型

## 監控與日誌

### 查看日誌

```bash
gcloud run services logs read translategemma-4b \
  --region=us-central1 \
  --limit=50
```

### 監控指標

前往 GCP Console：
1. Cloud Run → 選擇服務
2. 查看 Metrics 頁面
3. 監控：請求數、延遲、錯誤率、CPU/Memory/GPU 使用率

## 清理資源

如果要刪除服務：

```bash
gcloud run services delete translategemma-4b \
  --region=us-central1
```

刪除 Docker 映像：

```bash
gcloud container images delete gcr.io/$PROJECT_ID/translategemma-4b
```

## 下一步

- 參考 [API 文件](api-reference.md) 了解完整 API 規格
- 查看 [效能基準測試](performance-benchmarks.md) 了解效能數據
- 探索 [examples/](../examples/) 中的使用範例

## 相關資源

- [Cloud Run GPU 文檔](https://cloud.google.com/run/docs/configuring/services/gpu)
- [Cloud Run 定價](https://cloud.google.com/run/pricing)
- [GitHub Actions for GCP](https://github.com/google-github-actions)
