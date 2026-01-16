# Hugging Face 模型存取設定

TranslateGemma 是一個 **gated repository**，需要先在 Hugging Face 上申請存取權限才能下載模型。

## 步驟 1: 申請模型存取權限

### 1.1 前往 Hugging Face 模型頁面

訪問：[https://huggingface.co/google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it)

### 1.2 登入 Hugging Face 帳號

如果還沒有帳號，請先註冊：[https://huggingface.co/join](https://huggingface.co/join)

### 1.3 申請存取

1. 在模型頁面上，你會看到「**Request access**」或「**申請存取**」的按鈕
2. 點擊後，閱讀並同意使用條款
3. 提交申請

**注意**：通常申請會立即獲得批准，但有時可能需要幾分鐘到幾小時。

### 1.4 確認存取權限

申請批准後，你應該會收到 email 通知，或在模型頁面上看到「**You have been granted access**」的訊息。

## 步驟 2: 建立 Hugging Face Access Token

### 2.1 前往 Token 設定頁面

訪問：[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 2.2 建立新 Token

1. 點擊「**New token**」
2. 設定 Token 名稱（例如：`colab-translategemma`）
3. 選擇 Token 類型：
   - **Read**: 足夠用於下載模型（推薦）
   - **Write**: 如果需要上傳模型
4. 點擊「**Generate token**」
5. **複製並保存你的 token**（只會顯示一次！）

### 2.3 Token 格式

你的 token 會類似這樣：
```
hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 步驟 3: 在 Colab 中使用 Token

### 方式 A: 使用 Colab Secrets（推薦）

1. 在 Colab notebook 中，點擊左側欄的 **🔑 Secrets** 圖示
2. 點擊「**Add new secret**」
3. 設定：
   - Name: `HF_TOKEN`
   - Value: 貼上你的 Hugging Face token
4. 啟用「Notebook access」

然後在 notebook 中使用：

```python
from google.colab import userdata
from huggingface_hub import login

# 從 Colab Secrets 讀取 token
hf_token = userdata.get('HF_TOKEN')
login(token=hf_token)
```

### 方式 B: 直接輸入（較不安全）

在 notebook 中執行：

```python
from huggingface_hub import login

# 會跳出輸入框讓你貼上 token
login()
```

### 方式 C: 硬編碼（不推薦，僅測試用）

```python
from huggingface_hub import login

# ⚠️ 不要將 token commit 到 Git！
login(token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
```

## 步驟 4: 驗證設定

執行以下代碼確認可以存取模型：

```python
from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("google/translategemma-4b-it")
    print("✅ 成功存取 TranslateGemma 模型！")
except Exception as e:
    print(f"❌ 無法存取模型：{e}")
```

## 常見問題

### Q: 為什麼模型需要申請存取？

A: TranslateGemma 是 Google 的官方模型，需要用戶同意使用條款才能使用。這是常見的 gated repository 做法。

### Q: 申請需要多久？

A: 通常立即獲得批准。如果超過 1 小時還未批准，請檢查：
1. 是否已登入正確的 Hugging Face 帳號
2. Email 是否已驗證
3. 聯繫 Hugging Face 支援

### Q: Token 會過期嗎？

A: Read token 通常不會過期，但你可以隨時在設定頁面撤銷並建立新的 token。

### Q: 可以分享 token 嗎？

A: ⚠️ **不可以**！Token 等同於你的帳號密碼，不應該分享給他人或 commit 到公開的 Git repository。

### Q: 在 Cloud Run 部署時如何使用 token？

A: 在部署時，你可以：

1. **使用 Secret Manager**（推薦）：
   ```bash
   # 建立 secret
   echo -n "hf_xxx" | gcloud secrets create HF_TOKEN --data-file=-

   # 在 Cloud Run 中使用
   gcloud run deploy ... \
     --set-secrets=HF_TOKEN=HF_TOKEN:latest
   ```

2. **使用環境變數**（較不安全）：
   ```bash
   gcloud run deploy ... \
     --set-env-vars="HF_TOKEN=hf_xxx"
   ```

## 其他 Gated Models

如果未來需要存取其他 gated models（例如 Llama、Gemma），流程類似：

1. 前往模型頁面
2. 申請存取
3. 使用相同的 Hugging Face token

## 相關連結

- [Hugging Face Access Control 文檔](https://huggingface.co/docs/hub/security-tokens)
- [Google TranslateGemma 官方頁面](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/)
- [Hugging Face Hub Python Library](https://huggingface.co/docs/huggingface_hub/index)

---

**準備好了嗎？** 現在你可以回到 [Colab notebook](../translategemma-colab.ipynb) 繼續實驗！
