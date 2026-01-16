"""
本地測試腳本 - 使用 .env 檔案儲存 HF_TOKEN

使用方式：
1. 複製 .env.example 到 .env
2. 在 .env 中填入你的 HF_TOKEN
3. 執行: python examples/local-test.py
"""

import os
import sys
from pathlib import Path

# 添加專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_env():
    """載入 .env 檔案"""
    env_file = project_root / ".env"

    if not env_file.exists():
        print("❌ .env 檔案不存在")
        print()
        print("請執行以下步驟：")
        print("1. 複製 .env.example 到 .env:")
        print("   cp .env.example .env")
        print()
        print("2. 編輯 .env 並填入你的 HF_TOKEN:")
        print("   # 從 https://huggingface.co/settings/tokens 取得")
        print("   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return False

    # 讀取 .env 檔案
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

    return True

def test_token():
    """測試 HF_TOKEN 是否有效"""
    token = os.getenv("HF_TOKEN")

    if not token or token.startswith("hf_xxx"):
        print("❌ HF_TOKEN 未設定或使用預設值")
        print("請在 .env 檔案中設定有效的 HF_TOKEN")
        return False

    print("✅ HF_TOKEN 已設定")
    print(f"   Token: {token[:10]}...{token[-5:]}")

    try:
        from huggingface_hub import login
        login(token=token)
        print("✅ Hugging Face 認證成功")
        return True
    except Exception as e:
        print(f"❌ Hugging Face 認證失敗: {e}")
        return False

def test_model_access():
    """測試模型存取"""
    print("\n測試 TranslateGemma 模型存取...")

    try:
        from transformers import AutoTokenizer

        MODEL_ID = os.getenv("MODEL_ID", "google/translategemma-4b-it")
        print(f"載入 tokenizer: {MODEL_ID}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        print(f"✅ Tokenizer 載入成功")
        print(f"   詞彙表大小: {len(tokenizer)}")
        return True

    except Exception as e:
        print(f"❌ 模型存取失敗: {e}")
        print()
        print("可能的原因：")
        print("1. 尚未申請 TranslateGemma 存取權限")
        print("   前往: https://huggingface.co/google/translategemma-4b-it")
        print("2. Token 權限不足（需要 Read 權限）")
        return False

def test_translation():
    """測試翻譯功能"""
    print("\n測試翻譯功能...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        MODEL_ID = os.getenv("MODEL_ID", "google/translategemma-4b-it")

        print("載入模型（可能需要幾分鐘）...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        print("✅ 模型載入成功")

        # 測試翻譯（使用正確的 TranslateGemma 格式）
        print("\n執行測試翻譯...")
        text = "Hello, world!"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "source_lang_code": "eng",
                        "target_lang_code": "zho_Hant"
                    }
                ]
            }
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"原文: {text}")
        print(f"譯文: {result}")
        print()
        print("✅ 翻譯測試成功")
        return True

    except Exception as e:
        print(f"❌ 翻譯測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    print("="*80)
    print("TranslateGemma 本地測試")
    print("="*80)
    print()

    # 1. 載入 .env
    print("步驟 1: 載入 .env 檔案")
    if not load_env():
        return 1
    print("✅ .env 檔案載入成功")
    print()

    # 2. 測試 token
    print("步驟 2: 測試 HF_TOKEN")
    if not test_token():
        return 1
    print()

    # 3. 測試模型存取
    print("步驟 3: 測試模型存取")
    if not test_model_access():
        return 1
    print()

    # 4. 測試翻譯（可選，因為載入模型需要較長時間）
    print("步驟 4: 測試翻譯功能")
    response = input("是否執行翻譯測試？(載入模型需要較長時間) [y/N]: ")
    if response.lower() == 'y':
        if not test_translation():
            return 1
    else:
        print("⏭️  跳過翻譯測試")

    print()
    print("="*80)
    print("🎉 所有測試完成！")
    print("="*80)
    print()
    print("下一步：")
    print("1. 在 Colab 中開啟 translategemma-colab.ipynb")
    print("2. 或部署到 Cloud Run: cd cloudrun && ./deploy.sh")

    return 0

if __name__ == "__main__":
    sys.exit(main())
