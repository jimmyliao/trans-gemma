"""
TranslateGemma 正確使用範例

TranslateGemma 使用特殊的 chat template 格式，需要包含 source_lang_code 和 target_lang_code
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 語言代碼映射（TranslateGemma 使用 ISO 639-1 兩碼標準）
# 修正重點：
# 1. 改用 ISO 639-1（兩碼），例如 English 是 "en" 而非 "eng"
# 2. 中文使用 "zh-TW" / "zh-CN" 格式（符合模型對 Regionalized variant 的要求）

LANGUAGE_CODES = {
    # 主要常用語言
    "English": "en",
    "Traditional Chinese (Taiwan)": "zh-TW",  # 繁體中文
    "Simplified Chinese (China)": "zh-CN",    # 簡體中文
    "Japanese": "ja",
    "Korean": "ko",

    # 歐洲語言
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Dutch": "nl",
    "Polish": "pl",

    # 亞洲與中東語言
    "Arabic": "ar",
    "Hindi": "hi",
    "Vietnamese": "vi",
    "Thai": "th",
    "Indonesian": "id",
    "Hebrew": "he",
    "Persian": "fa",
}

def get_lang_code(lang_name):
    """
    取得語言代碼，如果不存在則拋出錯誤

    Args:
        lang_name: 語言名稱

    Returns:
        對應的語言代碼

    Raises:
        ValueError: 當語言不被支援時
    """
    code = LANGUAGE_CODES.get(lang_name)
    if not code:
        raise ValueError(f"Language '{lang_name}' not supported.")
    return code

def translate(
    text: str,
    target_lang: str = "Traditional Chinese (Taiwan)",
    source_lang: str = "English",
    model=None,
    tokenizer=None,
    max_new_tokens: int = 256
):
    """
    使用 TranslateGemma 翻譯文本（正確格式）

    Args:
        text: 要翻譯的文本
        target_lang: 目標語言名稱
        source_lang: 來源語言名稱
        model: TranslateGemma 模型
        tokenizer: TranslateGemma tokenizer
        max_new_tokens: 最大生成 token 數

    Returns:
        翻譯結果
    """
    # 取得語言代碼（會自動驗證）
    source_code = get_lang_code(source_lang)
    target_code = get_lang_code(target_lang)

    # TranslateGemma 的正確格式
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                    "source_lang_code": source_code,
                    "target_lang_code": target_code
                }
            ]
        }
    ]

    # 使用 chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    # 生成翻譯
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 解碼結果
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取翻譯結果（TranslateGemma 通常只返回翻譯文本）
    # 如果包含原文，需要分離
    lines = result.strip().split('\n')
    translation = lines[-1].strip() if lines else result.strip()

    return translation


def demo():
    """示範使用"""
    print("="*80)
    print("TranslateGemma 正確使用範例")
    print("="*80)

    # 載入模型
    print("\n載入模型...")
    MODEL_ID = "google/translategemma-4b-it"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("模型載入成功！\n")

    # 測試 1: 英文 → 繁體中文
    print("測試 1: 英文 → 繁體中文")
    text1 = "Hello, how are you today?"
    result1 = translate(text1, "Traditional Chinese (Taiwan)", "English", model, tokenizer)
    print(f"原文: {text1}")
    print(f"譯文: {result1}\n")

    # 測試 2: 英文 → 日文
    print("測試 2: 英文 → 日文")
    text2 = "Machine learning is transforming the world."
    result2 = translate(text2, "Japanese", "English", model, tokenizer)
    print(f"原文: {text2}")
    print(f"譯文: {result2}\n")

    # 測試 3: 繁體中文 → 英文
    print("測試 3: 繁體中文 → 英文")
    text3 = "我喜歡在週末閱讀和寫程式。"
    result3 = translate(text3, "English", "Traditional Chinese (Taiwan)", model, tokenizer)
    print(f"原文: {text3}")
    print(f"譯文: {result3}\n")

    print("="*80)
    print("所有測試完成！")
    print("="*80)


if __name__ == "__main__":
    demo()
