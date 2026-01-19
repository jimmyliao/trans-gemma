"""
Improved translation extraction logic for TranslateGemma

This module provides robust extraction strategies to handle various output formats
"""

import re


def extract_translation_v2(full_output: str, source_lang: str, target_lang: str) -> str:
    """
    Enhanced translation extraction with multiple fallback strategies

    Args:
        full_output: Full model output (decoded with skip_special_tokens=True)
        source_lang: Source language code (e.g., 'en')
        target_lang: Target language code (e.g., 'zh-TW')

    Returns:
        Extracted translation text
    """

    # Strategy 1: Remove the prompt template if present
    # TranslateGemma sometimes returns the full prompt + response
    if 'user\n' in full_output or 'You are a professional' in full_output:
        # Find the last occurrence of common prompt patterns
        patterns = [
            r'You are a professional.*?Please translate the following.*?text into.*?:\s*',
            r'user\n.*?\n\n',  # Remove user prefix
            r'^.*?Please translate.*?:\s*\n+',
        ]

        for pattern in patterns:
            match = re.search(pattern, full_output, re.DOTALL)
            if match:
                # Take everything after the prompt
                translation = full_output[match.end():].strip()
                if translation and len(translation) > 10:
                    return translation

    # Strategy 2: Split by double newline and take the last non-empty part
    if '\n\n' in full_output:
        parts = [p.strip() for p in full_output.split('\n\n') if p.strip()]
        if parts:
            # Try to find a part that looks like the target language
            for part in reversed(parts):
                if looks_like_target_language(part, target_lang):
                    return part
            # Fallback: just take the last part
            return parts[-1]

    # Strategy 3: Remove source text if detected
    # Sometimes the output contains: "Original text\n\nTranslation"
    lines = full_output.strip().split('\n')
    if len(lines) > 1:
        # Check if first line is in source language and last line is in target language
        first_line = lines[0].strip()
        last_line = lines[-1].strip()

        if looks_like_source_language(first_line, source_lang) and \
           looks_like_target_language(last_line, target_lang):
            # Take everything after the first line
            return '\n'.join(lines[1:]).strip()

    # Strategy 4: Check for colon-separated format (e.g., "Translation: xxx")
    if ':' in full_output:
        parts = full_output.split(':', 1)
        if len(parts) > 1 and len(parts[1].strip()) > 10:
            potential = parts[1].strip()
            if looks_like_target_language(potential, target_lang):
                return potential

    # Strategy 5: Fallback - return full output if it's reasonable
    cleaned = full_output.strip()
    if cleaned and len(cleaned) > 10:
        return cleaned

    # Last resort: return as is
    return full_output


def looks_like_target_language(text: str, lang_code: str) -> bool:
    """
    Heuristic to check if text looks like the target language

    Args:
        text: Text to check
        lang_code: Language code (e.g., 'zh-TW', 'en', 'ja')

    Returns:
        True if text appears to be in the target language
    """
    if not text or len(text) < 5:
        return False

    # Chinese (Traditional or Simplified)
    if lang_code.startswith('zh'):
        # Check for CJK characters
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return cjk_count > len(text) * 0.3  # At least 30% CJK characters

    # Japanese
    elif lang_code == 'ja':
        # Check for Hiragana, Katakana, or Kanji
        jp_count = sum(1 for c in text if
                      ('\u3040' <= c <= '\u309f') or  # Hiragana
                      ('\u30a0' <= c <= '\u30ff') or  # Katakana
                      ('\u4e00' <= c <= '\u9fff'))    # Kanji
        return jp_count > len(text) * 0.2

    # Korean
    elif lang_code == 'ko':
        # Check for Hangul
        kr_count = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
        return kr_count > len(text) * 0.3

    # English and other Latin-script languages
    elif lang_code == 'en' or lang_code in ['de', 'fr', 'es', 'it', 'pt']:
        # Check for ASCII characters
        ascii_count = sum(1 for c in text if ord(c) < 128)
        return ascii_count > len(text) * 0.8

    # Default: can't determine, assume true
    return True


def looks_like_source_language(text: str, lang_code: str) -> bool:
    """
    Check if text looks like the source language
    (Same logic as looks_like_target_language)
    """
    return looks_like_target_language(text, lang_code)


def extract_translation_simple(full_output: str) -> str:
    """
    Simple extraction: just take the last non-empty line
    This is the baseline strategy used in the original code

    Args:
        full_output: Full model output

    Returns:
        Extracted translation (last line)
    """
    lines = full_output.strip().split('\n')
    return lines[-1].strip() if lines else full_output.strip()


if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "name": "Case 1: Simple translation",
            "output": "這是一個測試翻譯。",
            "source": "en",
            "target": "zh-TW",
            "expected": "這是一個測試翻譯。"
        },
        {
            "name": "Case 2: With prompt",
            "output": """user
You are a professional English (en) to Chinese (zh-TW) translator.
Please translate the following text:

Hello world

這是翻譯結果。""",
            "source": "en",
            "target": "zh-TW",
            "expected": "這是翻譯結果。"
        },
        {
            "name": "Case 3: Double newline separated",
            "output": """Some English text here

這是中文翻譯
包含多行""",
            "source": "en",
            "target": "zh-TW",
            "expected": "這是中文翻譯\n包含多行"
        },
    ]

    print("Testing extraction logic...\n")
    for test in test_cases:
        result = extract_translation_v2(test["output"], test["source"], test["target"])
        success = "✅" if result.strip() == test["expected"].strip() else "❌"
        print(f"{success} {test['name']}")
        print(f"   Expected: {test['expected'][:50]}...")
        print(f"   Got:      {result[:50]}...")
        print()
