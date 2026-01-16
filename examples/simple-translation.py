"""
Simple Translation Example using TranslateGemma API

This script demonstrates how to use the TranslateGemma API
deployed on Cloud Run.
"""

import requests
import json

# Configuration
API_URL = "https://translategemma-4b-xxxxx-uc.a.run.app"  # Replace with your Cloud Run URL

def translate(text: str, target_lang: str = "Traditional Chinese") -> dict:
    """
    Translate text using TranslateGemma API

    Args:
        text: Text to translate
        target_lang: Target language (default: Traditional Chinese)

    Returns:
        Translation response as dictionary
    """
    endpoint = f"{API_URL}/translate"

    payload = {
        "text": text,
        "target_lang": target_lang,
        "max_tokens": 256
    }

    response = requests.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

def main():
    """Main function with examples"""

    print("TranslateGemma API - Simple Translation Examples")
    print("=" * 60)

    # Example 1: English to Traditional Chinese
    print("\n1. English → Traditional Chinese")
    result = translate(
        "Hello, how are you?",
        "Traditional Chinese"
    )
    print(f"Original: {result['original']}")
    print(f"Translated: {result['translated']}")

    # Example 2: English to Japanese
    print("\n2. English → Japanese")
    result = translate(
        "Machine learning is transforming the world.",
        "Japanese"
    )
    print(f"Original: {result['original']}")
    print(f"Translated: {result['translated']}")

    # Example 3: Chinese to English
    print("\n3. Traditional Chinese → English")
    result = translate(
        "我喜歡在週末閱讀和寫程式。",
        "English"
    )
    print(f"Original: {result['original']}")
    print(f"Translated: {result['translated']}")

    # Example 4: Longer text
    print("\n4. Longer Text Translation")
    long_text = """
    Artificial intelligence is rapidly advancing and changing
    how we live and work. From natural language processing to
    computer vision, AI technologies are becoming more powerful
    and accessible every day.
    """
    result = translate(long_text.strip(), "Traditional Chinese")
    print(f"Original: {result['original'][:50]}...")
    print(f"Translated: {result['translated'][:50]}...")

    print("\n" + "=" * 60)
    print("Examples completed successfully!")

if __name__ == "__main__":
    # Check if API_URL is configured
    if "xxxxx" in API_URL:
        print("⚠️  Please update API_URL with your Cloud Run service URL")
        print("You can find it after deployment or by running:")
        print("gcloud run services describe translategemma-4b --region=us-central1 --format='value(status.url)'")
    else:
        main()
