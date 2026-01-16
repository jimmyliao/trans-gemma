"""
Verify Hugging Face Token and Model Access

This script tests if your HF_TOKEN can access the TranslateGemma model.
"""

import os
import sys

def verify_token(token: str = None):
    """
    Verify Hugging Face token and model access

    Args:
        token: HF token (if None, will read from HF_TOKEN env var)
    """
    from huggingface_hub import login, HfApi
    from transformers import AutoTokenizer

    # Get token
    if token is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            print("❌ No token provided and HF_TOKEN env var not set")
            return False

    try:
        # Test 1: Login
        print("Test 1: Authenticating with Hugging Face...")
        login(token=token)
        print("✅ Authentication successful")

        # Test 2: Check API access
        print("\nTest 2: Checking API access...")
        api = HfApi()
        user_info = api.whoami(token=token)
        print(f"✅ Logged in as: {user_info['name']}")

        # Test 3: Check model access
        print("\nTest 3: Checking TranslateGemma model access...")
        MODEL_ID = "google/translategemma-4b-it"

        try:
            # Try to get model info
            model_info = api.model_info(MODEL_ID, token=token)
            print(f"✅ Can access model: {MODEL_ID}")
            print(f"   Model tags: {', '.join(model_info.tags[:5])}")

            # Test 4: Try loading tokenizer
            print("\nTest 4: Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
            print(f"✅ Tokenizer loaded successfully")
            print(f"   Vocab size: {len(tokenizer)}")

            print("\n" + "="*60)
            print("🎉 All tests passed! Your token is valid and has access to TranslateGemma.")
            print("="*60)
            return True

        except Exception as e:
            print(f"❌ Cannot access model: {e}")
            print("\nPossible reasons:")
            print("1. You haven't requested access to the model yet")
            print("2. Your access request is pending approval")
            print(f"3. Visit: https://huggingface.co/{MODEL_ID}")
            return False

    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("\nPossible reasons:")
        print("1. Invalid token")
        print("2. Token has been revoked")
        print("3. Network connectivity issues")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Hugging Face Token Verification")
    print("="*60)
    print()

    # Check if token provided as argument
    if len(sys.argv) > 1:
        token = sys.argv[1]
        print("Using token from command line argument")
    else:
        token = os.getenv("HF_TOKEN")
        if token:
            print("Using token from HF_TOKEN environment variable")
        else:
            print("No token found. Please provide token:")
            print("  Option 1: export HF_TOKEN=your_token")
            print("  Option 2: python verify-hf-token.py your_token")
            sys.exit(1)

    print()
    success = verify_token(token)
    sys.exit(0 if success else 1)
