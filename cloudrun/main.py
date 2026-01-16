"""
TranslateGemma FastAPI Application for Cloud Run

This module provides a REST API for TranslateGemma translation service.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TranslateGemma API",
    description="Translation API powered by Google TranslateGemma 4B",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None

class TranslationRequest(BaseModel):
    """Translation request model"""
    text: str = Field(..., description="Text to translate", min_length=1)
    target_lang: str = Field(
        default="Traditional Chinese",
        description="Target language for translation"
    )
    max_tokens: int = Field(
        default=256,
        description="Maximum number of tokens to generate",
        ge=1,
        le=512
    )

class TranslationResponse(BaseModel):
    """Translation response model"""
    original: str
    translated: str
    target_lang: str
    model_version: str = "translategemma-4b-it"

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    gpu_available: bool

@app.on_event("startup")
async def load_model():
    """Load the TranslateGemma model on startup"""
    global model, tokenizer

    try:
        logger.info("Loading TranslateGemma model...")

        # Authenticate with Hugging Face if token is provided
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("Authenticated with Hugging Face")
        else:
            logger.warning("HF_TOKEN not found. Make sure model access is public or token is set.")

        MODEL_ID = os.getenv("MODEL_ID", "google/translategemma-4b-it")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        logger.info("Tokenizer loaded successfully")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        logger.info(f"Model loaded successfully on device: {model.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TranslateGemma API",
        "version": "1.0.0",
        "endpoints": {
            "translate": "/translate",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available()
    )

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate text to target language

    Args:
        request: TranslationRequest containing text and target language

    Returns:
        TranslationResponse with original and translated text
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    try:
        logger.info(f"Translating to {request.target_lang}: {request.text[:50]}...")

        # Construct the translation prompt
        messages = [{
            "role": "user",
            "content": f"Translate this to {request.target_lang}: {request.text}"
        }]

        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)

        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=request.max_tokens,
                do_sample=False,  # Use greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the result
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the translation (remove prompt)
        if "Translate this to" in result:
            # The translation is typically on the last line
            result = result.split("\n")[-1].strip()

        logger.info(f"Translation completed: {result[:50]}...")

        return TranslationResponse(
            original=request.text,
            translated=result,
            target_lang=request.target_lang
        )

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
