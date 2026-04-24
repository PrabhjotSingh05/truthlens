"""
config.py - Configuration & Environment Variable Handling
=========================================================
Switched from local Ollama to Groq Cloud API.
Groq is free, ultra-fast, and works perfectly in Google Colab.
Get your free API key at: https://console.groq.com
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration class for TrustScanner AI."""

    # --- Flask Settings ---
    SECRET_KEY = os.getenv("SECRET_KEY", "trustscanner-default-secret-key")
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    # --- Groq Cloud AI Settings ---
    # Groq is a free, blazing-fast cloud API that runs Llama 3 models.
    # It is the replacement for local Ollama and works in Google Colab.
    # Get your free API key at https://console.groq.com
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL = "llama-3.3-70b-versatile"  # Latest flagship Llama 3 model on Groq

    # --- Scraper Settings ---
    REQUEST_TIMEOUT = 12  # seconds to wait for a webpage to respond
    MAX_REVIEWS = 30      # reduced for faster processing
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # --- Analysis Settings ---
    TRUST_SCORE_DEFAULT = None

    @classmethod
    def is_ai_enabled(cls):
        """Check if Groq AI is available (API key is set)."""
        return bool(cls.GROQ_API_KEY and cls.GROQ_API_KEY.strip())

    @classmethod
    def get_status_message(cls):
        """Returns a user-friendly status message."""
        if cls.is_ai_enabled():
            return "AI Analysis Active"
        return "Basic Analysis Mode - Add GROQ_API_KEY for full AI features"
