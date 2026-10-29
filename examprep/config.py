"""Central configuration. Values come from environment variables / .env."""
import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # gemini | ollama | openai_compat | anthropic | mock
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")  # used by anthropic / ollama / openai_compat

# Free option 1: Google Gemini (free tier via Google AI Studio)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Free option 2: any OpenAI-compatible server (Ollama on your laptop, Groq, OpenRouter...)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").lower()  # auto | sbert | tfidf
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))      # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters
TOP_K = int(os.getenv("TOP_K", "6"))
