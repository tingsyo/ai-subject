import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4")

# LLM Providers: "gemini", "openai", "anthropic", "ollama"
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "gemini")

# Model configuration mapping
DEFAULT_MODELS = {
    "gemini": os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
    "ollama": os.getenv("OLLAMA_MODEL", "gemma4")
}

# Co-Scientist Pipeline Parameters
DEFAULT_NUM_HYPOTHESES = int(os.getenv("DEFAULT_NUM_HYPOTHESES", "4"))
TOURNAMENT_ELO_K = int(os.getenv("TOURNAMENT_ELO_K", "32"))
TOURNAMENT_INITIAL_ELO = int(os.getenv("TOURNAMENT_INITIAL_ELO", "1200"))

# Proxy configuration for Scholarly
# Options: "none", "free", "custom"
PROXY_METHOD = os.getenv("PROXY_METHOD", "free")
# Comma-separated list of custom HTTP/HTTPS proxies (e.g. "http://user:pass@ip:port,http://ip2:port2")
CUSTOM_PROXIES = os.getenv("CUSTOM_PROXIES", "")
