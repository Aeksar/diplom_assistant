import os

EMBEDDINGS_MODEL = "intfloat/multilingual-e5-base"
LLM_MODELS = "mistral-large-latest"
PROMPT_MODELS = "rlm/rag-prompt"

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")