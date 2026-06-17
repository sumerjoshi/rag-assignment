from llama_index.core import Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

def _require(name: str) -> str:
    my_val = os.getenv(name)
    if not my_val:
        raise ValueError(f"{name} is missing from your .env file")
    return my_val


# Define variables and make sure that have been set
FW_MODEL = _require("FIREWORKS_LLM_MODEL")
FW_API_KEY = _require("FIREWORKS_API_KEY")
FW_API_BASE = _require("FIREWORKS_BASE_URL")
FW_EMBED_MODEL = _require("FIREWORKS_EMBEDDING_MODEL")
DATABASE_PATH = _require("DATABASE_PATH")
PDF_DIR = _require("PDF_DIR")

# Set the _REPO_ROOT and other variables coming from SRC
_REPO_ROOT = Path(__file__).resolve().parents[1]
ABSOLUTE_DB_PATH = (_REPO_ROOT / DATABASE_PATH).resolve()
ABSOLUTE_PDF_DIR_PATH = (_REPO_ROOT / PDF_DIR).resolve()
VECTOR_STORE_DIR=os.getenv("VECTOR_STORE_DIR")

def get_llm() -> OpenAILike:
    """
    Returns the Chat LLM as an OpenAILike object to be used everywhere
    """
    llm = OpenAILike(
        model=FW_MODEL,
        api_key=FW_API_KEY,
        api_base=FW_API_BASE,
        is_chat_model=True,
        is_function_calling_model=True
    )
    return llm

def get_embed_model() -> OpenAILikeEmbedding:
    """
    Returns the Embedding Model as OpenAILikeObject to be used everywhere
    """
    embedding_model = OpenAILikeEmbedding(
        model_name=FW_EMBED_MODEL,
        api_key=FW_API_KEY,
        api_base=FW_API_BASE,
    )
    return embedding_model

def configure_settings() -> None:
    Settings.llm = get_llm()
    Settings.embed_model = get_embed_model()