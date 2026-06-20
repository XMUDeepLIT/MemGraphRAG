from .BGE import BGELargeEnV15EmbeddingModel
from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    name_lower = embedding_model_name.lower().replace("\\", "/")
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "bge-large-en-v1.5" in name_lower:
        return BGELargeEnV15EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"
