import os
from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)

# BGE 系列官方示例使用最后一层 hidden states + mean pooling（与 Contriever 相同公式）
def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


# BAAI/bge-large-en-v1.5 等 BERT 类 checkpoint 的有效长度上限
BGE_LARGE_EN_V15_MAX_SEQ_LEN = 512
BGE_LARGE_EN_V15_HF_ID = "BAAI/bge-large-en-v1.5"


def resolve_bge_large_en_v15_repo(model_name: str) -> str:
    """把简写 id（无组织名）解析为 HuggingFace 完整仓库名；已是本地目录则返回绝对路径。"""
    name = (model_name or "").strip()
    if not name:
        return name
    expanded = os.path.expanduser(name)
    for path in (expanded, os.path.abspath(expanded)):
        if os.path.isdir(path):
            return path
    norm = name.replace("\\", "/").lower().strip("/")
    base = norm.split("/")[-1]
    # 例如 bge-large-en-v1.5（无 BAAI/）在 Hub 上不存在，需映射到官方仓库
    if base == "bge-large-en-v1.5" and not norm.startswith("baai/"):
        return BGE_LARGE_EN_V15_HF_ID
    return name


class BGELargeEnV15EmbeddingModel(BaseEmbeddingModel):
    """bge-large-en-v1.5（及本地同名目录）：HF AutoModel + mean pooling + 可选 L2 归一化。"""

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}"
            )

        resolved = resolve_bge_large_en_v15_repo(self.embedding_model_name)
        if resolved != self.embedding_model_name:
            logger.info(
                "Resolved embedding checkpoint %r -> %r (use local folder path to force offline weights)",
                self.embedding_model_name,
                resolved,
            )
            self.embedding_model_name = resolved

        self._init_embedding_config()

        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: "
            f"{self.embedding_config.model_init_params}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(**self.embedding_config.model_init_params)
        self.embedding_model.eval()
        self.embedding_dim = self.embedding_model.config.hidden_size

    def _init_embedding_config(self) -> None:
        max_len = min(self.global_config.embedding_max_seq_len, BGE_LARGE_EN_V15_MAX_SEQ_LEN)
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                "device_map": "auto",
            },
            "encode_params": {
                "max_length": max_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32,
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str]) -> torch.Tensor:
        max_length = self.embedding_config.encode_params["max_length"]
        device = self.embedding_model.device

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = self.embedding_model(**inputs)
            embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])

        return embeddings

    def batch_encode(self, texts: List[str], **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        batch_size = params.pop("batch_size", 16)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            chunks = []
            for i in range(0, len(texts), batch_size):
                chunks.append(self.encode(texts[i : i + batch_size]))
                pbar.update(batch_size)
            pbar.close()
            results = torch.cat(chunks, dim=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()

        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
