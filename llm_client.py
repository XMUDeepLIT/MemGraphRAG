#!/usr/bin/env python3
"""
可复用的LLM客户端模块
功能：
1. 支持OpenAI兼容的API调用
2. 支持SQLite缓存以避免重复调用
3. 支持并行批量调用
4. 支持JSON格式输出解析
"""

import json
import hashlib
import os
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, TypeVar
from pathlib import Path
import logging

import httpx
from openai import OpenAI
from filelock import FileLock

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM配置"""
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.0
    seed: int = 42
    cache_dir: str = "./cache"
    max_workers: int = 10
    timeout: float = 300.0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")


class LLMClient:
    """LLM客户端，支持缓存和并行调用"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化LLM客户端
        
        Args:
            config: LLM配置，为None时使用默认配置
        """
        self.config = config or LLMConfig()
        
        # 创建缓存目录
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存文件路径
        safe_model_name = self.config.model_name.replace("/", "_").replace(":", "_")
        self.cache_file = self.cache_dir / f"{safe_model_name}_cache.sqlite"
        self.lock_file = str(self.cache_file) + ".lock"
        
        # 初始化缓存数据库
        self._init_cache_db()
        
        # 初始化OpenAI客户端
        limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
        client = httpx.Client(
            limits=limits, 
            timeout=httpx.Timeout(self.config.timeout, read=self.config.timeout)
        )
        
        self.openai_client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            http_client=client
        )
        
        logger.info(f"LLM客户端初始化完成: model={self.config.model_name}")
    
    def _init_cache_db(self):
        """初始化缓存数据库"""
        with FileLock(self.lock_file):
            conn = sqlite3.connect(str(self.cache_file))
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
    
    def _get_cache_key(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """生成缓存键"""
        key_data = {
            "messages": messages,
            "model": self.config.model_name,
            "seed": kwargs.get("seed", self.config.seed),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[tuple]:
        """从缓存获取结果"""
        with FileLock(self.lock_file):
            conn = sqlite3.connect(str(self.cache_file))
            c = conn.cursor()
            c.execute("SELECT response, metadata FROM cache WHERE key = ?", (key,))
            row = c.fetchone()
            conn.close()
            if row:
                return row[0], json.loads(row[1])
            return None
    
    def _save_to_cache(self, key: str, response: str, metadata: dict):
        """保存结果到缓存"""
        with FileLock(self.lock_file):
            conn = sqlite3.connect(str(self.cache_file))
            c = conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO cache (key, response, metadata) VALUES (?, ?, ?)",
                (key, response, json.dumps(metadata))
            )
            conn.commit()
            conn.close()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        use_cache: bool = True,
        **kwargs
    ) -> tuple[str, dict, bool]:
        """
        发送聊天请求
        
        Args:
            messages: 聊天消息列表，格式为 [{"role": "user", "content": "..."}]
            use_cache: 是否使用缓存
            **kwargs: 其他参数，如temperature, max_tokens等
            
        Returns:
            (response_text, metadata, cache_hit)
        """
        # 检查缓存
        cache_key = self._get_cache_key(messages, **kwargs)
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.debug(f"缓存命中: {cache_key[:16]}...")
                return cached[0], cached[1], True
        
        # 调用API
        params = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "seed": kwargs.get("seed", self.config.seed),
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.openai_client.chat.completions.create(**params)
                response_text = response.choices[0].message.content
                
                metadata = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": response.choices[0].finish_reason,
                    "model": self.config.model_name,
                }
                
                # 保存到缓存
                if use_cache:
                    self._save_to_cache(cache_key, response_text, metadata)
                
                return response_text, metadata, False
                
            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        raise RuntimeError("API调用失败")
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        use_cache: bool = True,
        **kwargs
    ) -> tuple[Any, dict, bool]:
        """
        发送聊天请求并解析JSON响应
        
        Args:
            messages: 聊天消息列表
            use_cache: 是否使用缓存
            **kwargs: 其他参数
            
        Returns:
            (parsed_json, metadata, cache_hit)
        """
        response_text, metadata, cache_hit = self.chat(messages, use_cache, **kwargs)
        
        # 尝试解析JSON
        try:
            # 处理可能的markdown代码块
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            parsed = json.loads(text.strip())
            return parsed, metadata, cache_hit
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"原始响应: {response_text[:500]}...")
            raise ValueError(f"无法解析JSON响应: {e}")
    
    def batch_chat(
        self,
        batch_messages: List[List[Dict[str, str]]],
        use_cache: bool = True,
        show_progress: bool = True,
        **kwargs
    ) -> List[tuple[str, dict, bool]]:
        """
        并行批量聊天请求
        
        Args:
            batch_messages: 批量消息列表
            use_cache: 是否使用缓存
            show_progress: 是否显示进度
            **kwargs: 其他参数
            
        Returns:
            结果列表，每个元素为 (response_text, metadata, cache_hit)
        """
        results = [None] * len(batch_messages)
        
        def process_one(idx: int, messages: List[Dict[str, str]]) -> tuple[int, Any]:
            try:
                result = self.chat(messages, use_cache, **kwargs)
                return idx, result
            except Exception as e:
                logger.error(f"批量处理索引 {idx} 失败: {e}")
                return idx, (None, {"error": str(e)}, False)
        
        completed = 0
        total = len(batch_messages)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(process_one, i, msgs): i 
                for i, msgs in enumerate(batch_messages)
            }
            
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                
                if show_progress and completed % 10 == 0:
                    logger.info(f"进度: {completed}/{total} ({100*completed/total:.1f}%)")
        
        if show_progress:
            logger.info(f"批量处理完成: {total} 个请求")
        
        return results
    
    def batch_chat_json(
        self,
        batch_messages: List[List[Dict[str, str]]],
        use_cache: bool = True,
        show_progress: bool = True,
        **kwargs
    ) -> List[tuple[Any, dict, bool]]:
        """
        并行批量聊天请求并解析JSON
        
        Args:
            batch_messages: 批量消息列表
            use_cache: 是否使用缓存
            show_progress: 是否显示进度
            **kwargs: 其他参数
            
        Returns:
            结果列表，每个元素为 (parsed_json, metadata, cache_hit)
        """
        results = [None] * len(batch_messages)
        
        def process_one(idx: int, messages: List[Dict[str, str]]) -> tuple[int, Any]:
            try:
                result = self.chat_json(messages, use_cache, **kwargs)
                return idx, result
            except Exception as e:
                logger.error(f"批量处理索引 {idx} 失败: {e}")
                return idx, (None, {"error": str(e)}, False)
        
        completed = 0
        total = len(batch_messages)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(process_one, i, msgs): i 
                for i, msgs in enumerate(batch_messages)
            }
            
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                
                if show_progress and completed % 10 == 0:
                    logger.info(f"进度: {completed}/{total} ({100*completed/total:.1f}%)")
        
        if show_progress:
            logger.info(f"批量处理完成: {total} 个请求")
        
        return results


# 便捷函数
_default_client: Optional[LLMClient] = None


def get_default_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """获取默认LLM客户端（单例模式）"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient(config)
    return _default_client


def chat(
    messages: List[Dict[str, str]],
    config: Optional[LLMConfig] = None,
    **kwargs
) -> tuple[str, dict, bool]:
    """便捷函数：发送聊天请求"""
    client = get_default_client(config)
    return client.chat(messages, **kwargs)


def chat_json(
    messages: List[Dict[str, str]],
    config: Optional[LLMConfig] = None,
    **kwargs
) -> tuple[Any, dict, bool]:
    """便捷函数：发送聊天请求并解析JSON"""
    client = get_default_client(config)
    return client.chat_json(messages, **kwargs)


if __name__ == "__main__":
    # 测试示例
    config = LLMConfig(
        model_name="gpt-4o-mini",
        temperature=0.0,
    )
    
    client = LLMClient(config)
    
    messages = [
        {"role": "user", "content": "Say 'Hello, World!' in JSON format: {\"greeting\": \"...\"}"}
    ]
    
    response, metadata, cache_hit = client.chat_json(messages)
    print(f"Response: {response}")
    print(f"Metadata: {metadata}")
    print(f"Cache hit: {cache_hit}")
