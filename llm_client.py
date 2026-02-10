#!/usr/bin/env python3
"""
Reusable LLM client module
Features:
1. Support OpenAI compatible API calls
2. Support SQLite caching to avoid repeated calls
3. Support parallel batch calls
4. Support JSON format output parsing
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM configuration"""
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
    """LLM client, supports caching and parallel calls"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client
        
        Args:
            config: LLM configuration, uses default configuration if None
        """
        self.config = config or LLMConfig()
        
        # Create cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file path
        safe_model_name = self.config.model_name.replace("/", "_").replace(":", "_")
        self.cache_file = self.cache_dir / f"{safe_model_name}_cache.sqlite"
        self.lock_file = str(self.cache_file) + ".lock"
        
        # Initialize cache database
        self._init_cache_db()
        
        # Initialize OpenAI client
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
        
        logger.info(f"LLM client initialization completed: model={self.config.model_name}")
    
    def _init_cache_db(self):
        """Initialize cache database"""
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
        """Generate cache key"""
        key_data = {
            "messages": messages,
            "model": self.config.model_name,
            "seed": kwargs.get("seed", self.config.seed),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[tuple]:
        """Get result from cache"""
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
        """Save result to cache"""
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
        Send chat request
        
        Args:
            messages: Chat message list, format: [{"role": "user", "content": "..."}]
            use_cache: Whether to use cache
            **kwargs: Other parameters, like temperature, max_tokens, etc.
            
        Returns:
            (response_text, metadata, cache_hit)
        """
        # Check cache
        cache_key = self._get_cache_key(messages, **kwargs)
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.debug(f"Cache hit: {cache_key[:16]}...")
                return cached[0], cached[1], True
        
        # Call API
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
                
                # Save to cache
                if use_cache:
                    self._save_to_cache(cache_key, response_text, metadata)
                
                return response_text, metadata, False
                
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        raise RuntimeError("API call failed")
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        use_cache: bool = True,
        **kwargs
    ) -> tuple[Any, dict, bool]:
        """
        Send chat request and parse JSON response
        
        Args:
            messages: Chat message list
            use_cache: Whether to use cache
            **kwargs: Other parameters
            
        Returns:
            (parsed_json, metadata, cache_hit)
        """
        response_text, metadata, cache_hit = self.chat(messages, use_cache, **kwargs)
        
        # Try to parse JSON
        try:
            # Handle possible markdown code blocks
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
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Original response: {response_text[:500]}...")
            raise ValueError(f"Unable to parse JSON response: {e}")
    
    def batch_chat(
        self,
        batch_messages: List[List[Dict[str, str]]],
        use_cache: bool = True,
        show_progress: bool = True,
        **kwargs
    ) -> List[tuple[str, dict, bool]]:
        """
        Parallel batch chat requests
        
        Args:
            batch_messages: Batch message list
            use_cache: Whether to use cache
            show_progress: Whether to show progress
            **kwargs: Other parameters
            
        Returns:
            Result list, each element: (response_text, metadata, cache_hit)
        """
        results = [None] * len(batch_messages)
        
        def process_one(idx: int, messages: List[Dict[str, str]]) -> tuple[int, Any]:
            try:
                result = self.chat(messages, use_cache, **kwargs)
                return idx, result
            except Exception as e:
                logger.error(f"Batch processing index {idx} failed: {e}")
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
                    logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")
        
        if show_progress:
            logger.info(f"Batch processing completed: {total} requests")
        
        return results
    
    def batch_chat_json(
        self,
        batch_messages: List[List[Dict[str, str]]],
        use_cache: bool = True,
        show_progress: bool = True,
        **kwargs
    ) -> List[tuple[Any, dict, bool]]:
        """
        Parallel batch chat requests with JSON parsing
        
        Args:
            batch_messages: Batch message list
            use_cache: Whether to use cache
            show_progress: Whether to show progress
            **kwargs: Other parameters
            
        Returns:
            Result list, each element: (parsed_json, metadata, cache_hit)
        """
        results = [None] * len(batch_messages)
        
        def process_one(idx: int, messages: List[Dict[str, str]]) -> tuple[int, Any]:
            try:
                result = self.chat_json(messages, use_cache, **kwargs)
                return idx, result
            except Exception as e:
                logger.error(f"Batch processing index {idx} failed: {e}")
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
                    logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")
        
        if show_progress:
            logger.info(f"Batch processing completed: {total} requests")
        
        return results


# Convenience functions
_default_client: Optional[LLMClient] = None


def get_default_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """Get default LLM client (singleton pattern)"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient(config)
    return _default_client


def chat(
    messages: List[Dict[str, str]],
    config: Optional[LLMConfig] = None,
    **kwargs
) -> tuple[str, dict, bool]:
    """Convenience function: send chat request"""
    client = get_default_client(config)
    return client.chat(messages, **kwargs)


def chat_json(
    messages: List[Dict[str, str]],
    config: Optional[LLMConfig] = None,
    **kwargs
) -> tuple[Any, dict, bool]:
    """Convenience function: send chat request and parse JSON"""
    client = get_default_client(config)
    return client.chat_json(messages, **kwargs)


if __name__ == "__main__":
    # Test example
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