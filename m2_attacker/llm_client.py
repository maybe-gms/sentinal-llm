"""Thin Ollama client with an on-disk cache.

The cache matters more than it looks. The evolutionary loop re-evaluates
surviving elite strategies every generation against the same task pairs, so
without caching you burn most of your wall-clock time recomputing identical
prompts. With it, generation 2+ is several times faster.
"""

import hashlib
import json
import os
import time
from typing import Dict, Optional

import requests

import config


class OllamaClient:
    def __init__(self, cache_path: str = config.CACHE_PATH, use_cache: bool = True):
        self.cache_path = cache_path
        self.use_cache = use_cache
        self._cache: Dict[str, str] = {}
        self._dirty = 0
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    @staticmethod
    def _key(model: str, prompt: str, temperature: float) -> str:
        raw = f"{model}||{temperature}||{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        model = model or config.BACKEND_MODEL
        temperature = config.TEMPERATURE if temperature is None else temperature
        max_tokens = max_tokens or config.MAX_TOKENS

        key = self._key(model, prompt, temperature)
        if self.use_cache and key in self._cache:
            return self._cache[key]

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": config.SEED,
                "num_predict": max_tokens,
            },
        }

        last_err = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    config.OLLAMA_URL, json=payload, timeout=config.REQUEST_TIMEOUT
                )
                resp.raise_for_status()
                text = resp.json().get("response", "").strip()
                if self.use_cache:
                    self._cache[key] = text
                    self._dirty += 1
                    if self._dirty >= 20:
                        self.flush()
                return text
            except requests.RequestException as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(
            f"Ollama request failed after 3 attempts ({last_err}). "
            f"Is `ollama serve` running and is '{model}' pulled?"
        )

    def flush(self) -> None:
        if not self.use_cache:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)
        os.replace(tmp, self.cache_path)
        self._dirty = 0
