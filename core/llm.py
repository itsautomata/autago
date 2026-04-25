"""
multi-provider LLM backend.
one interface, three providers: openrouter, gemini, ollama.
"""

import os
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class LLMProvider:
    """base interface. all providers implement call()."""

    def call(self, system, messages, temperature=0.0, max_tokens=2048):
        raise NotImplementedError


class OpenRouterProvider(LLMProvider):
    """openrouter.ai: one API, any model."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model, api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

    def call(self, system, messages, temperature=0.0, max_tokens=2048):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        with httpx.Client(timeout=120) as client:
            resp = client.post(self.BASE_URL, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


class GeminiProvider(LLMProvider):
    """google gemini API: free Gemma 4 access."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model="gemma-4-31b-it", api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

    def call(self, system, messages, temperature=0.0, max_tokens=2048):
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        with httpx.Client(timeout=120) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


class OllamaProvider(LLMProvider):
    """ollama: local inference. free, runs on your machine."""

    def __init__(self, model="qwen3:8b", host="http://localhost:11434"):
        self.model = model
        self.host = host

    def call(self, system, messages, temperature=0.0, max_tokens=2048):
        url = f"{self.host}/api/chat"
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        with httpx.Client(timeout=300) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
        return data["message"]["content"]


# provider registry
PROVIDERS = {
    "openrouter": OpenRouterProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}


def create_provider(provider_name, **kwargs):
    """factory. create_provider("ollama", model="qwen3:8b")"""
    cls = PROVIDERS.get(provider_name)
    if not cls:
        raise ValueError(f"unknown provider: {provider_name}. options: {list(PROVIDERS.keys())}")
    return cls(**kwargs)


# module-level default provider (set by config)
_default_provider = None


def init(provider_name, **kwargs):
    """initialize the default provider."""
    global _default_provider
    _default_provider = create_provider(provider_name, **kwargs)
    return _default_provider


def call(system, messages, temperature=0.0, max_tokens=2048):
    """call the default provider."""
    if _default_provider is None:
        raise RuntimeError("llm not initialized. call llm.init() first.")
    return _default_provider.call(system, messages, temperature, max_tokens)


def verify():
    """verify connection works. returns True or raises."""
    call("respond with ok", [{"role": "user", "content": "ping"}], max_tokens=5)
    return True
