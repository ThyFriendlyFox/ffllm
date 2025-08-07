from __future__ import annotations

import os
from typing import List, Dict, Optional

import requests

# Optional imports: these may fail if users don't want all providers
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore

try:
    from mistralai import Mistral  # type: ignore
except Exception:  # pragma: no cover
    Mistral = None  # type: ignore


RoleMessage = Dict[str, str]


class LLMChat:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or self._default_api_key()
        self._clients = {}
        self._init_clients()

    def _default_api_key(self) -> Optional[str]:
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "ollama": None,
        }
        env = env_map.get(self.provider)
        return os.environ.get(env) if env else None

    def _init_clients(self) -> None:
        if self.provider == "openai":
            if OpenAI is not None:
                self._clients["openai"] = OpenAI(base_url=self.base_url, api_key=self.api_key)
        elif self.provider == "anthropic" and anthropic is not None:
            self._clients["anthropic"] = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "google" and genai is not None:
            if self.api_key:
                genai.configure(api_key=self.api_key)
            self._clients["google"] = genai
        elif self.provider == "mistral" and Mistral is not None:
            self._clients["mistral"] = Mistral(api_key=self.api_key)
        elif self.provider == "ollama":
            # Uses HTTP to OLLAMA_HOST or default localhost
            pass

    def complete(self, messages: List[RoleMessage], temperature: float = 0.2, max_tokens: int = 512) -> str:
        if self.provider == "openai":
            return self._openai(messages, temperature, max_tokens)
        if self.provider == "anthropic":
            return self._anthropic(messages, temperature, max_tokens)
        if self.provider == "google":
            return self._google(messages, temperature, max_tokens)
        if self.provider == "mistral":
            return self._mistral(messages, temperature, max_tokens)
        if self.provider == "ollama":
            return self._ollama(messages, temperature, max_tokens)
        raise ValueError(f"Unknown provider: {self.provider}")

    # Provider implementations
    def _openai(self, messages: List[RoleMessage], temperature: float, max_tokens: int) -> str:
        if OpenAI is None:
            # Fallback to raw HTTP OpenAI-compatible endpoint
            url = (self.base_url or "https://api.openai.com/v1") + "/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            resp = requests.post(url, json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        client = self._clients.get("openai")
        chat = client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return chat.choices[0].message.content or ""

    def _anthropic(self, messages: List[RoleMessage], temperature: float, max_tokens: int) -> str:
        if anthropic is None:
            raise RuntimeError("anthropic not installed")
        client = self._clients.get("anthropic")
        system = "\n".join([m["content"] for m in messages if m["role"] == "system"]) or None
        user_content = []
        for m in messages:
            if m["role"] in ("user", "assistant"):
                user_content.append({"role": m["role"], "content": m["content"]})
        msg = client.messages.create(
            model=self.model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=user_content,
        )
        return "".join([block.text for block in msg.content if getattr(block, "text", None)])

    def _google(self, messages: List[RoleMessage], temperature: float, max_tokens: int) -> str:
        if genai is None:
            raise RuntimeError("google-generativeai not installed")
        client = self._clients.get("google")
        model = client.GenerativeModel(self.model)
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        resp = model.generate_content(prompt, generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        })
        return resp.text or ""

    def _mistral(self, messages: List[RoleMessage], temperature: float, max_tokens: int) -> str:
        if Mistral is None:
            raise RuntimeError("mistralai not installed")
        client = self._clients.get("mistral")
        resp = client.chat.complete(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content or ""

    def _ollama(self, messages: List[RoleMessage], temperature: float, max_tokens: int) -> str:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        resp = requests.post(f"{host}/api/generate", json={
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": temperature},
            "stream": False,
        }, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
