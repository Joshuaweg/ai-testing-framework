from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import requests

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_SEED = 42


@dataclass
class ModelConfig:
    """Full configuration snapshot for a model run.

    Passed to Model.from_config() or ComparisonRunner.compare() to capture
    every parameter that could affect output — useful for regression comparisons.
    """
    model: str
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    seed: int = OLLAMA_DEFAULT_SEED
    num_ctx: Optional[int] = None
    system_prompt: Optional[str] = None

    def label(self) -> str:
        parts = [self.model]
        if self.top_k is not None:
            parts.append(f"top_k={self.top_k}")
        if self.top_p is not None:
            parts.append(f"top_p={self.top_p}")
        if self.repeat_penalty is not None:
            parts.append(f"rp={self.repeat_penalty}")
        if self.num_ctx is not None:
            parts.append(f"ctx={self.num_ctx}")
        if self.system_prompt is not None:
            parts.append("sys=custom")
        return " | ".join(parts)


@dataclass
class _OllamaBackend:
    name: str
    base_url: str = OLLAMA_BASE_URL


class Model:
    def __init__(self, backend: _OllamaBackend, config: Optional[ModelConfig] = None) -> None:
        self._backend = backend
        self._config = config

    @classmethod
    def ollama(cls, name: str, base_url: str = OLLAMA_BASE_URL) -> "Model":
        return cls(_OllamaBackend(name=name, base_url=base_url))

    @classmethod
    def from_config(cls, config: ModelConfig, base_url: str = OLLAMA_BASE_URL) -> "Model":
        return cls(_OllamaBackend(name=config.model, base_url=base_url), config=config)

    @property
    def name(self) -> str:
        return self._backend.name

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        options: dict = {
            "temperature": temperature,
            "seed": OLLAMA_DEFAULT_SEED,
        }
        if self._config is not None:
            options["seed"] = self._config.seed
            if self._config.top_k is not None:
                options["top_k"] = self._config.top_k
            if self._config.top_p is not None:
                options["top_p"] = self._config.top_p
            if self._config.repeat_penalty is not None:
                options["repeat_penalty"] = self._config.repeat_penalty
            if self._config.num_ctx is not None:
                options["num_ctx"] = self._config.num_ctx

        payload: dict = {
            "model": self._backend.name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if self._config is not None and self._config.system_prompt is not None:
            payload["system"] = self._config.system_prompt

        resp = requests.post(
            f"{self._backend.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if "response" not in data:
            raise RuntimeError(f"Ollama response missing 'response' key: {data}")
        return data["response"]
