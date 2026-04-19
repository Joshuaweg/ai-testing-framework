from __future__ import annotations
from dataclasses import dataclass
import requests

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_SEED = 42


@dataclass
class _OllamaBackend:
    name: str
    base_url: str = OLLAMA_BASE_URL


class Model:
    def __init__(self, backend: _OllamaBackend) -> None:
        self._backend = backend

    @classmethod
    def ollama(cls, name: str, base_url: str = OLLAMA_BASE_URL) -> "Model":
        return cls(_OllamaBackend(name=name, base_url=base_url))

    @property
    def name(self) -> str:
        return self._backend.name

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        payload = {
            "model": self._backend.name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": OLLAMA_DEFAULT_SEED,
            },
        }
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
