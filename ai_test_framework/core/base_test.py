from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ClaimType(Enum):
    BEHAVIORAL = "behavioral"
    OBSERVATIONAL = "observational"
    CAUSAL = "causal"


class Verdict(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIP = "SKIP"


@dataclass
class Evidence:
    data: dict = field(default_factory=dict)

    def add(self, key: str, value: object) -> None:
        self.data[key] = value

    def items(self) -> list[tuple[str, object]]:
        return list(self.data.items())


@dataclass
class TestResult:
    name: str
    verdict: Verdict
    claim_type: ClaimType
    evidence: Evidence
    fail_mode: Optional[str] = None
    error_msg: Optional[str] = None
    duration_ms: float = 0.0


class BaseTest(ABC):
    claim_type: ClaimType

    @abstractmethod
    def run(self, model) -> TestResult:
        ...
