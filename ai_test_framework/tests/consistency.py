from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

from ai_test_framework.core.base_test import BaseTest, ClaimType, Verdict, Evidence, TestResult
from ai_test_framework.core.extractors import extract


@dataclass
class FramingVariant:
    label: str
    prompt: str
    expected: str


class SurfaceFramingConsistencyTest(BaseTest):
    claim_type = ClaimType.BEHAVIORAL

    def __init__(
        self,
        variants: list[FramingVariant],
        extraction_method: str,
        expected_answer: str,
        name: Optional[str] = None,
    ) -> None:
        if len(variants) < 2:
            raise ValueError("SurfaceFramingConsistencyTest requires at least 2 variants")
        self.variants = variants
        self.extraction_method = extraction_method
        self.expected_answer = expected_answer
        self._name = name or "SurfaceFramingConsistencyTest"

    def run(self, model) -> TestResult:
        t0 = time.perf_counter()
        evidence = Evidence()
        extracted: dict[str, Optional[str]] = {}

        for v in self.variants:
            raw = model.generate(v.prompt, temperature=0.0)
            value = extract(raw, self.extraction_method)
            evidence.add(f"{v.label}_raw", raw[:120])
            evidence.add(f"{v.label}_extracted", value)
            extracted[v.label] = value

        duration = (time.perf_counter() - t0) * 1000
        unique = set(extracted.values())

        if len(unique) > 1:
            return TestResult(
                name=self._name,
                verdict=Verdict.FAIL,
                claim_type=self.claim_type,
                evidence=evidence,
                fail_mode="INCONSISTENT",
                duration_ms=duration,
            )

        answer = next(iter(unique))
        if answer != self.expected_answer:
            return TestResult(
                name=self._name,
                verdict=Verdict.FAIL,
                claim_type=self.claim_type,
                evidence=evidence,
                fail_mode="CONSISTENT_WRONG",
                duration_ms=duration,
            )

        return TestResult(
            name=self._name,
            verdict=Verdict.PASS,
            claim_type=self.claim_type,
            evidence=evidence,
            duration_ms=duration,
        )


class RepeatedPromptConsistencyTest(BaseTest):
    claim_type = ClaimType.BEHAVIORAL

    def __init__(
        self,
        prompt: str,
        extraction_method: str,
        expected_answer: str,
        n_runs: int = 3,
        name: Optional[str] = None,
    ) -> None:
        if n_runs < 2:
            raise ValueError("RepeatedPromptConsistencyTest requires n_runs >= 2")
        self.prompt = prompt
        self.extraction_method = extraction_method
        self.expected_answer = expected_answer
        self.n_runs = n_runs
        self._name = name or f"RepeatedPromptConsistencyTest(n={n_runs})"

    def run(self, model) -> TestResult:
        t0 = time.perf_counter()
        evidence = Evidence()
        results = []

        for i in range(self.n_runs):
            raw = model.generate(self.prompt, temperature=0.0)
            value = extract(raw, self.extraction_method)
            evidence.add(f"run_{i}_raw", raw[:120])
            evidence.add(f"run_{i}_extracted", value)
            results.append(value)

        duration = (time.perf_counter() - t0) * 1000
        unique = set(results)

        if len(unique) > 1:
            return TestResult(
                name=self._name,
                verdict=Verdict.FAIL,
                claim_type=self.claim_type,
                evidence=evidence,
                fail_mode="UNSTABLE",
                duration_ms=duration,
            )

        answer = next(iter(unique))
        if answer != self.expected_answer:
            return TestResult(
                name=self._name,
                verdict=Verdict.FAIL,
                claim_type=self.claim_type,
                evidence=evidence,
                fail_mode="CONSISTENT_WRONG",
                duration_ms=duration,
            )

        return TestResult(
            name=self._name,
            verdict=Verdict.PASS,
            claim_type=self.claim_type,
            evidence=evidence,
            duration_ms=duration,
        )
