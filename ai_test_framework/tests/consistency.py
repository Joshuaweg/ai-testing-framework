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


class StatisticalConsistencyTest(BaseTest):
    """Runs a prompt N times at a given temperature; passes when consistency_rate >= threshold.

    Addresses the non-determinism gap: behavioral stability at real operating temperatures
    rather than the forced-determinism of temperature=0 tests.

    ClaimType: BEHAVIORAL
    Claim: "This model answers this prompt consistently ≥ {threshold*100}% of the time at
            temperature={temperature}."
    """

    claim_type = ClaimType.BEHAVIORAL

    def __init__(
        self,
        prompt: str,
        extraction_method: str,
        expected_answer: str,
        n_runs: int = 10,
        threshold: float = 0.8,
        temperature: float = 0.7,
        name: Optional[str] = None,
    ) -> None:
        if n_runs < 3:
            raise ValueError("StatisticalConsistencyTest requires n_runs >= 3")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.prompt = prompt
        self.extraction_method = extraction_method
        self.expected_answer = expected_answer
        self.n_runs = n_runs
        self.threshold = threshold
        self.temperature = temperature
        self._name = name or f"StatisticalConsistencyTest(n={n_runs},t={temperature},thresh={threshold})"

    def run(self, model) -> TestResult:
        t0 = time.perf_counter()
        evidence = Evidence()
        results: list[Optional[str]] = []

        for i in range(self.n_runs):
            raw = model.generate(self.prompt, temperature=self.temperature)
            value = extract(raw, self.extraction_method)
            evidence.add(f"run_{i}_extracted", value)
            results.append(value)

        duration = (time.perf_counter() - t0) * 1000

        # modal answer = most common extracted value
        counts: dict[Optional[str], int] = {}
        for v in results:
            counts[v] = counts.get(v, 0) + 1
        modal_answer = max(counts, key=lambda k: counts[k])

        consistency_rate = counts[modal_answer] / self.n_runs
        correct_rate = counts.get(self.expected_answer, 0) / self.n_runs

        evidence.add("n_runs", self.n_runs)
        evidence.add("temperature", self.temperature)
        evidence.add("threshold", self.threshold)
        evidence.add("modal_answer", modal_answer)
        evidence.add("consistency_rate", round(consistency_rate, 3))
        evidence.add("correct_rate", round(correct_rate, 3))

        if consistency_rate < self.threshold:
            return TestResult(
                name=self._name,
                verdict=Verdict.FAIL,
                claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"INCONSISTENT (rate={consistency_rate:.0%} < threshold={self.threshold:.0%})",
                duration_ms=duration,
            )

        if modal_answer != self.expected_answer:
            return TestResult(
                name=self._name,
                verdict=Verdict.FAIL,
                claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"CONSISTENT_WRONG (modal='{modal_answer}', expected='{self.expected_answer}')",
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
