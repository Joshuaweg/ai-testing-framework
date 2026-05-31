from __future__ import annotations
import time
from typing import Optional

from ai_test_framework.core.base_test import BaseTest, ClaimType, Verdict, Evidence, TestResult
from ai_test_framework.core.extractors import extract


class WhatIfConsistencyTest(BaseTest):
    """Tests that the model's output changes when a causally relevant input feature is perturbed.

    Provides a base prompt with a known expected answer and a perturbed prompt where one
    causally critical feature has changed to produce a different expected answer.

    ClaimType: CAUSAL
    Covers: feature_perturbation_accuracy (Q6 — What If)

    Fail modes:
      BASE_WRONG      — base prompt produced the wrong answer
      INSENSITIVE     — model returned the same answer for both prompts
      PERTURBED_WRONG — model changed its answer but not to the expected perturbed value
    """

    claim_type = ClaimType.CAUSAL

    def __init__(
        self,
        base_prompt: str,
        perturbed_prompt: str,
        extraction_method: str,
        base_expected: str,
        perturbed_expected: str,
        name: Optional[str] = None,
    ) -> None:
        self.base_prompt = base_prompt
        self.perturbed_prompt = perturbed_prompt
        self.extraction_method = extraction_method
        self.base_expected = base_expected
        self.perturbed_expected = perturbed_expected
        self._name = name or "WhatIfConsistencyTest"

    def run(self, model) -> TestResult:
        t0 = time.perf_counter()
        evidence = Evidence()

        base_raw = model.generate(self.base_prompt, temperature=0.0)
        base_val = extract(base_raw, self.extraction_method)
        evidence.add("base_raw", base_raw[:120])
        evidence.add("base_extracted", base_val)
        evidence.add("base_expected", self.base_expected)

        perturbed_raw = model.generate(self.perturbed_prompt, temperature=0.0)
        perturbed_val = extract(perturbed_raw, self.extraction_method)
        evidence.add("perturbed_raw", perturbed_raw[:120])
        evidence.add("perturbed_extracted", perturbed_val)
        evidence.add("perturbed_expected", self.perturbed_expected)

        duration = (time.perf_counter() - t0) * 1000

        if base_val != self.base_expected:
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"BASE_WRONG (extracted='{base_val}', expected='{self.base_expected}')",
                duration_ms=duration,
            )

        if base_val == perturbed_val:
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"INSENSITIVE (both prompts returned '{base_val}')",
                duration_ms=duration,
            )

        if perturbed_val != self.perturbed_expected:
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"PERTURBED_WRONG (extracted='{perturbed_val}', expected='{self.perturbed_expected}')",
                duration_ms=duration,
            )

        return TestResult(
            name=self._name, verdict=Verdict.PASS, claim_type=self.claim_type,
            evidence=evidence, duration_ms=duration,
        )


class InvariantExplanationTest(BaseTest):
    """Tests that the model's output is stable when causally irrelevant features change.

    Provides a base prompt and variant prompts where only irrelevant details differ.
    All variants should produce the same answer as the base.

    ClaimType: CAUSAL
    Covers: irrelevant_fact_sensitivity (Q5 — How to Still Be This)

    Fail modes:
      BASE_WRONG           — base prompt produced the wrong answer
      SPURIOUS_SENSITIVITY — answer changed when only irrelevant features changed
    """

    claim_type = ClaimType.CAUSAL

    def __init__(
        self,
        base_prompt: str,
        irrelevant_variants: list[tuple[str, str]],
        extraction_method: str,
        expected_answer: str,
        name: Optional[str] = None,
    ) -> None:
        if not irrelevant_variants:
            raise ValueError("InvariantExplanationTest requires at least 1 irrelevant_variant")
        self.base_prompt = base_prompt
        self.irrelevant_variants = irrelevant_variants  # list of (label, prompt)
        self.extraction_method = extraction_method
        self.expected_answer = expected_answer
        self._name = name or "InvariantExplanationTest"

    def run(self, model) -> TestResult:
        t0 = time.perf_counter()
        evidence = Evidence()

        base_raw = model.generate(self.base_prompt, temperature=0.0)
        base_val = extract(base_raw, self.extraction_method)
        evidence.add("base_raw", base_raw[:120])
        evidence.add("base_extracted", base_val)

        if base_val != self.expected_answer:
            duration = (time.perf_counter() - t0) * 1000
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"BASE_WRONG (extracted='{base_val}', expected='{self.expected_answer}')",
                duration_ms=duration,
            )

        spurious: list[str] = []
        for label, prompt in self.irrelevant_variants:
            raw = model.generate(prompt, temperature=0.0)
            val = extract(raw, self.extraction_method)
            evidence.add(f"{label}_raw", raw[:120])
            evidence.add(f"{label}_extracted", val)
            if val != self.expected_answer:
                spurious.append(f"{label}:'{val}'")

        duration = (time.perf_counter() - t0) * 1000

        if spurious:
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"SPURIOUS_SENSITIVITY ({', '.join(spurious)})",
                duration_ms=duration,
            )

        return TestResult(
            name=self._name, verdict=Verdict.PASS, claim_type=self.claim_type,
            evidence=evidence, duration_ms=duration,
        )


class CounterfactualValidityTest(BaseTest):
    """Tests that a minimal change sufficient to flip the outcome actually does so.

    Given a base case producing outcome A and a minimally modified case expected to
    produce outcome B, verifies both outcomes are correctly produced. Exercises whether
    the model is sensitive to the causal minimum needed to change a prediction.

    ClaimType: CAUSAL
    Covers: counterfactual_sensitivity, counterfactual_actionability (Q4 — How to Be That)

    Fail modes:
      BASE_WRONG             — base prompt produced the wrong answer
      COUNTERFACTUAL_INVALID — minimal change did not produce the expected different outcome
    """

    claim_type = ClaimType.CAUSAL

    def __init__(
        self,
        base_prompt: str,
        counterfactual_prompt: str,
        extraction_method: str,
        base_expected: str,
        counterfactual_expected: str,
        name: Optional[str] = None,
    ) -> None:
        self.base_prompt = base_prompt
        self.counterfactual_prompt = counterfactual_prompt
        self.extraction_method = extraction_method
        self.base_expected = base_expected
        self.counterfactual_expected = counterfactual_expected
        self._name = name or "CounterfactualValidityTest"

    def run(self, model) -> TestResult:
        t0 = time.perf_counter()
        evidence = Evidence()

        base_raw = model.generate(self.base_prompt, temperature=0.0)
        base_val = extract(base_raw, self.extraction_method)
        evidence.add("base_raw", base_raw[:120])
        evidence.add("base_extracted", base_val)
        evidence.add("base_expected", self.base_expected)

        cf_raw = model.generate(self.counterfactual_prompt, temperature=0.0)
        cf_val = extract(cf_raw, self.extraction_method)
        evidence.add("counterfactual_raw", cf_raw[:120])
        evidence.add("counterfactual_extracted", cf_val)
        evidence.add("counterfactual_expected", self.counterfactual_expected)

        duration = (time.perf_counter() - t0) * 1000

        if base_val != self.base_expected:
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"BASE_WRONG (extracted='{base_val}', expected='{self.base_expected}')",
                duration_ms=duration,
            )

        if cf_val != self.counterfactual_expected:
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"COUNTERFACTUAL_INVALID (extracted='{cf_val}', expected='{self.counterfactual_expected}')",
                duration_ms=duration,
            )

        return TestResult(
            name=self._name, verdict=Verdict.PASS, claim_type=self.claim_type,
            evidence=evidence, duration_ms=duration,
        )
