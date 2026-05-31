from __future__ import annotations
import time
from typing import Optional

from ai_test_framework.core.base_test import BaseTest, ClaimType, Verdict, Evidence, TestResult
from ai_test_framework.core.extractors import extract


class DataProvenanceConsistencyTest(BaseTest):
    """Tests that the model gives consistent answers about its training data across phrasings.

    Since training data claims cannot be verified model-agnostically, this test checks
    only consistency — the model should give the same extracted answer regardless of
    how the data question is phrased.

    ClaimType: BEHAVIORAL
    Covers: data_provenance_consistency (Q8 — Data)

    Fail modes:
      DATA_CLAIM_INCONSISTENT — model gives different answers across data-query phrasings
    """

    claim_type = ClaimType.BEHAVIORAL

    def __init__(
        self,
        data_query_variants: list[tuple[str, str]],
        extraction_method: str,
        name: Optional[str] = None,
    ) -> None:
        if len(data_query_variants) < 2:
            raise ValueError("DataProvenanceConsistencyTest requires at least 2 variants")
        self.data_query_variants = data_query_variants  # list of (label, prompt)
        self.extraction_method = extraction_method
        self._name = name or "DataProvenanceConsistencyTest"

    def run(self, model) -> TestResult:
        t0 = time.perf_counter()
        evidence = Evidence()
        extracted: dict[str, Optional[str]] = {}

        for label, prompt in self.data_query_variants:
            raw = model.generate(prompt, temperature=0.0)
            val = extract(raw, self.extraction_method)
            evidence.add(f"{label}_raw", raw[:120])
            evidence.add(f"{label}_extracted", val)
            extracted[label] = val

        duration = (time.perf_counter() - t0) * 1000
        unique = set(extracted.values())

        if len(unique) > 1:
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence, fail_mode="DATA_CLAIM_INCONSISTENT",
                duration_ms=duration,
            )

        return TestResult(
            name=self._name, verdict=Verdict.PASS, claim_type=self.claim_type,
            evidence=evidence, duration_ms=duration,
        )


class OutputScopeConsistencyTest(BaseTest):
    """Tests that the model's claimed output capabilities match its actual behavior.

    Phase 1: Ask N capability-claim prompts → extract yes/no → check consistency.
    Phase 2: Run the actual task → verify behavior matches the claimed capability.

    ClaimType: BEHAVIORAL
    Covers: output_scope_accuracy (Q9 — Output)

    Fail modes:
      SCOPE_INCONSISTENT — capability claims disagree across phrasings
      SCOPE_OVERCLAIM    — model claims it can do X but fails the task
      SCOPE_UNDERCLAIM   — model claims it cannot do X but succeeds at the task
    """

    claim_type = ClaimType.BEHAVIORAL

    def __init__(
        self,
        capability_claim_prompts: list[tuple[str, str]],
        task_prompt: str,
        task_extraction_method: str,
        task_expected: str,
        expected_capability: str,
        name: Optional[str] = None,
    ) -> None:
        # expected_capability: "yes" or "no"
        if not capability_claim_prompts:
            raise ValueError("OutputScopeConsistencyTest requires at least 1 capability_claim_prompt")
        self.capability_claim_prompts = capability_claim_prompts
        self.task_prompt = task_prompt
        self.task_extraction_method = task_extraction_method
        self.task_expected = task_expected
        self.expected_capability = expected_capability
        self._name = name or "OutputScopeConsistencyTest"

    def run(self, model) -> TestResult:
        t0 = time.perf_counter()
        evidence = Evidence()
        claims: dict[str, Optional[str]] = {}

        for label, prompt in self.capability_claim_prompts:
            raw = model.generate(prompt, temperature=0.0)
            val = extract(raw, "yn")
            evidence.add(f"{label}_claim_raw", raw[:120])
            evidence.add(f"{label}_claim", val)
            claims[label] = val

        claim_values = set(claims.values())

        if len(claim_values) > 1:
            duration = (time.perf_counter() - t0) * 1000
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence, fail_mode="SCOPE_INCONSISTENT",
                duration_ms=duration,
            )

        task_raw = model.generate(self.task_prompt, temperature=0.0)
        task_val = extract(task_raw, self.task_extraction_method)
        evidence.add("task_raw", task_raw[:120])
        evidence.add("task_extracted", task_val)
        evidence.add("task_expected", self.task_expected)

        duration = (time.perf_counter() - t0) * 1000
        task_succeeded = task_val == self.task_expected
        claimed = next(iter(claim_values), None)

        if claimed == "yes" and not task_succeeded:
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"SCOPE_OVERCLAIM (claimed capable but task failed: extracted='{task_val}', expected='{self.task_expected}')",
                duration_ms=duration,
            )

        if claimed == "no" and task_succeeded:
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode="SCOPE_UNDERCLAIM (claimed incapable but task succeeded)",
                duration_ms=duration,
            )

        return TestResult(
            name=self._name, verdict=Verdict.PASS, claim_type=self.claim_type,
            evidence=evidence, duration_ms=duration,
        )
