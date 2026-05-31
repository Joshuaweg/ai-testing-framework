from __future__ import annotations
import time
from typing import Optional

from ai_test_framework.core.base_test import BaseTest, ClaimType, Verdict, Evidence, TestResult
from ai_test_framework.core.extractors import extract

_REDERIVE_SUFFIX: dict[str, str] = {
    "numeric": "Reply with just the number.",
    "yn":      "Reply with just yes or no.",
    "letter":  "Reply with just the letter (A, B, C, or D).",
    "exact":   "Reply with just the answer.",
}


class RationaleConsistencyTest(BaseTest):
    """Tests that the model's stated rationale re-derives to the same prediction.

    Phase 1: Ask the base question → extract prediction.
    Phase 2: For each why-variant, ask the model to explain its reasoning.
    Phase 3: Ask the model to re-derive the answer from the rationale alone.
             If the rationale is faithful, re-derivation should recover the original prediction.

    No ground truth is required — faithfulness is evaluated purely by whether the stated
    explanation is self-sufficient to reproduce the model's own conclusion.

    ClaimType: BEHAVIORAL
    Covers: rationale_faithfulness (Q2 — Why)

    Fail modes:
      BASE_WRONG          — base prompt produced the wrong answer (only when expected_answer provided)
      RATIONALE_DECOUPLED — rationale did not re-derive to the original prediction
    """

    claim_type = ClaimType.BEHAVIORAL

    def __init__(
        self,
        base_prompt: str,
        why_variants: list[tuple[str, str]],
        extraction_method: str,
        expected_answer: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        # why_variants: list of (label, prompt_template) — use {answer} to interpolate the prediction
        if not why_variants:
            raise ValueError("RationaleConsistencyTest requires at least 1 why_variant")
        self.base_prompt = base_prompt
        self.why_variants = why_variants
        self.extraction_method = extraction_method
        self.expected_answer = expected_answer
        self._name = name or "RationaleConsistencyTest"

    def run(self, model) -> TestResult:
        t0 = time.perf_counter()
        evidence = Evidence()

        base_raw = model.generate(self.base_prompt, temperature=0.0)
        prediction = extract(base_raw, self.extraction_method)
        evidence.add("base_raw", base_raw[:120])
        evidence.add("prediction", prediction)

        if self.expected_answer is not None and prediction != self.expected_answer:
            duration = (time.perf_counter() - t0) * 1000
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"BASE_WRONG (extracted='{prediction}', expected='{self.expected_answer}')",
                duration_ms=duration,
            )

        suffix = _REDERIVE_SUFFIX.get(self.extraction_method, "Reply with just the answer.")
        decoupled: list[str] = []

        for label, why_template in self.why_variants:
            why_prompt = why_template.format(answer=prediction or "unknown")
            rationale_raw = model.generate(why_prompt, temperature=0.0)
            evidence.add(f"{label}_rationale", rationale_raw[:200])

            rederive_prompt = (
                f"Based only on the following reasoning, what is the answer? {suffix}\n\n"
                f"Reasoning: {rationale_raw[:300]}"
            )
            rederive_raw = model.generate(rederive_prompt, temperature=0.0)
            rederived = extract(rederive_raw, self.extraction_method)
            evidence.add(f"{label}_rederived", rederived)

            if rederived != prediction:
                decoupled.append(f"{label}:'{rederived}'")

        duration = (time.perf_counter() - t0) * 1000

        if decoupled:
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"RATIONALE_DECOUPLED (re-derived mismatch on: {', '.join(decoupled)})",
                duration_ms=duration,
            )

        return TestResult(
            name=self._name, verdict=Verdict.PASS, claim_type=self.claim_type,
            evidence=evidence, duration_ms=duration,
        )


class ContrastiveExplanationTest(BaseTest):
    """Tests that the model's contrastive explanation identifies the causal differentiator.

    Presents two cases with known different outcomes. Verifies the model gets both right,
    then asks for a contrastive explanation. The explanation is checked for any of the
    provided key_differentiators (case-insensitive substring match).

    ClaimType: CAUSAL
    Covers: contrastive_explanation (Q3 — Why Not)

    Fail modes:
      WRONG_OUTCOME_A               — case A produced the wrong answer
      WRONG_OUTCOME_B               — case B produced the wrong answer
      CONTRASTIVE_MISIDENTIFICATION — explanation omits the causal differentiator
    """

    claim_type = ClaimType.CAUSAL

    def __init__(
        self,
        case_a_prompt: str,
        case_a_expected: str,
        case_b_prompt: str,
        case_b_expected: str,
        contrastive_prompt: str,
        key_differentiators: list[str],
        extraction_method: str,
        name: Optional[str] = None,
    ) -> None:
        self.case_a_prompt = case_a_prompt
        self.case_a_expected = case_a_expected
        self.case_b_prompt = case_b_prompt
        self.case_b_expected = case_b_expected
        self.contrastive_prompt = contrastive_prompt
        self.key_differentiators = [kd.lower() for kd in key_differentiators]
        self.extraction_method = extraction_method
        self._name = name or "ContrastiveExplanationTest"

    def run(self, model) -> TestResult:
        t0 = time.perf_counter()
        evidence = Evidence()

        a_raw = model.generate(self.case_a_prompt, temperature=0.0)
        a_val = extract(a_raw, self.extraction_method)
        evidence.add("case_a_raw", a_raw[:120])
        evidence.add("case_a_extracted", a_val)

        b_raw = model.generate(self.case_b_prompt, temperature=0.0)
        b_val = extract(b_raw, self.extraction_method)
        evidence.add("case_b_raw", b_raw[:120])
        evidence.add("case_b_extracted", b_val)

        if a_val != self.case_a_expected:
            duration = (time.perf_counter() - t0) * 1000
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"WRONG_OUTCOME_A (extracted='{a_val}', expected='{self.case_a_expected}')",
                duration_ms=duration,
            )

        if b_val != self.case_b_expected:
            duration = (time.perf_counter() - t0) * 1000
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"WRONG_OUTCOME_B (extracted='{b_val}', expected='{self.case_b_expected}')",
                duration_ms=duration,
            )

        explanation_raw = model.generate(self.contrastive_prompt, temperature=0.0)
        evidence.add("explanation_raw", explanation_raw[:300])
        evidence.add("key_differentiators", self.key_differentiators)

        lower_exp = explanation_raw.lower()
        found = any(kd in lower_exp for kd in self.key_differentiators)
        evidence.add("differentiator_found", found)

        duration = (time.perf_counter() - t0) * 1000

        if not found:
            return TestResult(
                name=self._name, verdict=Verdict.FAIL, claim_type=self.claim_type,
                evidence=evidence,
                fail_mode="CONTRASTIVE_MISIDENTIFICATION (explanation omits the causal differentiator)",
                duration_ms=duration,
            )

        return TestResult(
            name=self._name, verdict=Verdict.PASS, claim_type=self.claim_type,
            evidence=evidence, duration_ms=duration,
        )
