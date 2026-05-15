from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

from ai_test_framework.core.base_test import BaseTest, ClaimType, Verdict, Evidence, TestResult
from ai_test_framework.core.extractors import extract


@dataclass
class FormatInstruction:
    label: str
    instruction: str  # the format directive appended to the base prompt


# ── Built-in instruction sets ──────────────────────────────────────────────────

NUMERIC_INSTRUCTIONS: list[FormatInstruction] = [
    FormatInstruction("direct",   "Reply with just the number."),
    FormatInstruction("only",     "Answer using only a number."),
    FormatInstruction("single",   "Respond with a single number only."),
    FormatInstruction("output",   "Output only the numeric answer."),
]

YN_INSTRUCTIONS: list[FormatInstruction] = [
    FormatInstruction("direct",   "Reply with just yes or no."),
    FormatInstruction("only",     "Answer using only yes or no."),
    FormatInstruction("single",   "Respond with yes or no only."),
]

LETTER_INSTRUCTIONS: list[FormatInstruction] = [
    FormatInstruction("direct",   "Reply with just the letter of your answer."),
    FormatInstruction("only",     "Answer using only the letter (A, B, C, or D)."),
    FormatInstruction("single",   "Output a single letter only."),
]


# ── Test class ─────────────────────────────────────────────────────────────────

class FormatContractTest(BaseTest):
    """Tests whether a model reliably follows format instructions across varied phrasings.

    Analogous to contract testing: the model has an implicit contract with callers
    that it will respect output format constraints. This test verifies the contract
    holds regardless of how the format instruction is worded.

    The full prompt for each variant is: f"{base_prompt} {instruction.instruction}"

    ClaimType: BEHAVIORAL
    Claim: "The model produces extractable output under all tested format instruction phrasings."

    Fail modes:
      FORMAT_BREACH     — extraction returned None for one or more variants
      INCONSISTENT      — extraction succeeded but answers differ across variants
      CONSISTENT_WRONG  — answers agree and are extractable but wrong
    """

    claim_type = ClaimType.BEHAVIORAL

    def __init__(
        self,
        base_prompt: str,
        format_instructions: list[FormatInstruction],
        extraction_method: str,
        expected_answer: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        if len(format_instructions) < 2:
            raise ValueError("FormatContractTest requires at least 2 format_instructions")
        self.base_prompt = base_prompt
        self.format_instructions = format_instructions
        self.extraction_method = extraction_method
        self.expected_answer = expected_answer
        self._name = name or "FormatContractTest"

    def run(self, model) -> TestResult:
        t0 = time.perf_counter()
        evidence = Evidence()
        extracted: dict[str, Optional[str]] = {}
        breach_labels: list[str] = []

        for fi in self.format_instructions:
            full_prompt = f"{self.base_prompt} {fi.instruction}"
            raw = model.generate(full_prompt, temperature=0.0)
            value = extract(raw, self.extraction_method)
            evidence.add(f"{fi.label}_instruction", fi.instruction)
            evidence.add(f"{fi.label}_raw", raw[:120])
            evidence.add(f"{fi.label}_extracted", value)
            extracted[fi.label] = value
            if value is None:
                breach_labels.append(fi.label)

        duration = (time.perf_counter() - t0) * 1000

        if breach_labels:
            evidence.add("format_breach_on", breach_labels)
            evidence.add(
                "extraction_rate",
                f"{len(self.format_instructions) - len(breach_labels)}/{len(self.format_instructions)}",
            )
            return TestResult(
                name=self._name,
                verdict=Verdict.FAIL,
                claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"FORMAT_BREACH ({len(breach_labels)} instruction(s) produced un-extractable output)",
                duration_ms=duration,
            )

        unique = set(extracted.values())
        if len(unique) > 1:
            return TestResult(
                name=self._name,
                verdict=Verdict.FAIL,
                claim_type=self.claim_type,
                evidence=evidence,
                fail_mode="INCONSISTENT (format phrasing affected the answer)",
                duration_ms=duration,
            )

        answer = next(iter(unique))
        if self.expected_answer is not None and answer != self.expected_answer:
            return TestResult(
                name=self._name,
                verdict=Verdict.FAIL,
                claim_type=self.claim_type,
                evidence=evidence,
                fail_mode=f"CONSISTENT_WRONG (extracted='{answer}', expected='{self.expected_answer}')",
                duration_ms=duration,
            )

        return TestResult(
            name=self._name,
            verdict=Verdict.PASS,
            claim_type=self.claim_type,
            evidence=evidence,
            duration_ms=duration,
        )
