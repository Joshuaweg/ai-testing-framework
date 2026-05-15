from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from ai_test_framework.core.base_test import BaseTest, Verdict
from ai_test_framework.tests.consistency import (
    SurfaceFramingConsistencyTest,
    RepeatedPromptConsistencyTest,
    StatisticalConsistencyTest,
)


@dataclass
class AdversarialProbe:
    """A test constructed with a deliberately broken configuration.

    Running this against a model must produce `expected_verdict`.
    If it produces a different verdict, the test design is too weak
    to catch this class of failure — analogous to a mutation that
    survives your test suite.
    """
    name: str
    test: BaseTest
    expected_verdict: Verdict
    failure_description: str


@dataclass
class ProbeResult:
    probe: AdversarialProbe
    actual_verdict: Verdict

    @property
    def caught(self) -> bool:
        return self.actual_verdict == self.probe.expected_verdict


@dataclass
class ValidationResult:
    probe_results: list[ProbeResult]

    @property
    def all_caught(self) -> bool:
        return all(pr.caught for pr in self.probe_results)

    @property
    def missed(self) -> list[ProbeResult]:
        return [pr for pr in self.probe_results if not pr.caught]

    @property
    def caught(self) -> list[ProbeResult]:
        return [pr for pr in self.probe_results if pr.caught]


class AdversarialValidator:
    """Runs adversarial probes to verify the test suite catches failures.

    Analogous to mutation testing: if a test passes on deliberately broken input,
    the test is too weak to catch that failure mode.

    Usage:
        probes = [wrong_answer_probe(fixture, wrong_answer="99")]
        validator = AdversarialValidator()
        result = validator.validate(model, probes)
    """

    def validate(
        self,
        model,
        probes: list[AdversarialProbe],
        reporter=None,
    ) -> ValidationResult:
        if reporter is None:
            reporter = AdversarialReporter()

        probe_results = []
        for probe in probes:
            result = probe.test.run(model)
            pr = ProbeResult(probe=probe, actual_verdict=result.verdict)
            probe_results.append(pr)
            reporter.on_probe_result(pr)

        vr = ValidationResult(probe_results=probe_results)
        reporter.on_validation_end(vr)
        return vr


# ── Probe factories ────────────────────────────────────────────────────────────

def wrong_answer_probe(
    test: SurfaceFramingConsistencyTest | RepeatedPromptConsistencyTest,
    wrong_answer: str,
    name: Optional[str] = None,
) -> AdversarialProbe:
    """Probe that replaces expected_answer with a deliberately wrong value.

    Must fail with CONSISTENT_WRONG. If it passes, the test cannot detect
    when a model answers a problem correctly but the expected answer is wrong —
    meaning the test would pass regardless of what the model says.
    """
    base_name = getattr(test, "_name", type(test).__name__)

    if isinstance(test, SurfaceFramingConsistencyTest):
        broken = SurfaceFramingConsistencyTest(
            variants=test.variants,
            extraction_method=test.extraction_method,
            expected_answer=wrong_answer,
            name=f"{base_name}[wrong_answer={wrong_answer!r}]",
        )
    elif isinstance(test, RepeatedPromptConsistencyTest):
        broken = RepeatedPromptConsistencyTest(
            prompt=test.prompt,
            extraction_method=test.extraction_method,
            expected_answer=wrong_answer,
            n_runs=test.n_runs,
            name=f"{base_name}[wrong_answer={wrong_answer!r}]",
        )
    else:
        raise TypeError(f"wrong_answer_probe does not support {type(test).__name__}")

    return AdversarialProbe(
        name=name or f"wrong_answer_probe({base_name})",
        test=broken,
        expected_verdict=Verdict.FAIL,
        failure_description=f"expected_answer forcibly set to '{wrong_answer}' (deliberately wrong)",
    )


def high_threshold_probe(
    test: StatisticalConsistencyTest,
    name: Optional[str] = None,
) -> AdversarialProbe:
    """Probe that forces threshold=1.0 (perfect consistency required).

    At any temperature > 0, nearly all models will fail this. If it passes,
    either the temperature is so low the test is not exercising non-determinism,
    or the threshold comparison logic is broken.
    """
    base_name = getattr(test, "_name", type(test).__name__)
    broken = StatisticalConsistencyTest(
        prompt=test.prompt,
        extraction_method=test.extraction_method,
        expected_answer=test.expected_answer,
        n_runs=test.n_runs,
        threshold=1.0,
        temperature=test.temperature,
        name=f"{base_name}[threshold=1.0]",
    )
    return AdversarialProbe(
        name=name or f"high_threshold_probe({base_name})",
        test=broken,
        expected_verdict=Verdict.FAIL,
        failure_description="threshold forced to 1.0 — perfect consistency required at temp>0",
    )


def build_probes(test: BaseTest) -> list[AdversarialProbe]:
    """Auto-generate the standard adversarial probe set for a test.

    Returns all probes that can be automatically derived from the test's
    configuration without requiring domain knowledge of the correct answer.
    """
    probes: list[AdversarialProbe] = []
    if isinstance(test, (SurfaceFramingConsistencyTest, RepeatedPromptConsistencyTest)):
        correct = test.expected_answer
        # Pick a wrong answer that can't accidentally be correct
        wrong = "__ADVERSARIAL_WRONG_ANSWER__"
        probes.append(wrong_answer_probe(test, wrong_answer=wrong))
    if isinstance(test, StatisticalConsistencyTest):
        probes.append(high_threshold_probe(test))
    return probes


# ── Reporter ───────────────────────────────────────────────────────────────────

class AdversarialReporter:
    def on_probe_result(self, pr: ProbeResult) -> None:
        if pr.caught:
            icon, label = "✓", "CAUGHT "
        else:
            icon, label = "✗", "MISSED "
        print(f"  {icon} {label}  {pr.probe.name}")
        print(f"           broken: {pr.probe.failure_description}")
        print(f"           expected: {pr.probe.expected_verdict.value}  |  got: {pr.actual_verdict.value}", end="")
        if not pr.caught:
            print("  ← test design failure", end="")
        print()

    def on_validation_end(self, vr: ValidationResult) -> None:
        sep = "─" * 56
        print(f"\n  {sep}")
        total = len(vr.probe_results)
        caught = len(vr.caught)
        missed = len(vr.missed)
        print(f"  {caught}/{total} probes caught  |  {missed} test design failure(s)")
        if vr.missed:
            print("  !! These tests did not catch their adversarial conditions:")
            for pr in vr.missed:
                print(f"     - {pr.probe.name}")
