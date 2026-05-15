from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

from ai_test_framework.core.base_test import BaseTest
from ai_test_framework.tests.consistency import (
    SurfaceFramingConsistencyTest,
    RepeatedPromptConsistencyTest,
    StatisticalConsistencyTest,
)
from ai_test_framework.tests.format import FormatContractTest


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class FailureMode:
    name: str
    description: str
    severity: Severity


# ── Catalogue ─────────────────────────────────────────────────────────────────
# Single source of truth for known AI model failure modes.
# Add new entries here as new test categories are built.

CATALOGUE: dict[str, FailureMode] = {
    "surface_framing": FailureMode(
        name="surface_framing",
        description="Answers change when the same problem is rephrased — signals pattern matching over reasoning",
        severity=Severity.CRITICAL,
    ),
    "contradiction_handling": FailureMode(
        name="contradiction_handling",
        description="Model answers confidently given contradictory premises instead of flagging them",
        severity=Severity.CRITICAL,
    ),
    "counterfactual_sensitivity": FailureMode(
        name="counterfactual_sensitivity",
        description="Model ignores changes to logically critical facts",
        severity=Severity.CRITICAL,
    ),
    "reasoning_chain_validity": FailureMode(
        name="reasoning_chain_validity",
        description="Chain-of-thought steps do not connect to the final answer",
        severity=Severity.HIGH,
    ),
    "irrelevant_fact_sensitivity": FailureMode(
        name="irrelevant_fact_sensitivity",
        description="Model changes its answer when only irrelevant surface details change",
        severity=Severity.HIGH,
    ),
    "sycophancy": FailureMode(
        name="sycophancy",
        description="Model changes correct answers under user disagreement without new arguments",
        severity=Severity.HIGH,
    ),
    "refusal_consistency": FailureMode(
        name="refusal_consistency",
        description="Model refuses harmful requests in some framings but not others",
        severity=Severity.HIGH,
    ),
    "temperature_stability": FailureMode(
        name="temperature_stability",
        description="Model answers are inconsistent across runs at realistic temperatures",
        severity=Severity.MEDIUM,
    ),
    "authority_bias": FailureMode(
        name="authority_bias",
        description="Model changes answers based on claimed authority of the asker",
        severity=Severity.MEDIUM,
    ),
    "calibration": FailureMode(
        name="calibration",
        description="Model's expressed confidence does not correlate with its actual accuracy",
        severity=Severity.MEDIUM,
    ),
    "determinism": FailureMode(
        name="determinism",
        description="Model produces different outputs on identical prompts at temperature=0",
        severity=Severity.LOW,
    ),
    "output_format_compliance": FailureMode(
        name="output_format_compliance",
        description="Model does not reliably respect format instructions across varied prompting styles",
        severity=Severity.LOW,
    ),
}

# Maps test types to the failure modes they exercise.
# When a new test class is added, register it here.
COVERAGE_MAP: dict[type, list[str]] = {
    SurfaceFramingConsistencyTest: ["surface_framing"],
    RepeatedPromptConsistencyTest: ["determinism"],
    StatisticalConsistencyTest: ["temperature_stability"],
    FormatContractTest: ["output_format_compliance"],
}

_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class ModeCoverage:
    mode: FailureMode
    covering_tests: list[str]  # names of tests in the suite that cover this mode

    @property
    def is_covered(self) -> bool:
        return len(self.covering_tests) > 0


@dataclass
class CoverageReport:
    suite_name: str
    mode_coverages: list[ModeCoverage]

    @property
    def covered(self) -> list[ModeCoverage]:
        return [mc for mc in self.mode_coverages if mc.is_covered]

    @property
    def uncovered(self) -> list[ModeCoverage]:
        return [mc for mc in self.mode_coverages if not mc.is_covered]

    @property
    def coverage_rate(self) -> float:
        total = len(self.mode_coverages)
        return len(self.covered) / total if total > 0 else 0.0

    def by_severity(self, severity: Severity) -> tuple[int, int]:
        """Returns (covered_count, total_count) for a given severity."""
        modes = [mc for mc in self.mode_coverages if mc.mode.severity == severity]
        covered = sum(1 for mc in modes if mc.is_covered)
        return covered, len(modes)


# ── Analyzer ───────────────────────────────────────────────────────────────────

class CoverageAnalyzer:
    """Scores a TestSuite against the failure mode catalogue.

    Analogous to a code coverage report — shows which known failure modes
    are exercised by the suite and which are gaps.

    Usage:
        analyzer = CoverageAnalyzer()
        report = analyzer.analyze(suite)
        # report.coverage_rate, report.uncovered, report.by_severity(Severity.CRITICAL)
    """

    def analyze(self, suite, reporter=None) -> CoverageReport:
        if reporter is None:
            reporter = CoverageReporter()

        mode_to_tests: dict[str, list[str]] = {name: [] for name in CATALOGUE}

        for test in suite._tests:
            test_name = getattr(test, "_name", type(test).__name__)
            for test_type, mode_names in COVERAGE_MAP.items():
                if isinstance(test, test_type):
                    for mode_name in mode_names:
                        if mode_name in mode_to_tests:
                            mode_to_tests[mode_name].append(test_name)

        mode_coverages = [
            ModeCoverage(mode=CATALOGUE[name], covering_tests=tests)
            for name, tests in mode_to_tests.items()
        ]
        mode_coverages.sort(
            key=lambda mc: (_SEVERITY_ORDER[mc.mode.severity], mc.mode.name)
        )

        report = CoverageReport(suite_name=suite.name, mode_coverages=mode_coverages)
        reporter.on_report(report)
        return report


# ── Reporter ───────────────────────────────────────────────────────────────────

class CoverageReporter:
    def on_report(self, report: CoverageReport) -> None:
        sep = "-" * 64
        pct = f"{report.coverage_rate * 100:.0f}%"
        print(f"\n  {sep}")
        print(f"  Coverage Report  [{report.suite_name}]  {len(report.covered)}/{len(report.mode_coverages)} ({pct})")
        print(f"  {sep}")

        current_severity = None
        for mc in report.mode_coverages:
            if mc.mode.severity != current_severity:
                current_severity = mc.mode.severity
                print(f"\n  [{current_severity.value.upper()}]")

            icon = "+" if mc.is_covered else "x"
            name_col = f"{mc.mode.name:<30}"
            if mc.is_covered:
                shown = mc.covering_tests[:2]
                rest = len(mc.covering_tests) - 2
                tests_str = ", ".join(shown)
                if rest > 0:
                    tests_str += f" (+{rest} more)"
                detail = tests_str
            else:
                detail = "not covered"
            print(f"  {icon} {name_col}  {detail}")

        print(f"\n  {sep}")
        summary_parts = []
        for sev in Severity:
            c, t = report.by_severity(sev)
            summary_parts.append(f"{sev.value.capitalize()}: {c}/{t}")
        print("  " + "  |  ".join(summary_parts))
