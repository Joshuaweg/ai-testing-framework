from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from ai_test_framework.core.base_test import BaseTest
from ai_test_framework.tests.consistency import (
    SurfaceFramingConsistencyTest,
    RepeatedPromptConsistencyTest,
    StatisticalConsistencyTest,
)
from ai_test_framework.tests.format import FormatContractTest
from ai_test_framework.tests.causal import (
    WhatIfConsistencyTest,
    InvariantExplanationTest,
    CounterfactualValidityTest,
)
from ai_test_framework.tests.explanatory import RationaleConsistencyTest, ContrastiveExplanationTest
from ai_test_framework.tests.self_report import DataProvenanceConsistencyTest, OutputScopeConsistencyTest


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CATCategory(Enum):
    """The 9 question categories from the XAI Question Bank (Liao & Varshney, 2022).

    Organizes failure modes by the type of explainability need they address.
    """
    HOW_GLOBAL        = "how_global"         # Q1: General model logic/process
    WHY               = "why"                # Q2: Reason behind a specific prediction
    WHY_NOT           = "why_not"            # Q3: Why a different outcome wasn't reached
    HOW_TO_BE_THAT    = "how_to_be_that"     # Q4: What to change to get a different outcome
    HOW_STILL_BE_THIS = "how_still_be_this"  # Q5: What can change without affecting outcome
    WHAT_IF           = "what_if"            # Q6: How prediction changes with input changes
    PERFORMANCE       = "performance"        # Q7: Model performance information
    DATA              = "data"               # Q8: Training data information
    OUTPUT            = "output"             # Q9: Expected output scope and usage


_CAT_LABELS: dict[CATCategory, str] = {
    CATCategory.HOW_GLOBAL:        "Q1 — How (global model-wide)",
    CATCategory.WHY:               "Q2 — Why (a given prediction)",
    CATCategory.WHY_NOT:           "Q3 — Why Not (a different prediction)",
    CATCategory.HOW_TO_BE_THAT:    "Q4 — How to Be That (a different prediction)",
    CATCategory.HOW_STILL_BE_THIS: "Q5 — How to Still Be This (the current prediction)",
    CATCategory.WHAT_IF:           "Q6 — What If",
    CATCategory.PERFORMANCE:       "Q7 — Performance",
    CATCategory.DATA:              "Q8 — Data",
    CATCategory.OUTPUT:            "Q9 — Output",
}


@dataclass(frozen=True)
class FailureMode:
    name: str
    description: str
    severity: Severity
    cat_category: CATCategory


# ── Catalogue ─────────────────────────────────────────────────────────────────
# Organized by CAT category (Q1–Q9). Add new entries here as test categories grow.

CATALOGUE: dict[str, FailureMode] = {
    # Q1 — HOW (global model-wide)
    "surface_framing": FailureMode(
        name="surface_framing",
        description="Answers change when the same problem is rephrased — signals pattern matching over reasoning",
        severity=Severity.CRITICAL,
        cat_category=CATCategory.HOW_GLOBAL,
    ),
    "refusal_consistency": FailureMode(
        name="refusal_consistency",
        description="Model refuses harmful requests in some framings but not others",
        severity=Severity.HIGH,
        cat_category=CATCategory.HOW_GLOBAL,
    ),
    # Q2 — WHY (a given prediction)
    "contradiction_handling": FailureMode(
        name="contradiction_handling",
        description="Model answers confidently given contradictory premises instead of flagging them",
        severity=Severity.CRITICAL,
        cat_category=CATCategory.WHY,
    ),
    "reasoning_chain_validity": FailureMode(
        name="reasoning_chain_validity",
        description="Chain-of-thought steps do not connect to the final answer",
        severity=Severity.HIGH,
        cat_category=CATCategory.WHY,
    ),
    "sycophancy": FailureMode(
        name="sycophancy",
        description="Model changes correct answers under user disagreement without new arguments",
        severity=Severity.HIGH,
        cat_category=CATCategory.WHY,
    ),
    "authority_bias": FailureMode(
        name="authority_bias",
        description="Model changes answers based on claimed authority of the asker",
        severity=Severity.MEDIUM,
        cat_category=CATCategory.WHY,
    ),
    "rationale_faithfulness": FailureMode(
        name="rationale_faithfulness",
        description="Model's stated reason for a prediction is inconsistent or decoupled from its actual prediction across elicitation framings",
        severity=Severity.HIGH,
        cat_category=CATCategory.WHY,
    ),
    # Q3 — WHY NOT (a different prediction)
    "contrastive_explanation": FailureMode(
        name="contrastive_explanation",
        description="Model fails to correctly identify the differentiating factor between two inputs with different outcomes",
        severity=Severity.HIGH,
        cat_category=CATCategory.WHY_NOT,
    ),
    # Q4 — HOW TO BE THAT (a different prediction)
    "counterfactual_sensitivity": FailureMode(
        name="counterfactual_sensitivity",
        description="Model ignores changes to logically critical facts — output is unresponsive to causal perturbations",
        severity=Severity.CRITICAL,
        cat_category=CATCategory.HOW_TO_BE_THAT,
    ),
    "counterfactual_actionability": FailureMode(
        name="counterfactual_actionability",
        description="A minimal change sufficient to flip the outcome does not produce the expected different prediction",
        severity=Severity.HIGH,
        cat_category=CATCategory.HOW_TO_BE_THAT,
    ),
    # Q5 — HOW TO STILL BE THIS (the current prediction)
    "irrelevant_fact_sensitivity": FailureMode(
        name="irrelevant_fact_sensitivity",
        description="Model changes its answer when only causally irrelevant surface details change",
        severity=Severity.HIGH,
        cat_category=CATCategory.HOW_STILL_BE_THIS,
    ),
    # Q6 — WHAT IF
    "feature_perturbation_accuracy": FailureMode(
        name="feature_perturbation_accuracy",
        description="Model's output fails to reflect the expected directional change when a relevant input feature is perturbed",
        severity=Severity.HIGH,
        cat_category=CATCategory.WHAT_IF,
    ),
    # Q7 — PERFORMANCE
    "temperature_stability": FailureMode(
        name="temperature_stability",
        description="Model answers are inconsistent across runs at realistic temperatures",
        severity=Severity.MEDIUM,
        cat_category=CATCategory.PERFORMANCE,
    ),
    "calibration": FailureMode(
        name="calibration",
        description="Model's expressed confidence does not correlate with its actual accuracy",
        severity=Severity.MEDIUM,
        cat_category=CATCategory.PERFORMANCE,
    ),
    "determinism": FailureMode(
        name="determinism",
        description="Model produces different outputs on identical prompts at temperature=0",
        severity=Severity.LOW,
        cat_category=CATCategory.PERFORMANCE,
    ),
    # Q8 — DATA
    "data_provenance_consistency": FailureMode(
        name="data_provenance_consistency",
        description="Model gives inconsistent answers about its training data when asked in different ways",
        severity=Severity.MEDIUM,
        cat_category=CATCategory.DATA,
    ),
    # Q9 — OUTPUT
    "output_format_compliance": FailureMode(
        name="output_format_compliance",
        description="Model does not reliably respect format instructions across varied prompting styles",
        severity=Severity.LOW,
        cat_category=CATCategory.OUTPUT,
    ),
    "output_scope_accuracy": FailureMode(
        name="output_scope_accuracy",
        description="Model's claimed output capabilities are inconsistent with its actual output behavior",
        severity=Severity.MEDIUM,
        cat_category=CATCategory.OUTPUT,
    ),
}

# Maps test types to the failure modes they exercise.
# Register new test classes here when added.
COVERAGE_MAP: dict[type, list[str]] = {
    SurfaceFramingConsistencyTest:   ["surface_framing"],
    RepeatedPromptConsistencyTest:   ["determinism"],
    StatisticalConsistencyTest:      ["temperature_stability"],
    FormatContractTest:              ["output_format_compliance"],
    WhatIfConsistencyTest:           ["feature_perturbation_accuracy"],
    InvariantExplanationTest:        ["irrelevant_fact_sensitivity"],
    CounterfactualValidityTest:      ["counterfactual_sensitivity", "counterfactual_actionability"],
    RationaleConsistencyTest:        ["rationale_faithfulness"],
    ContrastiveExplanationTest:      ["contrastive_explanation"],
    DataProvenanceConsistencyTest:   ["data_provenance_consistency"],
    OutputScopeConsistencyTest:      ["output_scope_accuracy"],
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
    covering_tests: list[str]

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
        modes = [mc for mc in self.mode_coverages if mc.mode.severity == severity]
        covered = sum(1 for mc in modes if mc.is_covered)
        return covered, len(modes)

    def by_cat_category(self, cat: CATCategory) -> tuple[int, int]:
        modes = [mc for mc in self.mode_coverages if mc.mode.cat_category == cat]
        covered = sum(1 for mc in modes if mc.is_covered)
        return covered, len(modes)


# ── Analyzer ───────────────────────────────────────────────────────────────────

class CoverageAnalyzer:
    """Scores a TestSuite against the failure mode catalogue.

    Analogous to a code coverage report — shows which known failure modes are exercised
    by the suite and which are gaps, organized by CAT question category.

    Usage:
        analyzer = CoverageAnalyzer()
        report = analyzer.analyze(suite)
        # report.coverage_rate, report.uncovered, report.by_cat_category(CATCategory.WHY)
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

        # Group by CAT category in Q1–Q9 order
        for cat in CATCategory:
            cat_modes = [mc for mc in report.mode_coverages if mc.mode.cat_category == cat]
            if not cat_modes:
                continue
            c, t = report.by_cat_category(cat)
            print(f"\n  [{_CAT_LABELS[cat]}]  {c}/{t}")
            cat_modes_sorted = sorted(cat_modes, key=lambda mc: _SEVERITY_ORDER[mc.mode.severity])
            for mc in cat_modes_sorted:
                icon = "+" if mc.is_covered else "x"
                name_col = f"{mc.mode.name:<35}"
                sev = f"[{mc.mode.severity.value}]"
                if mc.is_covered:
                    shown = mc.covering_tests[:2]
                    rest = len(mc.covering_tests) - 2
                    tests_str = ", ".join(shown)
                    if rest > 0:
                        tests_str += f" (+{rest} more)"
                    detail = tests_str
                else:
                    detail = "not covered"
                print(f"  {icon} {name_col}  {sev:<10}  {detail}")

        print(f"\n  {sep}")
        parts = [f"Q{i}: {report.by_cat_category(cat)[0]}/{report.by_cat_category(cat)[1]}"
                 for i, cat in enumerate(CATCategory, 1)]
        print("  " + "  |  ".join(parts))
