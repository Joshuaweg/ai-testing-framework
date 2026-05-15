from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional

from ai_test_framework.core.base_test import BaseTest, TestResult, Verdict, ClaimType, Evidence


@dataclass
class SuiteResult:
    suite_name: str
    results: list
    total_ms: float

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.FAIL)

    @property
    def errored(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.ERROR)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.errored == 0


class TestSuite:
    def __init__(self, name: str) -> None:
        self.name = name
        self._tests: list[BaseTest] = []

    def add(self, test: BaseTest) -> "TestSuite":
        self._tests.append(test)
        return self

    def run(self, model, reporter=None) -> SuiteResult:
        if reporter is None:
            reporter = TerminalReporter()
        results = []
        start = time.perf_counter()
        for test in self._tests:
            t0 = time.perf_counter()
            try:
                result = test.run(model)
            except Exception as exc:
                result = TestResult(
                    name=getattr(test, "_name", type(test).__name__),
                    verdict=Verdict.ERROR,
                    claim_type=getattr(type(test), "claim_type", ClaimType.BEHAVIORAL),
                    evidence=Evidence(),
                    error_msg=str(exc),
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
            reporter.on_result(result)
            results.append(result)
        total_ms = (time.perf_counter() - start) * 1000
        suite_result = SuiteResult(suite_name=self.name, results=results, total_ms=total_ms)
        reporter.on_suite_end(suite_result)
        return suite_result


class TerminalReporter:
    def on_result(self, result: TestResult) -> None:
        icons = {Verdict.PASS: "\u2713", Verdict.FAIL: "\u2717", Verdict.ERROR: "!", Verdict.SKIP: "-"}
        icon = icons.get(result.verdict, "?")
        claim = result.claim_type.value
        if result.verdict == Verdict.PASS:
            line = f"  {icon} PASS  ({claim})  {result.name}"
        elif result.verdict == Verdict.FAIL:
            suffix = f" \u2014 {result.fail_mode}" if result.fail_mode else ""
            line = f"  {icon} FAIL{suffix}  ({claim})  {result.name}"
        elif result.verdict == Verdict.ERROR:
            line = f"  {icon} ERROR  {result.name}  \u2014 {result.error_msg}"
        else:
            line = f"  {icon} SKIP  {result.name}"
        print(line)
        for k, v in result.evidence.items():
            print(f"      {k}: {v}")
        print(f"      duration: {result.duration_ms:.0f}ms")

    def on_suite_end(self, suite_result: SuiteResult) -> None:
        separator = "\u2500" * 56
        print(f"\n  {separator}")
        pct = f"{suite_result.pass_rate*100:.0f}%"
        print(
            f"  {suite_result.passed}/{suite_result.total} passed ({pct})"
            f"  |  {suite_result.failed} failed"
            f"  |  {suite_result.errored} errors"
            f"  |  {suite_result.total_ms:.0f}ms"
            f"  [{suite_result.suite_name}]"
        )


class _SilentReporter:
    def on_result(self, result: TestResult) -> None:
        pass

    def on_suite_end(self, suite_result: SuiteResult) -> None:
        pass


@dataclass
class TestComparison:
    name: str
    verdict_a: Verdict
    verdict_b: Verdict

    @property
    def changed(self) -> bool:
        return self.verdict_a != self.verdict_b

    @property
    def regressed(self) -> bool:
        return self.verdict_a == Verdict.PASS and self.verdict_b != Verdict.PASS

    @property
    def improved(self) -> bool:
        return self.verdict_a != Verdict.PASS and self.verdict_b == Verdict.PASS


@dataclass
class ComparisonResult:
    label_a: str
    label_b: str
    suite_name: str
    comparisons: list[TestComparison]
    result_a: SuiteResult
    result_b: SuiteResult

    @property
    def regressions(self) -> list[TestComparison]:
        return [c for c in self.comparisons if c.regressed]

    @property
    def improvements(self) -> list[TestComparison]:
        return [c for c in self.comparisons if c.improved]

    @property
    def unchanged(self) -> list[TestComparison]:
        return [c for c in self.comparisons if not c.changed]

    @property
    def all_passed_both(self) -> bool:
        return self.result_a.all_passed and self.result_b.all_passed


class ComparisonRunner:
    """Runs the same TestSuite against two models and diffs the results.

    Use to detect regressions when upgrading models, changing quantization,
    tuning sampling parameters, or altering system prompts.
    """

    def __init__(self, suite: TestSuite) -> None:
        self.suite = suite

    def compare(
        self,
        model_a,
        model_b,
        label_a: Optional[str] = None,
        label_b: Optional[str] = None,
        reporter=None,
    ) -> ComparisonResult:
        label_a = label_a or getattr(model_a, "name", "model_a")
        label_b = label_b or getattr(model_b, "name", "model_b")
        if reporter is None:
            reporter = ComparisonReporter()

        silent = _SilentReporter()
        result_a = self.suite.run(model_a, reporter=silent)
        result_b = self.suite.run(model_b, reporter=silent)

        comparisons = [
            TestComparison(name=ra.name, verdict_a=ra.verdict, verdict_b=rb.verdict)
            for ra, rb in zip(result_a.results, result_b.results)
        ]

        cr = ComparisonResult(
            label_a=label_a,
            label_b=label_b,
            suite_name=self.suite.name,
            comparisons=comparisons,
            result_a=result_a,
            result_b=result_b,
        )
        reporter.on_comparison(cr)
        return cr


class ComparisonReporter:
    _VERDICT_SYMBOL = {
        Verdict.PASS: "PASS",
        Verdict.FAIL: "FAIL",
        Verdict.ERROR: "ERR ",
        Verdict.SKIP: "SKIP",
    }

    def on_comparison(self, cr: ComparisonResult) -> None:
        sep = "\u2500" * 64
        print(f"\n  {sep}")
        print(f"  Comparison  [{cr.suite_name}]")
        print(f"  A: {cr.label_a}")
        print(f"  B: {cr.label_b}")
        print(f"  {sep}")

        col_w = max((len(c.name) for c in cr.comparisons), default=20)
        header = f"  {'Test':<{col_w}}  {'A':<6}  {'B':<6}  Change"
        print(header)
        row_sep = "\u2500" * (col_w + 24)
        print(f"  {row_sep}")

        for c in cr.comparisons:
            va = self._VERDICT_SYMBOL[c.verdict_a]
            vb = self._VERDICT_SYMBOL[c.verdict_b]
            if c.regressed:
                change = "\u2193 REGRESSED"
            elif c.improved:
                change = "\u2191 IMPROVED"
            elif c.changed:
                change = "~ CHANGED"
            else:
                change = "\u2014"
            print(f"  {c.name:<{col_w}}  {va:<6}  {vb:<6}  {change}")

        print(f"  {sep}")
        print(
            f"  A: {cr.result_a.passed}/{cr.result_a.total} passed"
            f"  |  B: {cr.result_b.passed}/{cr.result_b.total} passed"
            f"  |  {len(cr.regressions)} regression(s)"
            f"  |  {len(cr.improvements)} improvement(s)"
        )
