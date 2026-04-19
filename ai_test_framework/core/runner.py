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
