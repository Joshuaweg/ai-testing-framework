import pytest
from ai_test_framework.core.base_test import BaseTest, ClaimType, Verdict, Evidence, TestResult
from ai_test_framework.core.runner import TestSuite, SuiteResult, TerminalReporter


class _NoOpReporter:
    def on_result(self, result): pass
    def on_suite_end(self, suite_result): pass


def _fixed_test(verdict: Verdict) -> BaseTest:
    class FixedTest(BaseTest):
        claim_type = ClaimType.BEHAVIORAL
        def __init__(self, v):
            self._v = v
        def run(self, model):
            return TestResult(
                name="fixed",
                verdict=self._v,
                claim_type=ClaimType.BEHAVIORAL,
                evidence=Evidence(),
            )
    return FixedTest(verdict)


class TestSuiteResult:
    def test_empty_pass_rate(self):
        sr = SuiteResult(suite_name="s", results=[], total_ms=0.0)
        assert sr.pass_rate == 0.0
        assert sr.total == 0

    def test_all_passed_true(self):
        r = TestResult(name="t", verdict=Verdict.PASS, claim_type=ClaimType.BEHAVIORAL, evidence=Evidence())
        sr = SuiteResult(suite_name="s", results=[r], total_ms=0.0)
        assert sr.all_passed is True

    def test_all_passed_false_on_fail(self):
        r = TestResult(name="t", verdict=Verdict.FAIL, claim_type=ClaimType.BEHAVIORAL, evidence=Evidence())
        sr = SuiteResult(suite_name="s", results=[r], total_ms=0.0)
        assert sr.all_passed is False


class TestTestSuite:
    def test_empty_suite(self):
        suite = TestSuite("empty")
        result = suite.run(model=None, reporter=_NoOpReporter())
        assert result.pass_rate == 0.0
        assert result.total == 0

    def test_all_pass(self):
        suite = TestSuite("s")
        suite.add(_fixed_test(Verdict.PASS)).add(_fixed_test(Verdict.PASS))
        result = suite.run(model=None, reporter=_NoOpReporter())
        assert result.pass_rate == 1.0
        assert result.all_passed is True

    def test_mixed_results(self):
        suite = TestSuite("s")
        suite.add(_fixed_test(Verdict.PASS))
        suite.add(_fixed_test(Verdict.FAIL))
        result = suite.run(model=None, reporter=_NoOpReporter())
        assert result.passed == 1
        assert result.failed == 1
        assert result.all_passed is False

    def test_exception_becomes_error(self):
        class ExplodingTest(BaseTest):
            claim_type = ClaimType.BEHAVIORAL
            def run(self, model):
                raise RuntimeError("boom")
        suite = TestSuite("s")
        suite.add(ExplodingTest())
        result = suite.run(model=None, reporter=_NoOpReporter())
        assert result.errored == 1
        assert result.results[0].verdict == Verdict.ERROR
        assert "boom" in result.results[0].error_msg

    def test_fluent_add_returns_suite(self):
        suite = TestSuite("s")
        returned = suite.add(_fixed_test(Verdict.PASS))
        assert returned is suite

    def test_terminal_reporter_smoke(self):
        r_pass = TestResult(name="p", verdict=Verdict.PASS, claim_type=ClaimType.BEHAVIORAL, evidence=Evidence())
        r_fail = TestResult(name="f", verdict=Verdict.FAIL, claim_type=ClaimType.BEHAVIORAL, evidence=Evidence(), fail_mode="INCONSISTENT")
        sr = SuiteResult(suite_name="smoke", results=[r_pass, r_fail], total_ms=100.0)
        reporter = TerminalReporter()
        reporter.on_result(r_pass)
        reporter.on_result(r_fail)
        reporter.on_suite_end(sr)  # must not raise
