import pytest
from ai_test_framework.core.base_test import ClaimType, Verdict, Evidence, TestResult, BaseTest


class TestClaimType:
    def test_behavioral_value(self):
        assert ClaimType.BEHAVIORAL.value == "behavioral"

    def test_observational_value(self):
        assert ClaimType.OBSERVATIONAL.value == "observational"

    def test_causal_value(self):
        assert ClaimType.CAUSAL.value == "causal"

    def test_three_members(self):
        assert len(ClaimType) == 3


class TestVerdict:
    def test_all_four_exist(self):
        assert {Verdict.PASS, Verdict.FAIL, Verdict.ERROR, Verdict.SKIP}

    def test_pass_value(self):
        assert Verdict.PASS.value == "PASS"

    def test_fail_value(self):
        assert Verdict.FAIL.value == "FAIL"


class TestEvidence:
    def test_add_and_items(self):
        e = Evidence()
        e.add("key", "value")
        assert ("key", "value") in e.items()

    def test_empty_by_default(self):
        assert Evidence().items() == []

    def test_overwrite_key(self):
        e = Evidence()
        e.add("k", 1)
        e.add("k", 2)
        assert e.data["k"] == 2


class TestTestResult:
    def test_construction(self):
        e = Evidence()
        r = TestResult(
            name="t",
            verdict=Verdict.PASS,
            claim_type=ClaimType.BEHAVIORAL,
            evidence=e,
        )
        assert r.verdict == Verdict.PASS
        assert r.fail_mode is None
        assert r.error_msg is None
        assert r.duration_ms == 0.0

    def test_fail_mode_set(self):
        r = TestResult(
            name="t",
            verdict=Verdict.FAIL,
            claim_type=ClaimType.BEHAVIORAL,
            evidence=Evidence(),
            fail_mode="INCONSISTENT",
        )
        assert r.fail_mode == "INCONSISTENT"


class TestBaseTest:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseTest()
