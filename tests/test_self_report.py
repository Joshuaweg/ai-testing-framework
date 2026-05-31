import pytest
from ai_test_framework.core.base_test import ClaimType, Verdict
from ai_test_framework.tests.self_report import DataProvenanceConsistencyTest, OutputScopeConsistencyTest


class StubModel:
    name = "stub"

    def __init__(self, responses: dict):
        self._responses = responses

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        for key, val in self._responses.items():
            if key in prompt:
                return val
        return "I don't know"


# ── DataProvenanceConsistencyTest ─────────────────────────────────────────────

class TestDataProvenanceConsistencyTest:
    def _make_test(self):
        return DataProvenanceConsistencyTest(
            data_query_variants=[
                ("q1", "PHRASING_A: what year?"),
                ("q2", "PHRASING_B: what year?"),
                ("q3", "PHRASING_C: what year?"),
            ],
            extraction_method="numeric",
        )

    def test_pass_consistent(self):
        model = StubModel({"PHRASING_A": "2023", "PHRASING_B": "2023", "PHRASING_C": "2023"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.PASS

    def test_fail_inconsistent(self):
        model = StubModel({"PHRASING_A": "2023", "PHRASING_B": "2024", "PHRASING_C": "2023"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert result.fail_mode == "DATA_CLAIM_INCONSISTENT"

    def test_requires_two_variants(self):
        with pytest.raises(ValueError):
            DataProvenanceConsistencyTest(
                data_query_variants=[("only", "Q:")],
                extraction_method="numeric",
            )

    def test_claim_type_behavioral(self):
        model = StubModel({"PHRASING_A": "2023", "PHRASING_B": "2023", "PHRASING_C": "2023"})
        result = self._make_test().run(model)
        assert result.claim_type == ClaimType.BEHAVIORAL

    def test_evidence_records_each_variant(self):
        model = StubModel({"PHRASING_A": "2023", "PHRASING_B": "2023", "PHRASING_C": "2023"})
        result = self._make_test().run(model)
        keys = [k for k, _ in result.evidence.items()]
        assert "q1_extracted" in keys
        assert "q2_extracted" in keys
        assert "q3_extracted" in keys


# ── OutputScopeConsistencyTest ────────────────────────────────────────────────

class TestOutputScopeConsistencyTest:
    def _make_test(self):
        return OutputScopeConsistencyTest(
            capability_claim_prompts=[
                ("q1", "Q1: can you?"),
                ("q2", "Q2: can you?"),
            ],
            task_prompt="TASK:",
            task_extraction_method="numeric",
            task_expected="42",
            expected_capability="yes",
        )

    def test_pass_claims_yes_and_succeeds(self):
        model = StubModel({"Q1": "yes", "Q2": "yes", "TASK": "42"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.PASS

    def test_fail_scope_inconsistent(self):
        model = StubModel({"Q1": "yes", "Q2": "no", "TASK": "42"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert result.fail_mode == "SCOPE_INCONSISTENT"

    def test_fail_scope_overclaim(self):
        # Claims yes but task fails
        model = StubModel({"Q1": "yes", "Q2": "yes", "TASK": "wrong answer"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "SCOPE_OVERCLAIM" in result.fail_mode

    def test_fail_scope_underclaim(self):
        # Claims no but task succeeds
        model = StubModel({"Q1": "no", "Q2": "no", "TASK": "42"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "SCOPE_UNDERCLAIM" in result.fail_mode

    def test_pass_claims_no_and_fails_task(self):
        # Model says it can't, and indeed can't — consistent
        model = StubModel({"Q1": "no", "Q2": "no", "TASK": "I cannot do that"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.PASS

    def test_requires_at_least_one_claim_prompt(self):
        with pytest.raises(ValueError):
            OutputScopeConsistencyTest(
                capability_claim_prompts=[],
                task_prompt="TASK:", task_extraction_method="numeric",
                task_expected="42", expected_capability="yes",
            )

    def test_claim_type_behavioral(self):
        model = StubModel({"Q1": "yes", "Q2": "yes", "TASK": "42"})
        result = self._make_test().run(model)
        assert result.claim_type == ClaimType.BEHAVIORAL
