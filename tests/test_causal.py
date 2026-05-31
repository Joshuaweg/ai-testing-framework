import pytest
from ai_test_framework.core.base_test import ClaimType, Verdict
from ai_test_framework.tests.causal import (
    WhatIfConsistencyTest,
    InvariantExplanationTest,
    CounterfactualValidityTest,
)


class StubModel:
    name = "stub"

    def __init__(self, responses: dict):
        self._responses = responses

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        for key, val in self._responses.items():
            if key in prompt:
                return val
        return "I don't know"


# ── WhatIfConsistencyTest ─────────────────────────────────────────────────────

class TestWhatIfConsistencyTest:
    def _make_test(self):
        return WhatIfConsistencyTest(
            base_prompt="BASE: how many?",
            perturbed_prompt="PERTURBED: how many?",
            extraction_method="numeric",
            base_expected="3",
            perturbed_expected="1",
        )

    def test_pass(self):
        model = StubModel({"BASE": "3", "PERTURBED": "1"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.PASS
        assert result.fail_mode is None

    def test_fail_base_wrong(self):
        model = StubModel({"BASE": "5", "PERTURBED": "1"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "BASE_WRONG" in result.fail_mode

    def test_fail_insensitive(self):
        model = StubModel({"BASE": "3", "PERTURBED": "3"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "INSENSITIVE" in result.fail_mode

    def test_fail_perturbed_wrong(self):
        # Base correct, perturbed changed but not to the right value
        model = StubModel({"BASE": "3", "PERTURBED": "7"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "PERTURBED_WRONG" in result.fail_mode

    def test_claim_type_causal(self):
        model = StubModel({"BASE": "3", "PERTURBED": "1"})
        result = self._make_test().run(model)
        assert result.claim_type == ClaimType.CAUSAL

    def test_evidence_keys(self):
        model = StubModel({"BASE": "3", "PERTURBED": "1"})
        result = self._make_test().run(model)
        keys = [k for k, _ in result.evidence.items()]
        assert "base_extracted" in keys
        assert "perturbed_extracted" in keys


# ── InvariantExplanationTest ──────────────────────────────────────────────────

class TestInvariantExplanationTest:
    def _make_test(self):
        return InvariantExplanationTest(
            base_prompt="BASE:",
            irrelevant_variants=[
                ("v1", "VARIANT1:"),
                ("v2", "VARIANT2:"),
            ],
            extraction_method="numeric",
            expected_answer="4",
        )

    def test_pass(self):
        model = StubModel({"BASE": "4", "VARIANT1": "4", "VARIANT2": "4"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.PASS

    def test_fail_base_wrong(self):
        model = StubModel({"BASE": "7", "VARIANT1": "4", "VARIANT2": "4"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "BASE_WRONG" in result.fail_mode

    def test_fail_spurious_sensitivity(self):
        model = StubModel({"BASE": "4", "VARIANT1": "5", "VARIANT2": "4"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "SPURIOUS_SENSITIVITY" in result.fail_mode

    def test_requires_at_least_one_variant(self):
        with pytest.raises(ValueError):
            InvariantExplanationTest(
                base_prompt="BASE:", irrelevant_variants=[],
                extraction_method="numeric", expected_answer="4",
            )

    def test_claim_type_causal(self):
        model = StubModel({"BASE": "4", "VARIANT1": "4", "VARIANT2": "4"})
        result = self._make_test().run(model)
        assert result.claim_type == ClaimType.CAUSAL


# ── CounterfactualValidityTest ────────────────────────────────────────────────

class TestCounterfactualValidityTest:
    def _make_test(self):
        return CounterfactualValidityTest(
            base_prompt="BASE:",
            counterfactual_prompt="CF:",
            extraction_method="yn",
            base_expected="yes",
            counterfactual_expected="no",
        )

    def test_pass(self):
        model = StubModel({"BASE": "yes", "CF": "no"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.PASS

    def test_fail_base_wrong(self):
        model = StubModel({"BASE": "no", "CF": "no"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "BASE_WRONG" in result.fail_mode

    def test_fail_counterfactual_invalid(self):
        model = StubModel({"BASE": "yes", "CF": "yes"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "COUNTERFACTUAL_INVALID" in result.fail_mode

    def test_evidence_keys(self):
        model = StubModel({"BASE": "yes", "CF": "no"})
        result = self._make_test().run(model)
        keys = [k for k, _ in result.evidence.items()]
        assert "base_extracted" in keys
        assert "counterfactual_extracted" in keys

    def test_claim_type_causal(self):
        model = StubModel({"BASE": "yes", "CF": "no"})
        result = self._make_test().run(model)
        assert result.claim_type == ClaimType.CAUSAL
