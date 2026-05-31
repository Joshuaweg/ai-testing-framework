import pytest
from ai_test_framework.core.base_test import ClaimType, Verdict
from ai_test_framework.tests.explanatory import RationaleConsistencyTest, ContrastiveExplanationTest


class StubModel:
    name = "stub"

    def __init__(self, responses: dict):
        self._responses = responses

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        for key, val in self._responses.items():
            if key in prompt:
                return val
        return "I don't know"


# ── RationaleConsistencyTest ──────────────────────────────────────────────────

class TestRationaleConsistencyTest:
    def _make_test(self):
        return RationaleConsistencyTest(
            base_prompt="BASE_Q:",
            why_variants=[
                ("formal", "FORMAL_WHY: answer is {answer}"),
                ("causal", "CAUSAL_WHY: answer is {answer}"),
            ],
            extraction_method="numeric",
            expected_answer="4",
        )

    def test_pass_when_rederived_matches(self):
        # Base returns 4; rationale prompts return text; re-derive prompts return 4
        model = StubModel({
            "BASE_Q":      "The answer is 4.",
            "FORMAL_WHY":  "Because subtraction gives 4.",
            "CAUSAL_WHY":  "Arithmetic yields 4.",
            "Reasoning:":  "4",  # re-derive prompt contains "Reasoning:"
        })
        result = self._make_test().run(model)
        assert result.verdict == Verdict.PASS

    def test_fail_base_wrong(self):
        model = StubModel({
            "BASE_Q":     "7",
            "Reasoning:": "4",
        })
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "BASE_WRONG" in result.fail_mode

    def test_fail_rationale_decoupled(self):
        # Re-derive returns something different from the original prediction
        model = StubModel({
            "BASE_Q":      "4",
            "FORMAL_WHY":  "The explanation is 4.",
            "CAUSAL_WHY":  "Because 4.",
            "Reasoning:":  "7",  # re-derive diverges
        })
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "RATIONALE_DECOUPLED" in result.fail_mode

    def test_requires_at_least_one_variant(self):
        with pytest.raises(ValueError):
            RationaleConsistencyTest(
                base_prompt="Q:", why_variants=[],
                extraction_method="numeric",
            )

    def test_claim_type_behavioral(self):
        model = StubModel({"BASE_Q": "4", "Reasoning:": "4"})
        result = self._make_test().run(model)
        assert result.claim_type == ClaimType.BEHAVIORAL

    def test_evidence_records_rationale_and_rederived(self):
        model = StubModel({
            "BASE_Q":     "4",
            "FORMAL_WHY": "Because 4.",
            "CAUSAL_WHY": "Arithmetic 4.",
            "Reasoning:": "4",
        })
        result = self._make_test().run(model)
        keys = [k for k, _ in result.evidence.items()]
        assert "formal_rationale" in keys
        assert "formal_rederived" in keys
        assert "causal_rederived" in keys


# ── ContrastiveExplanationTest ────────────────────────────────────────────────

class TestContrastiveExplanationTest:
    def _make_test(self):
        return ContrastiveExplanationTest(
            case_a_prompt="CASE_A:",
            case_a_expected="5",
            case_b_prompt="CASE_B:",
            case_b_expected="11",
            contrastive_prompt="CONTRAST:",
            key_differentiators=["spent", "deposit"],
            extraction_method="numeric",
        )

    def test_pass(self):
        model = StubModel({
            "CASE_A":   "5",
            "CASE_B":   "11",
            "CONTRAST": "Because money was spent in one case and deposited in another.",
        })
        result = self._make_test().run(model)
        assert result.verdict == Verdict.PASS

    def test_fail_wrong_outcome_a(self):
        model = StubModel({"CASE_A": "3", "CASE_B": "11", "CONTRAST": "spent"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "WRONG_OUTCOME_A" in result.fail_mode

    def test_fail_wrong_outcome_b(self):
        model = StubModel({"CASE_A": "5", "CASE_B": "3", "CONTRAST": "spent"})
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "WRONG_OUTCOME_B" in result.fail_mode

    def test_fail_misidentification(self):
        model = StubModel({
            "CASE_A":   "5",
            "CASE_B":   "11",
            "CONTRAST": "The numbers are different because math.",
        })
        result = self._make_test().run(model)
        assert result.verdict == Verdict.FAIL
        assert "CONTRASTIVE_MISIDENTIFICATION" in result.fail_mode

    def test_key_differentiator_case_insensitive(self):
        model = StubModel({
            "CASE_A":   "5",
            "CASE_B":   "11",
            "CONTRAST": "SPENT is the operative word here.",
        })
        result = self._make_test().run(model)
        assert result.verdict == Verdict.PASS

    def test_claim_type_causal(self):
        model = StubModel({"CASE_A": "5", "CASE_B": "11", "CONTRAST": "spent"})
        result = self._make_test().run(model)
        assert result.claim_type == ClaimType.CAUSAL
