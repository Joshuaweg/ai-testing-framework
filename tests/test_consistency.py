import pytest
from ai_test_framework.core.base_test import ClaimType, Verdict, Evidence
from ai_test_framework.tests.consistency import (
    FramingVariant,
    SurfaceFramingConsistencyTest,
    RepeatedPromptConsistencyTest,
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


def _three_variants():
    return [
        FramingVariant(label="v1", prompt="PROMPT_V1: how many?", expected="4"),
        FramingVariant(label="v2", prompt="PROMPT_V2: how many?", expected="4"),
        FramingVariant(label="v3", prompt="PROMPT_V3: how many?", expected="4"),
    ]


class TestSurfaceFramingConsistencyTest:
    def test_pass_all_correct(self):
        model = StubModel({"PROMPT_V1": "The answer is 4.", "PROMPT_V2": "4", "PROMPT_V3": "Answer: 4"})
        test = SurfaceFramingConsistencyTest(variants=_three_variants(), extraction_method="numeric", expected_answer="4")
        result = test.run(model)
        assert result.verdict == Verdict.PASS
        assert result.fail_mode is None

    def test_fail_inconsistent(self):
        model = StubModel({"PROMPT_V1": "4", "PROMPT_V2": "5", "PROMPT_V3": "4"})
        test = SurfaceFramingConsistencyTest(variants=_three_variants(), extraction_method="numeric", expected_answer="4")
        result = test.run(model)
        assert result.verdict == Verdict.FAIL
        assert result.fail_mode == "INCONSISTENT"

    def test_fail_consistent_wrong(self):
        model = StubModel({"PROMPT_V1": "7", "PROMPT_V2": "7", "PROMPT_V3": "7"})
        test = SurfaceFramingConsistencyTest(variants=_three_variants(), extraction_method="numeric", expected_answer="4")
        result = test.run(model)
        assert result.verdict == Verdict.FAIL
        assert result.fail_mode == "CONSISTENT_WRONG"

    def test_evidence_records_variant_keys(self):
        model = StubModel({"PROMPT_V1": "4", "PROMPT_V2": "4", "PROMPT_V3": "4"})
        test = SurfaceFramingConsistencyTest(variants=_three_variants(), extraction_method="numeric", expected_answer="4")
        result = test.run(model)
        keys = [k for k, _ in result.evidence.items()]
        assert "v1_extracted" in keys
        assert "v2_extracted" in keys
        assert "v3_extracted" in keys

    def test_claim_type_behavioral(self):
        model = StubModel({"PROMPT_V1": "4", "PROMPT_V2": "4", "PROMPT_V3": "4"})
        test = SurfaceFramingConsistencyTest(variants=_three_variants(), extraction_method="numeric", expected_answer="4")
        result = test.run(model)
        assert result.claim_type == ClaimType.BEHAVIORAL

    def test_requires_two_variants(self):
        with pytest.raises(ValueError):
            SurfaceFramingConsistencyTest(
                variants=[FramingVariant("v1", "p", "4")],
                extraction_method="numeric",
                expected_answer="4",
            )


class TestRepeatedPromptConsistencyTest:
    def test_pass_stable(self):
        model = StubModel({"Q:": "The answer is 4."})
        test = RepeatedPromptConsistencyTest(prompt="Q: how many?", extraction_method="numeric", expected_answer="4", n_runs=3)
        result = test.run(model)
        assert result.verdict == Verdict.PASS

    def test_fail_unstable(self):
        class AlternatingStub:
            name = "alt"
            def __init__(self): self._count = 0
            def generate(self, prompt, temperature=0.0):
                self._count += 1
                return "4" if self._count % 2 == 0 else "5"

        test = RepeatedPromptConsistencyTest(prompt="Q: how many?", extraction_method="numeric", expected_answer="4", n_runs=3)
        result = test.run(AlternatingStub())
        assert result.verdict == Verdict.FAIL
        assert result.fail_mode == "UNSTABLE"

    def test_requires_n_runs_minimum_two(self):
        with pytest.raises(ValueError):
            RepeatedPromptConsistencyTest(prompt="p", extraction_method="numeric", expected_answer="4", n_runs=1)
