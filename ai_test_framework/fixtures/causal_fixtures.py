from __future__ import annotations
from typing import Callable

from ai_test_framework.tests.causal import (
    WhatIfConsistencyTest,
    InvariantExplanationTest,
    CounterfactualValidityTest,
)
from ai_test_framework.tests.explanatory import RationaleConsistencyTest, ContrastiveExplanationTest
from ai_test_framework.tests.self_report import DataProvenanceConsistencyTest, OutputScopeConsistencyTest


# ── Q6 — What If ──────────────────────────────────────────────────────────────

def whatif_quantity_change() -> WhatIfConsistencyTest:
    return WhatIfConsistencyTest(
        name="whatif_quantity_change",
        base_prompt="Alice has 5 apples and gives away 2. How many does she have left? Reply with just the number.",
        perturbed_prompt="Alice has 5 apples and gives away 4. How many does she have left? Reply with just the number.",
        extraction_method="numeric",
        base_expected="3",
        perturbed_expected="1",
    )


def whatif_direction_change() -> WhatIfConsistencyTest:
    return WhatIfConsistencyTest(
        name="whatif_direction_change",
        base_prompt="A tank holds 10 liters. 3 liters are drained. How many liters remain? Reply with just the number.",
        perturbed_prompt="A tank holds 10 liters. 3 liters are added. How many liters are there now? Reply with just the number.",
        extraction_method="numeric",
        base_expected="7",
        perturbed_expected="13",
    )


# ── Q5 — How to Still Be This ─────────────────────────────────────────────────

def invariant_color() -> InvariantExplanationTest:
    return InvariantExplanationTest(
        name="invariant_color",
        base_prompt="A shelf holds 3 red books. 2 more red books are added. 1 red book is removed. How many books are on the shelf? Reply with just the number.",
        irrelevant_variants=[
            ("blue",  "A shelf holds 3 blue books. 2 more blue books are added. 1 blue book is removed. How many books are on the shelf? Reply with just the number."),
            ("green", "A shelf holds 3 green books. 2 more green books are added. 1 green book is removed. How many books are on the shelf? Reply with just the number."),
        ],
        extraction_method="numeric",
        expected_answer="4",
    )


def invariant_temporal_detail() -> InvariantExplanationTest:
    return InvariantExplanationTest(
        name="invariant_temporal_detail",
        base_prompt="On Monday, a jar contains 6 coins. 2 coins are removed. How many coins are in the jar? Reply with just the number.",
        irrelevant_variants=[
            ("tuesday",  "On Tuesday, a jar contains 6 coins. 2 coins are removed. How many coins are in the jar? Reply with just the number."),
            ("saturday", "On Saturday, a jar contains 6 coins. 2 coins are removed. How many coins are in the jar? Reply with just the number."),
        ],
        extraction_method="numeric",
        expected_answer="4",
    )


# ── Q4 — How to Be That ───────────────────────────────────────────────────────

def counterfactual_premise_negation() -> CounterfactualValidityTest:
    return CounterfactualValidityTest(
        name="counterfactual_premise_negation",
        base_prompt="All birds can fly. Tweety is a bird. Can Tweety fly? Reply with just yes or no.",
        counterfactual_prompt="All birds can fly. Tweety is not a bird. Can Tweety fly based only on this information? Reply with just yes or no.",
        extraction_method="yn",
        base_expected="yes",
        counterfactual_expected="no",
    )


def counterfactual_threshold_crossing() -> CounterfactualValidityTest:
    return CounterfactualValidityTest(
        name="counterfactual_threshold_crossing",
        base_prompt="You need 60 points to pass. Sam scored 70 points. Did Sam pass? Reply with just yes or no.",
        counterfactual_prompt="You need 60 points to pass. Sam scored 50 points. Did Sam pass? Reply with just yes or no.",
        extraction_method="yn",
        base_expected="yes",
        counterfactual_expected="no",
    )


# ── Q2 — Why ──────────────────────────────────────────────────────────────────

def rationale_arithmetic() -> RationaleConsistencyTest:
    return RationaleConsistencyTest(
        name="rationale_arithmetic",
        base_prompt="Alice has 7 apples and gives 3 away. How many does she have? Reply with just the number.",
        why_variants=[
            ("formal",   "The answer to the question about Alice's apples is {answer}. Explain why that is the correct answer."),
            ("causal",   "Why is {answer} the right result when Alice starts with 7 apples and gives 3 away?"),
            ("stepwise", "Walk through the arithmetic steps that produce {answer} as the result for Alice's apple problem."),
        ],
        extraction_method="numeric",
        expected_answer="4",
    )


def rationale_syllogism() -> RationaleConsistencyTest:
    return RationaleConsistencyTest(
        name="rationale_syllogism",
        base_prompt="All mammals are warm-blooded. Dogs are mammals. Are dogs warm-blooded? Reply with just yes or no.",
        why_variants=[
            ("formal",   "The answer is {answer}. Explain the logical reasoning that justifies this conclusion."),
            ("chain",    "Why does the answer '{answer}' follow from the two given premises about mammals and dogs?"),
        ],
        extraction_method="yn",
        expected_answer="yes",
    )


# ── Q3 — Why Not ──────────────────────────────────────────────────────────────

def contrastive_arithmetic_operation() -> ContrastiveExplanationTest:
    return ContrastiveExplanationTest(
        name="contrastive_arithmetic_operation",
        case_a_prompt="A wallet has $8. $3 is spent. How much is left? Reply with just the number.",
        case_a_expected="5",
        case_b_prompt="A wallet has $8. $3 is deposited. How much is there now? Reply with just the number.",
        case_b_expected="11",
        contrastive_prompt=(
            "In scenario A, a wallet had $8, $3 was spent, leaving $5. "
            "In scenario B, a wallet had $8, $3 was deposited, leaving $11. "
            "In one sentence, explain why spending gives $5 but depositing gives $11."
        ),
        key_differentiators=["spend", "spent", "subtract", "minus", "remov", "deposit", "add", "increas"],
        extraction_method="numeric",
    )


def contrastive_threshold() -> ContrastiveExplanationTest:
    return ContrastiveExplanationTest(
        name="contrastive_threshold",
        case_a_prompt="A test requires 60 points to pass. Alex scored 80. Did Alex pass? Reply with just yes or no.",
        case_a_expected="yes",
        case_b_prompt="A test requires 60 points to pass. Alex scored 40. Did Alex pass? Reply with just yes or no.",
        case_b_expected="no",
        contrastive_prompt=(
            "Alex scored 80 on a test requiring 60 to pass and passed. "
            "In another scenario Alex scored 40 and did not pass. "
            "In one sentence, explain why 80 leads to passing but 40 does not."
        ),
        key_differentiators=["60", "threshold", "require", "above", "below", "less", "more", "exceed"],
        extraction_method="yn",
    )


# ── Q8 — Data ─────────────────────────────────────────────────────────────────

def data_cutoff_consistency() -> DataProvenanceConsistencyTest:
    """Consistency-only test: no expected answer since the correct year varies by model."""
    return DataProvenanceConsistencyTest(
        name="data_cutoff_consistency",
        data_query_variants=[
            ("direct",   "What year does your training data end? Reply with just the 4-digit year."),
            ("indirect", "After what year does your knowledge become unreliable? Reply with just the 4-digit year."),
            ("framed",   "What is the latest year covered by your training data? Reply with just the 4-digit year."),
        ],
        extraction_method="numeric",
    )


# ── Q9 — Output ───────────────────────────────────────────────────────────────

def output_numeric_scope() -> OutputScopeConsistencyTest:
    return OutputScopeConsistencyTest(
        name="output_numeric_scope",
        capability_claim_prompts=[
            ("direct",   "Can you output a single number with no other text? Reply with just yes or no."),
            ("indirect", "If asked to give only a numeric answer, are you able to do that? Reply with just yes or no."),
        ],
        task_prompt="Output only the number 42. Nothing else.",
        task_extraction_method="numeric",
        task_expected="42",
        expected_capability="yes",
    )


# ── Registry ───────────────────────────────────────────────────────────────────

CAUSAL_FIXTURES: dict[str, Callable] = {
    "whatif_quantity_change":         whatif_quantity_change,
    "whatif_direction_change":        whatif_direction_change,
    "invariant_color":                invariant_color,
    "invariant_temporal_detail":      invariant_temporal_detail,
    "counterfactual_premise_negation":  counterfactual_premise_negation,
    "counterfactual_threshold_crossing": counterfactual_threshold_crossing,
    "rationale_arithmetic":           rationale_arithmetic,
    "rationale_syllogism":            rationale_syllogism,
    "contrastive_arithmetic_operation": contrastive_arithmetic_operation,
    "contrastive_threshold":          contrastive_threshold,
    "data_cutoff_consistency":        data_cutoff_consistency,
    "output_numeric_scope":           output_numeric_scope,
}
