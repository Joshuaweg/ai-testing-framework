from __future__ import annotations
from typing import Callable
from ai_test_framework.tests.consistency import SurfaceFramingConsistencyTest, FramingVariant


def apples_and_oranges() -> SurfaceFramingConsistencyTest:
    return SurfaceFramingConsistencyTest(
        name="apples_and_oranges",
        variants=[
            FramingVariant("original", "Alice has 3 apples. Bob gives her 2 more. She eats 1. How many apples does Alice have? Reply with just the number.", "4"),
            FramingVariant("renamed",  "Carlos has 3 oranges. Diana gives him 2 more. He eats 1. How many oranges does Carlos have? Reply with just the number.", "4"),
            FramingVariant("money",    "A wallet has $3. Someone adds $2. Then $1 is spent. How much money is left in the wallet? Reply with just the number.", "4"),
            FramingVariant("objects",  "A shelf holds 3 books. 2 more are placed on it. 1 is removed. How many books are on the shelf? Reply with just the number.", "4"),
        ],
        extraction_method="numeric",
        expected_answer="4",
    )


def simple_multiplication() -> SurfaceFramingConsistencyTest:
    return SurfaceFramingConsistencyTest(
        name="simple_multiplication",
        variants=[
            FramingVariant("original", "If there are 3 boxes and each box contains 4 apples, how many apples are there in total? Reply with just the number.", "12"),
            FramingVariant("renamed",  "A farmer has 3 coops. Each coop holds 4 chickens. How many chickens does the farmer have? Reply with just the number.", "12"),
            FramingVariant("abstract", "What is 3 multiplied by 4? Reply with just the number.", "12"),
        ],
        extraction_method="numeric",
        expected_answer="12",
    )


def transitive_logic() -> SurfaceFramingConsistencyTest:
    return SurfaceFramingConsistencyTest(
        name="transitive_logic",
        variants=[
            FramingVariant("original", "Alice is taller than Bob. Bob is taller than Carol. Is Alice taller than Carol? Reply with just yes or no.", "yes"),
            FramingVariant("renamed",  "Xavier is older than Yara. Yara is older than Zeke. Is Xavier older than Zeke? Reply with just yes or no.", "yes"),
            FramingVariant("abstract", "Object A weighs more than object B. Object B weighs more than object C. Does object A weigh more than object C? Reply with just yes or no.", "yes"),
            FramingVariant("geography","Mountain P is higher than mountain Q. Mountain Q is higher than mountain R. Is mountain P higher than mountain R? Reply with just yes or no.", "yes"),
        ],
        extraction_method="yn",
        expected_answer="yes",
    )


def negation_logic() -> SurfaceFramingConsistencyTest:
    return SurfaceFramingConsistencyTest(
        name="negation_logic",
        variants=[
            FramingVariant("original", "All glorbits are fizzy. Snorps are not glorbits. Are snorps fizzy? Reply with just yes or no.", "no"),
            FramingVariant("renamed",  "Every brixon is luminous. Treffles are not brixons. Are treffles luminous? Reply with just yes or no.", "no"),
            FramingVariant("abstract", "All members of group X have property P. Item Z is not a member of group X. Does item Z have property P? Reply with just yes or no.", "no"),
        ],
        extraction_method="yn",
        expected_answer="no",
    )


def rate_problem() -> SurfaceFramingConsistencyTest:
    return SurfaceFramingConsistencyTest(
        name="rate_problem",
        variants=[
            FramingVariant("original", "If 5 machines take 5 minutes to make 5 widgets, how many minutes does it take 1 machine to make 1 widget? Reply with just the number.", "5"),
            FramingVariant("renamed",  "If 5 workers take 5 hours to dig 5 holes, how many hours does it take 1 worker to dig 1 hole? Reply with just the number.", "5"),
            FramingVariant("abstract", "5 units of resource produce 5 units of output in 5 units of time. How many time units does 1 resource unit need to produce 1 output unit? Reply with just the number.", "5"),
        ],
        extraction_method="numeric",
        expected_answer="5",
    )


FIXTURES: dict[str, Callable[[], SurfaceFramingConsistencyTest]] = {
    "apples_and_oranges":    apples_and_oranges,
    "simple_multiplication": simple_multiplication,
    "transitive_logic":      transitive_logic,
    "negation_logic":        negation_logic,
    "rate_problem":          rate_problem,
}
