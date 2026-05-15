# AI Model Testing Framework

A pytest-style evaluation framework for testing AI model behavior — not just answer correctness, but **process validity**: whether a model gets the right answer for the right reasons.

## Why this exists

Standard benchmarks test the destination, not the path. A model can score 90% on a reasoning benchmark through genuine reasoning, pattern matching, statistical shortcuts, or lucky guessing — and existing benchmarks cannot distinguish between them.

This framework fills that gap with formal, reproducible tests that verify reasoning consistency, robustness to surface variation, and other behavioral properties that benchmarks miss. It occupies the space between training-time interpretability tools (Goodfire Silico) and production monitoring (Langfuse): **pre-deployment model qualification** — the test suite you run before you commit to a model, the way you run pytest before you ship code.

## Quick start

```bash
pip install -e .

# Run all built-in tests against a local Ollama model
python run_tests.py

# Run against a specific model
python run_tests.py --model llama3.1:8b

# Run a single fixture
python run_tests.py --fixture transitive_logic

# List available fixtures
python run_tests.py --list
```

Exit code mirrors pytest: `0` = all pass, `1` = any failure.

## Architecture

```
ai_test_framework/
├── core/
│   ├── model.py        # Model + ModelConfig — Ollama backend, full config snapshot
│   ├── base_test.py    # TestResult, BaseTest, ClaimType, Evidence
│   ├── runner.py       # TestSuite, ComparisonRunner, terminal reporters
│   ├── extractors.py   # Parse model responses: numeric, yn, letter, exact
│   ├── adversarial.py  # AdversarialValidator — mutation testing for the test suite
│   └── coverage.py     # CoverageAnalyzer — failure mode coverage reporting
├── tests/
│   ├── consistency.py  # SurfaceFramingConsistencyTest, RepeatedPromptConsistencyTest,
│   │                   # StatisticalConsistencyTest
│   └── format.py       # FormatContractTest
└── fixtures/
    ├── logic_problems.py  # 5 built-in multi-framing fixtures
    └── generator.py       # generate_variants, VariantTemplate
run_tests.py            # CLI entry point
```

### The ClaimType system

Every test declares the epistemological claim it makes:

| ClaimType | What it proves |
|---|---|
| `BEHAVIORAL` | The model produced output X for input Y. Input/output only. |
| `OBSERVATIONAL` | Information is encoded in representations. Correlation, not causation. |
| `CAUSAL` | Removing or changing component X changes output Y. Requires weight access. |

A passing OBSERVATIONAL test does **not** prove the model uses that information causally. This distinction is preserved in all test output.

---

## Tests

### SurfaceFramingConsistencyTest

Presents the same logical problem with different surface framing and verifies the model produces the same answer across all variants.

**Rationale:** If a model is reasoning, the answer is invariant to surface framing. If it is pattern-matching, different framings trigger different patterns and produce different answers.

- ClaimType: `BEHAVIORAL`
- Fail — `INCONSISTENT`: any variant differs → signals pattern matching
- Fail — `CONSISTENT_WRONG`: all agree but are wrong → consistent but incorrect reasoning
- Requires `temperature=0`

```python
from ai_test_framework.tests import SurfaceFramingConsistencyTest, FramingVariant
from ai_test_framework import Model, TestSuite

variants = [
    FramingVariant("v1", "If 5 machines take 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets? Reply with just the number.", "5"),
    FramingVariant("v2", "If 5 workers take 5 hours to dig 5 holes, how long for 100 workers to dig 100 holes? Reply with just the number.", "5"),
]
suite = TestSuite("my_suite")
suite.add(SurfaceFramingConsistencyTest(variants=variants, extraction_method="numeric", expected_answer="5"))
suite.run(Model.ollama("llama3.1:8b"))
```

### RepeatedPromptConsistencyTest

Runs the exact same prompt N times at `temperature=0` and verifies all responses are identical. Use as a baseline determinism check before any test campaign.

- ClaimType: `BEHAVIORAL`
- Fail — `UNSTABLE`: any run differs → non-determinism or seed issue

### StatisticalConsistencyTest

Runs a prompt N times at a configurable temperature and passes when `consistency_rate >= threshold`. Addresses the non-determinism gap: tests behavioral stability at realistic operating temperatures rather than forcing `temperature=0`.

- ClaimType: `BEHAVIORAL`
- Claim: "This model answers consistently ≥ {threshold}% of the time at temperature={temperature}"
- Fail — `INCONSISTENT`: consistency rate below threshold
- Fail — `CONSISTENT_WRONG`: consistent enough but the modal answer is wrong
- Evidence includes: `consistency_rate`, `correct_rate`, `modal_answer`

```python
from ai_test_framework.tests import StatisticalConsistencyTest

suite.add(StatisticalConsistencyTest(
    prompt="Alice has 3 apples and gets 1 more. How many does she have? Reply with just the number.",
    extraction_method="numeric",
    expected_answer="4",
    n_runs=10,
    threshold=0.8,
    temperature=0.7,
))
```

### FormatContractTest

Tests whether a model reliably follows format instructions across different phrasings of the same directive. Analogous to contract testing: verifies the model's implicit output format contract holds regardless of how the instruction is worded.

- ClaimType: `BEHAVIORAL`
- Fail — `FORMAT_BREACH`: extraction returned None on one or more variants
- Fail — `INCONSISTENT`: phrasing of the format instruction affected the answer
- Fail — `CONSISTENT_WRONG`: format followed correctly but answer is wrong

```python
from ai_test_framework.tests import FormatContractTest, NUMERIC_INSTRUCTIONS

suite.add(FormatContractTest(
    base_prompt="Alice has 3 apples. Bob gives her 2 more. She eats 1. How many apples does Alice have?",
    format_instructions=NUMERIC_INSTRUCTIONS,  # 4 built-in phrasings
    extraction_method="numeric",
    expected_answer="4",
))
```

Built-in instruction sets: `NUMERIC_INSTRUCTIONS` (4 phrasings), `YN_INSTRUCTIONS` (3), `LETTER_INSTRUCTIONS` (3). Custom instructions use `FormatInstruction(label, instruction)`.

### Built-in fixtures

| Fixture | Logic type | Variants | Extraction |
|---|---|---|---|
| `apples_and_oranges` | Arithmetic (give/receive) | 4 | numeric |
| `simple_multiplication` | Multiplication | 3 | numeric |
| `transitive_logic` | Transitivity (A > B > C) | 4 | yn |
| `negation_logic` | Syllogistic reasoning | 3 | yn |
| `rate_problem` | Rate × time = quantity | 3 | numeric |

```python
from ai_test_framework.fixtures.logic_problems import apples_and_oranges
suite.add(apples_and_oranges())
```

---

## Tools

### ComparisonRunner — model regression testing

Runs the same `TestSuite` against two models and diffs the results. Use when upgrading models, changing quantization, or tuning sampling parameters.

```python
from ai_test_framework import ComparisonRunner, Model

runner = ComparisonRunner(suite)
result = runner.compare(
    model_a=Model.ollama("llama3.1:8b"),
    model_b=Model.ollama("qwen3:8b"),
    label_a="llama3.1:8b",
    label_b="qwen3:8b",
)
# result.regressions, result.improvements, result.unchanged
```

Output:
```
  Comparison  [my_suite]
  A: llama3.1:8b
  B: qwen3:8b
  ----------------------------------------------------------------
  Test                    A       B       Change
  apples_and_oranges      PASS    PASS    —
  negation_logic          PASS    FAIL    ↓ REGRESSED
  rate_problem            FAIL    PASS    ↑ IMPROVED
  ----------------------------------------------------------------
  A: 4/5 passed  |  B: 4/5 passed  |  1 regression(s)  |  1 improvement(s)
```

**ModelConfig** — for comparing full configuration snapshots (quantization, sampling parameters, system prompt):

```python
from ai_test_framework import ModelConfig, Model

config = ModelConfig(model="llama3.1:8b", top_p=0.9, num_ctx=4096, system_prompt="You are a careful reasoner.")
model = Model.from_config(config)
```

`ModelConfig` fields: `model`, `top_k`, `top_p`, `repeat_penalty`, `seed`, `num_ctx`, `system_prompt`.

### AdversarialValidator — mutation testing for the test suite

Verifies that the test suite actually catches failures. Runs probes — tests with deliberately broken configurations — and checks that each produces the expected verdict. If a probe passes when it should fail, the test design is too weak.

```python
from ai_test_framework import AdversarialValidator, build_probes, wrong_answer_probe
from ai_test_framework.fixtures.logic_problems import apples_and_oranges

fixture = apples_and_oranges()
probes = build_probes(fixture)  # auto-generates standard probe set

validator = AdversarialValidator()
result = validator.validate(model, probes)
# result.all_caught, result.missed
```

Output:
```
  + CAUGHT   wrong_answer_probe(apples_and_oranges)
             broken: expected_answer forcibly set to '__ADVERSARIAL_WRONG_ANSWER__'
             expected: FAIL  |  got: FAIL
  ----------------------------------------------------------------
  1/1 probes caught  |  0 test design failure(s)
```

Probe factories:
- `wrong_answer_probe(test, wrong_answer)` — replaces `expected_answer`; must fail
- `high_threshold_probe(test)` — forces `threshold=1.0` on `StatisticalConsistencyTest`; must fail at temp>0
- `build_probes(test)` — auto-generates the standard set for any test

### CoverageAnalyzer — behavioral coverage metric

Scores a `TestSuite` against a catalogue of 12 known AI model failure modes. Analogous to a code coverage report — shows which failure modes are exercised and which are gaps.

```python
from ai_test_framework import CoverageAnalyzer

report = CoverageAnalyzer().analyze(suite)
# report.coverage_rate, report.uncovered, report.by_severity(Severity.CRITICAL)
```

Output:
```
  Coverage Report  [my_suite]  3/12 (25%)
  ----------------------------------------------------------------

  [CRITICAL]
  x contradiction_handling      not covered
  x counterfactual_sensitivity  not covered
  + surface_framing             apples_and_oranges, transitive_logic

  [HIGH]
  x reasoning_chain_validity    not covered
  ...

  Critical: 1/3  |  High: 0/4  |  Medium: 1/3  |  Low: 2/2
```

Failure modes catalogued: `surface_framing`, `contradiction_handling`, `counterfactual_sensitivity` (CRITICAL) · `reasoning_chain_validity`, `irrelevant_fact_sensitivity`, `sycophancy`, `refusal_consistency` (HIGH) · `temperature_stability`, `authority_bias`, `calibration` (MEDIUM) · `determinism`, `output_format_compliance` (LOW).

When adding a new test class, register it in `COVERAGE_MAP` in `core/coverage.py`.

### VariantTemplate — programmatic fixture generation

Generates `FramingVariant` lists from a prompt template and substitution dicts. Avoids copy-pasting prompt text when building large fixture libraries.

```python
from ai_test_framework import VariantTemplate

template = VariantTemplate(
    prompt_template=(
        "{name} has {start} {thing}s. Someone gives {give} more, {use} is used. "
        "How many {thing}s does {name} have? Reply with just the number."
    ),
    substitutions=[
        {"label": "alice_apples",   "name": "Alice",   "thing": "apple",  "start": 3, "give": 2, "use": 1},
        {"label": "carlos_oranges", "name": "Carlos",  "thing": "orange", "start": 3, "give": 2, "use": 1},
        {"label": "shelf_books",    "name": "A shelf", "thing": "book",   "start": 3, "give": 2, "use": 1},
    ],
    expected_answer="4",
    extraction_method="numeric",
)
test = template.build(name="give_and_take")
extended = template.add_substitution({"label": "carol_pens", "name": "Carol", "thing": "pen", "start": 3, "give": 2, "use": 1})
```

Template uses `str.format_map()` syntax (single braces: `{name}`). Label resolution: `"label"` key in dict → `label_key` param → `"variant_{i}"`.

---

## Interpreting results

| Output | Meaning |
|---|---|
| `PASS  (behavioral)` | Model output passed. Claim: output X for input Y. |
| `FAIL — INCONSISTENT` | Different answers across framings. Primary signal of pattern matching. |
| `FAIL — CONSISTENT_WRONG` | Model reasons consistently but incorrectly. |
| `FAIL — UNSTABLE` | Different answers on identical prompts. Determinism issue. |
| `FAIL — FORMAT_BREACH` | Model did not follow format instruction — extraction returned None. |
| `ERROR` | Test raised an exception. Check model availability and prompt format. |

---

## Roadmap

**v0.2 — Next**
- `ContradictionDetectionTest` — model must flag contradictory premises rather than answer confidently
- `ReasoningChainValidityTest` — chain-of-thought steps must connect to the final answer
- `CounterfactualSensitivityTest` — critical fact changes must change the answer; irrelevant changes must not
- JSON reporter for CI/CD integration

**Phase 2**
- `AlignmentBehaviorTest` suite: refusal consistency, sycophancy, authority bias, value stability
- `CalibrationTest` with Expected Calibration Error (ECE) metric
- HTML reporter with trend charts
- Cross-model comparison campaigns (extends `ComparisonRunner`)

**Phase 3 — Mechanistic**
- HuggingFace backend with TransformerLens integration
- `LinearProbeTest`, `LogitLensTest`, `AttentionPatternTest` (OBSERVATIONAL claims)

**Phase 4 — Causal**
- `ActivationPatchingTest`, `AblationTest`, `CircuitIdentificationTest`, `FeatureSteeringTest` (CAUSAL claims)

---

## Key principles

- **Tests are honest about what they prove.** ClaimType makes the epistemological claim explicit in every result.
- **Consistency and correctness are separate failure modes.** A model can be consistently wrong, or correct but inconsistent.
- **Failure modes are more valuable than pass rates.** The goal is identifying how and when a model fails, not maximizing a score.
- **The framework tests itself.** `AdversarialValidator` verifies the test suite catches failures; `RepeatedPromptConsistencyTest` verifies the framework's own determinism controls work. Infrastructure failures must be distinguishable from model failures.
- **Coverage is a first-class metric.** `CoverageAnalyzer` makes the gap between "tests you have" and "failure modes that exist" visible and actionable.
