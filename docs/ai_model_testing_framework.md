# AI Model Testing Framework
## Development Specification & Technical Design

*A formal, pytest-style evaluation framework for AI model behavior*
*Beyond benchmarks — testing process validity, reasoning consistency, and alignment*

Version 0.1 — April 2026

---

## 1. Motivation & Problem Statement

### 1.1 The Benchmark Problem

Current AI evaluation relies primarily on benchmarks — curated datasets with known correct answers against which a model's accuracy is measured. While benchmarks provide a useful signal of broad capability, they have a fundamental limitation: they measure the destination, not the path.

A model can score 90% on a reasoning benchmark through any of the following mechanisms, and existing benchmarks cannot distinguish between them:

- Genuine reasoning — the model correctly applies logic to derive the answer
- Pattern matching — the model recognizes the problem structure from training data
- Statistical shortcuts — the model exploits regularities in how benchmarks are constructed
- Lucky guessing — the model guesses correctly on multiple-choice questions
- Reasoning incorrectly but arriving at the right answer by coincidence

> **Core insight:** Benchmarks test whether a model gets the right answer. This framework tests whether the model gets the right answer *for the right reasons*. The difference matters enormously in deployment — a model that pattern-matches will fail unpredictably when it encounters novel problems not represented in its training distribution.

### 1.2 What Is Missing

The primary gap in current AI evaluation is the absence of **process validity testing** — formal, reproducible tests that verify not just what a model outputs but whether its reasoning process is valid, consistent, and robust.

| Evaluation type | What it tests | What is missing |
|---|---|---|
| Benchmarks | Answer correctness on known problems | Process validity, robustness, generalization |
| Human eval | Subjective quality of responses | Reproducibility, formal metrics |
| Red-teaming | Safety failure modes | Systematic coverage, consistency |
| This framework | Process validity, reasoning consistency, alignment behavior | — |

### 1.3 Design Goals

- **Formal** — every test produces a structured verdict with evidence, not just a score
- **Honest** — tests declare what claim they make (behavioral / observational / causal) and do not overstate what they prove
- **Reproducible** — same test on same model produces same result; tests are deterministic
- **Composable** — tests combine into suites; suites combine into evaluation campaigns
- **Progressive** — start with behavioral tests (no weight access needed); extend to mechanistic tests when weights are available
- **Failure-seeking** — like pytest, the framework is designed to find failures, not confirm success

---

## 2. Architecture

### 2.1 Framework Structure

The framework is organized into four layers, each building on the one below. Tests at any layer can run independently without requiring the layers above or below.

| Layer | Responsibility |
|---|---|
| Model backend | Abstracts model access (Ollama, HuggingFace) behind a uniform interface. Tests do not know which backend they are talking to. |
| Test primitives | BaseTest, TestResult, ClaimType, Evidence — the core data types. Every test produces a TestResult with structured evidence. |
| Test implementations | Concrete tests organized by category: consistency, contradiction, reasoning chain, counterfactual, calibration, and (later) mechanistic. |
| Runner & reporter | TestSuite collects and executes tests. Reporters format results as terminal output, JSON, or HTML. |

### 2.2 File Structure

```
ai_test_framework/
├── core/
│   ├── model.py          # Model backend abstraction
│   ├── base_test.py      # TestResult, BaseTest, ClaimType, Evidence
│   ├── runner.py         # TestSuite, SuiteResult
│   └── extractors.py     # Parse model responses into comparable values
├── tests/
│   ├── consistency.py    # BUILT: surface framing, repeated prompt
│   ├── contradiction.py  # NEXT: contradiction detection, control group
│   ├── reasoning.py      # NEXT: chain-of-thought validity
│   ├── counterfactual.py # NEXT: critical fact sensitivity
│   ├── calibration.py    # PLANNED: confidence vs accuracy
│   └── mechanistic.py    # PLANNED: probing, activation patching
├── fixtures/
│   ├── logic_problems.py    # BUILT: 5 fixtures with multiple framings
│   ├── contradictions.py    # NEXT: contradictory premise problems
│   └── reasoning_chains.py  # NEXT: multi-step problems with known steps
├── reporters/
│   ├── terminal.py       # BUILT (inline in runner)
│   ├── json_reporter.py  # PLANNED
│   └── html_reporter.py  # PLANNED
run_tests.py              # Entry point — pytest equivalent
```

### 2.3 The ClaimType System

Every test in the framework declares its `ClaimType` — what epistemological claim it is making. This is the most important design decision in the framework because it forces honesty about what each test actually proves.

```
BEHAVIORAL    — tests input/output only.
                Proves: the model produced output X for input Y.

OBSERVATIONAL — tests what is encoded in representations.
                Proves: information is present (correlation, not causation).

CAUSAL        — tests whether a component causes a behavior via intervention.
                Proves: removing/changing X changes Y.
```

> **Critical:** A passing OBSERVATIONAL test does NOT prove the model uses that information causally. This distinction must be preserved in all test reporting.

### 2.4 Model Backend Abstraction

The `Model` class provides a unified interface across backends. Tests interact only with `Model` — never with Ollama or HuggingFace directly. This means the same test can run against:

- **Ollama (local)** — black-box API access, behavioral tests only, no weight access
- **HuggingFace (local)** — full weight access, enables observational and causal tests via TransformerLens
- **API (remote)** — OpenAI, Anthropic, etc., behavioral tests only, useful for comparing model families

```python
# All three work identically from the test's perspective
model = Model.ollama("llama3.2:1b")
model = Model.huggingface("meta-llama/Llama-3.2-1B")
model = Model.api("anthropic", "claude-haiku-4-5")

# Same test runs against any backend
suite.run(model)
```

---

## 3. Tests — Built (v0.1)

### 3.1 SurfaceFramingConsistencyTest

The first and most fundamental test. Presents the same logical problem with different surface framing — different names, contexts, and wording — and verifies the model produces the same answer across all variants.

> **Rationale:** If a model is reasoning, the answer should be invariant to surface framing. If a model is pattern-matching to training data, different framings will trigger different patterns and produce different answers. This is the primary reasoning detector.

| Property | Value |
|---|---|
| ClaimType | BEHAVIORAL |
| Pass condition | All variants produce the same extracted answer |
| Fail — inconsistent | Any variant produces a different answer — signals pattern matching |
| Fail — consistent wrong | All variants agree but are wrong — consistent but incorrect reasoning |
| Extraction methods | numeric, yn (yes/no), letter (A/B/C/D), exact |
| Minimum variants | 2 (3–4 recommended for statistical confidence) |
| Temperature | 0.0 — deterministic generation required |

```python
test = SurfaceFramingConsistencyTest(
    variants=[
        FramingVariant(label="original", prompt="Alice has 3 apples...", expected="4"),
        FramingVariant(label="renamed",  prompt="Xavier has 3 apples...", expected="4"),
        FramingVariant(label="money",    prompt="A store has $3...", expected="4"),
    ],
    extraction_method="numeric",
    expected_answer="4",
)
```

### 3.2 RepeatedPromptConsistencyTest

Runs the exact same prompt N times at temperature=0 and verifies all responses are identical. Tests whether the model backend is truly deterministic under the framework's control.

| Property | Value |
|---|---|
| ClaimType | BEHAVIORAL |
| Pass condition | All N runs produce identical extracted answers |
| Fail condition | Any variation across runs — signals non-determinism or seed issues |
| Default N | 3 runs (configurable) |
| Use case | Baseline determinism check — run before any other test campaign |

### 3.3 Built-in Fixtures

Five pre-built test fixtures are included, each presenting the same underlying logic across 3–4 surface framings:

| Fixture | Logic type | Variants | Extraction |
|---|---|---|---|
| apples_and_oranges | Arithmetic (give/receive) | 4 | numeric |
| simple_multiplication | Multiplication | 3 | numeric |
| transitive_logic | Transitivity (A > B > C) | 4 | yn |
| negation_logic | Syllogistic reasoning | 3 | yn |
| rate_problem | Rate × time = quantity | 3 | numeric |

---

## 4. Tests — Planned (v0.2 and beyond)

### 4.1 ContradictionDetectionTest

Tests whether the model identifies when input premises are logically contradictory, rather than answering as if the premises were coherent.

> **Why this matters:** A model that answers confidently based on contradictory premises is dangerous in production. Real-world data is often inconsistent. A well-behaved model should flag contradictions rather than hallucinate a coherent answer from incoherent input.

| Property | Value |
|---|---|
| ClaimType | BEHAVIORAL |
| Pass condition | Model refuses, flags, or expresses uncertainty about contradictory premises |
| Fail condition | Model answers confidently despite contradictory premises |
| Control group | ConsistentPremiseTest — same structure, no contradiction, model should answer |
| Detection method | Keyword extraction: "contradict", "inconsistent", "cannot", "impossible", "conflict" |

Planned fixture categories:

- **Temporal:** "The meeting is on Monday" + "The meeting is on Wednesday" → "What day is the meeting?"
- **Quantitative:** "There are 5 items" + "There are 8 items" → "How many items are there?"
- **Logical:** "All A are B" + "No A are B" → "Are A and B related?"
- **Factual:** Explicitly contradictory facts about a fictional entity

### 4.2 ReasoningChainValidityTest

Prompts the model to show its work via chain-of-thought, then verifies that the reasoning steps are logically connected to the final answer — not post-hoc rationalization.

> **The post-hoc rationalization problem:** Models can produce plausible-sounding reasoning chains that are disconnected from how they actually computed the answer. A model that arrives at 42 by correct reasoning is very different from one that arrives at 42 and then constructs a plausible-sounding argument.

| Property | Value |
|---|---|
| ClaimType | BEHAVIORAL |
| Pass condition | Final answer matches the conclusion the reasoning chain implies |
| Fail — misaligned | Reasoning chain leads to answer X but model outputs answer Y |
| Fail — non-sequitur | Reasoning chain contains a logical gap or contradiction |
| Method | Extract intermediate conclusions from CoT; verify they chain correctly |
| Limitation | Requires problems with known valid reasoning paths |

Planned sub-tests:

- **StepConsistencyTest** — each step in the chain follows from the previous
- **ConclusionAlignmentTest** — the stated conclusion matches what the chain implies
- **RedundantStepTest** — the chain does not contain contradictory intermediate steps

### 4.3 CounterfactualSensitivityTest

Changes a single critical fact in a problem and verifies the answer changes appropriately. One of the most powerful tests for distinguishing genuine reasoning from sophisticated pattern matching.

> **Why counterfactual testing is powerful:** A model that pattern-matches will often give the same answer even when a critical fact changes, because the overall structure looks the same. A model that reasons will produce a different answer when a logically relevant fact changes, and the same answer when only irrelevant surface details change.

This test has two complementary sub-tests:

- **CriticalFactChange** — change a fact that *should* change the answer (model must respond)
- **IrrelevantFactChange** — change a fact that *should not* change the answer (model must not respond)

| Property | Value |
|---|---|
| ClaimType | BEHAVIORAL |
| Critical fact pass | Answer changes when a logically relevant fact changes |
| Critical fact fail | Answer stays the same — model ignoring the changed fact |
| Irrelevant fact pass | Answer stays the same when only surface details change |
| Irrelevant fact fail | Answer changes — model over-sensitive to irrelevant variation |

### 4.4 CalibrationTest

Tests whether the model's expressed confidence correlates with its actual accuracy. A well-calibrated model should be right ~80% of the time when it expresses 80% confidence.

| Property | Value |
|---|---|
| ClaimType | BEHAVIORAL |
| Method | Ask model to state confidence alongside answer across N problems |
| Metric | Expected Calibration Error (ECE) — deviation from the diagonal |
| Pass condition | ECE below threshold (e.g. 0.1 for 10% average miscalibration) |
| Limitation | Requires a large problem set with known answers |

### 4.5 AlignmentBehaviorTest

Tests whether the model's behavior is consistent with stated values — not just in obvious cases but across subtle variants.

- **RefusalConsistencyTest** — same harmful request in different framings — model should refuse all
- **ValueStabilityTest** — model maintains stated position when user pushes back without new arguments
- **SycophancyTest** — model does not change correct answers when user expresses disagreement
- **AuthorityBiasTest** — model does not change answers based on claimed authority of the asker

---

## 5. Mechanistic Interpretability Layer (Future)

### 5.1 Overview

The mechanistic layer extends the framework from behavioral black-box testing to internals-based testing. This requires open-weight models and access to activations. The primary tool is TransformerLens, with SAELens for feature decomposition.

> **Key distinction to preserve:** Mechanistic tests produce OBSERVATIONAL or CAUSAL claims — not behavioral ones. OBSERVATIONAL: "This information is encoded in layer 12" (correlation). CAUSAL: "Ablating this attention head reduces accuracy by 40%" (intervention). These claims are fundamentally different and must not be conflated in reports.

### 5.2 Planned Mechanistic Tests

| Test | ClaimType | Description |
|---|---|---|
| LinearProbeTest | OBSERVATIONAL | Train linear classifier on layer activations to detect concept encoding |
| LogitLensTest | OBSERVATIONAL | Track how predictions evolve layer-by-layer |
| AttentionPatternTest | OBSERVATIONAL | Identify which tokens each attention head attends to |
| ActivationPatchingTest | CAUSAL | Replace activations from one run to another; measure output change |
| AblationTest | CAUSAL | Zero out components; measure which are necessary for behavior |
| CircuitIdentificationTest | CAUSAL | Map which attention heads + MLPs implement a specific behavior |
| FeatureSteeringTest | CAUSAL | Inject SAE feature directions; verify behavioral effect |

### 5.3 Tool Stack

| Tool | Role | Status |
|---|---|---|
| TransformerLens | Primary — activation hooks, caching, patching across 50+ architectures | Established (3,300+ stars) |
| SAELens | Feature decomposition — decompose polysemantic neurons into interpretable features | Active development |
| NNsight | Alternative for models TransformerLens does not support | Emerging (ICLR 2025) |
| Captum / SHAP | Attribution baselines for black-box models | Established |

---

## 6. Metrics & Reporting

### 6.1 Test-Level Metrics

| Metric | Definition |
|---|---|
| Verdict | PASS / FAIL / ERROR / SKIP — the primary outcome |
| ClaimType | behavioral / observational / causal — what the test proves |
| Evidence | Structured key-value pairs showing the raw data behind the verdict |
| Duration (ms) | Wall-clock time — useful for detecting latency regressions |
| Error | Exception message if the test errored rather than passed or failed |

### 6.2 Suite-Level Metrics

| Metric | Definition |
|---|---|
| Pass rate | Passed / Total — primary suite health metric |
| Fail rate by ClaimType | Separate pass rates for behavioral / observational / causal |
| Consistency score | Fraction of framing variants that agree — more granular than pass/fail |
| Calibration ECE | Expected Calibration Error across calibration tests |
| Total duration | Wall-clock time for the full suite |

### 6.3 Planned Report Formats

- **Terminal (built)** — structured output with pass/fail icons, evidence, and summary table
- **JSON (v0.2)** — machine-readable results for CI/CD integration and longitudinal tracking
- **HTML (v0.3)** — visual report with per-test evidence, response text, and trend charts
- **Comparative (v0.4)** — side-by-side results across model versions or model families

---

## 7. Development Roadmap

### Phase 1 — Behavioral Foundation (Current)

Target: Ollama local backend, behavioral tests, no weight access required.

| Item | Status |
|---|---|
| Model backend abstraction (Ollama) | Done |
| TestResult, ClaimType, Evidence primitives | Done |
| TestSuite runner with terminal output | Done |
| Answer extractors (numeric, yes/no, letter, exact) | Done |
| SurfaceFramingConsistencyTest | Done |
| RepeatedPromptConsistencyTest | Done |
| 5 built-in fixtures with multiple framings | Done |
| ContradictionDetectionTest + control group | Next |
| ReasoningChainValidityTest | Next |
| CounterfactualSensitivityTest | Next |
| JSON reporter | Next |

### Phase 2 — Alignment & Calibration

Target: Alignment behavior tests, calibration metrics, expanded fixture library.

- AlignmentBehaviorTest suite (refusal consistency, sycophancy, authority bias)
- CalibrationTest with ECE metric
- Expanded fixture library (20+ problems across domains)
- HTML reporter
- Cross-model comparison runner

### Phase 3 — Mechanistic Layer

Target: Open-weight model support, TransformerLens integration, observational tests.

- HuggingFace backend with TransformerLens hook integration
- LinearProbeTest — concept encoding detection
- LogitLensTest — layer-by-layer prediction tracking
- AttentionPatternTest — head role identification
- SAELens integration for feature decomposition

### Phase 4 — Causal Testing

Target: Activation patching, ablation, circuit analysis — causal claims.

- ActivationPatchingTest — causal tracing
- AblationTest — necessity testing
- CircuitIdentificationTest — behavior localization
- FeatureSteeringTest — SAE-based behavioral intervention
- Automated circuit discovery pipeline

---

## 8. Usage Reference

### 8.1 Running Tests

```bash
# Run all tests against default model
python run_tests.py

# Run against a specific model
python run_tests.py --model llama3.2:3b

# Run a single fixture
python run_tests.py --fixture transitive_logic

# List available fixtures
python run_tests.py --list

# Exit code: 0 = all pass, 1 = any failure (mirrors pytest)
```

### 8.2 Writing a Custom Test

```python
from ai_test_framework.tests import SurfaceFramingConsistencyTest, FramingVariant
from ai_test_framework.core import Model, TestSuite

variants = [
    FramingVariant(
        label="version_1",
        prompt=(
            "If 5 machines take 5 minutes to make 5 widgets, "
            "how long for 100 machines to make 100 widgets? "
            "Reply with just the number."
        ),
        expected="5",
    ),
    FramingVariant(
        label="version_2",
        prompt=(
            "If 5 workers take 5 hours to dig 5 holes, "
            "how long for 100 workers to dig 100 holes? "
            "Reply with just the number."
        ),
        expected="5",
    ),
]

model = Model.ollama("llama3.2:1b")
suite = TestSuite("my_suite")
suite.add(SurfaceFramingConsistencyTest(
    variants=variants,
    extraction_method="numeric",
    expected_answer="5",
))
suite.run(model)
```

### 8.3 Interpreting Results

| Output | Meaning |
|---|---|
| `✓ PASS  (behavioral)` | Model output passed this test. Claim: output X for input Y. |
| `✗ FAIL — INCONSISTENT` | Different answers across framings. Primary signal of pattern matching. |
| `✗ FAIL — CONSISTENT but WRONG` | Model reasons consistently but incorrectly. Separate failure mode. |
| `✗ FAIL — UNSTABLE` | Different answers on identical prompts. Determinism issue. |
| `! ERROR` | Test raised an exception. Check model availability and prompt format. |

---

## 9. Key Principles

These principles govern every design decision in the framework.

**Tests are honest about what they prove.** A behavioral test proves nothing about internal mechanisms. An observational test proves encoding, not causal use. ClaimType makes this explicit in every result.

**Consistency and correctness are separate.** A model can be consistently wrong. A model can be correct on individual prompts but inconsistent across framings. These are different failure modes requiring different fixes.

**Failure modes are more valuable than pass rates.** The goal is not a high score — it is identifying the specific ways a model fails and under what conditions.

**The framework tests the framework too.** RepeatedPromptConsistencyTest exists partly to verify the framework's own determinism controls are working. Infrastructure failures must be distinguishable from model failures.

**Black-box and white-box tests are complementary.** Behavioral tests tell you what the model does. Mechanistic tests tell you why. Both are needed for a complete picture. Neither alone is sufficient.