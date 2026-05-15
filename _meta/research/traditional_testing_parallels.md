# Traditional Software Testing → AI Model Testing: Parallels
*Researched: 2026-05-15*

---

## The Core Translation Problem

Traditional software tests make deterministic claims: given input X, function F produces output Y. The claim is exact and binary. A passing test is a proof by demonstration.

AI model tests cannot make the same claim. The same prompt at temperature > 0 produces different tokens each run. Even at temperature=0, different hardware, batching, or model versions silently change outputs. The parallel is not "unit test → prompt test" — it's more nuanced.

---

## Parallel Map

| Traditional | What it tests | AI Parallel | Status |
|---|---|---|---|
| Unit test | Single function in isolation | Atomic behavioral probe — one prompt, one expected output | BUILT (fixtures + extractors) |
| Regression test | Bug not reintroduced | Re-run suite after model update, compare results | NOT BUILT |
| Integration test | Components work together | Reasoning chain test — step N feeds step N+1 | PLANNED (ReasoningChainValidityTest) |
| E2E test | Full user workflow | Multi-turn / agentic task completion test | NOT PLANNED |
| Property-based test | Invariant holds over many inputs | Surface framing consistency — "answer is invariant to surface" | BUILT (SurfaceFramingConsistencyTest) |
| Mutation testing | Tests catch code changes | Adversarial fixture validation — tests FAIL when they should | NOT BUILT |
| Contract testing | API boundaries hold | Output format contract — model respects format instructions | NOT PLANNED |
| Smoke test | System is alive | RepeatedPromptConsistencyTest at temp=0 | BUILT |
| Coverage metric | % of code exercised | Behavioral coverage — % of failure modes tested | NOT BUILT |
| Formal verification | Mathematical correctness | Known-answer arithmetic/logic problems (fixture approach) | BUILT (partial — for deterministic domains) |
| Fuzz testing | No crashes on random input | Adversarial prompt robustness | NOT PLANNED |
| Statistical assertion | P(pass) ≥ threshold over N runs | Statistical consistency at temp > 0 | NOT BUILT |

---

## The Non-Determinism Problem (Central Challenge)

Traditional solutions and their AI analogues:

| Traditional technique | AI equivalent | Trade-off |
|---|---|---|
| Fixed seed (RNG) | temperature=0 + seed parameter | Only tests one inference path; real-world behavior at temp>0 is untested |
| Mocking | Model stub / canned response | Tests framework, not model |
| Retry logic | Run N times, pass if any succeeds | Hides flakiness; can mask real failure modes |
| Property-based testing | Test invariants (not exact tokens) — "answer must contain a number" | Tests weaker claim; invariant may not catch important failures |
| Statistical assertion | Run N times, assert pass_rate ≥ threshold | Not binary; needs calibrated threshold; slower |

**Our current position:** We handle non-determinism by forcing temperature=0 and a fixed seed. This is correct for deterministic behavioral tests but leaves a gap: no framework for testing behavior at realistic temperatures, where statistical confidence intervals are needed instead of binary pass/fail.

---

## Key Principles — How They Translate

| Traditional principle | AI translation |
|---|---|
| Isolation: test one thing | Each probe tests one reasoning claim; no shared conversation state |
| Repeatability: same input → same output | temperature=0 + seed enforced; documented in ClaimType |
| Independence: test order doesn't matter | Each test is stateless; no fixture shares model state |
| Fast feedback | Behavioral tests should complete in <2 min against local Ollama |
| Test-as-specification | A test fixture IS the specification of what reasoning the model must demonstrate |

**Entanglement risk in AI:** Tests become coupled when they share conversation history (multi-turn), share model state (fine-tuning context), or when fixture expected answers are interdependent. Our current stateless probe design avoids this. The risk returns when we add agentic / multi-turn tests.

---

## Coverage: What's the AI Equivalent?

Code coverage measures "what % of lines did tests execute." For AI testing, coverage means:

- **Reasoning coverage:** What % of known reasoning failure modes are represented in the test suite?
- **Domain coverage:** What % of problem domains (arithmetic, logic, causal, temporal, spatial) are tested?
- **Framing coverage:** How many distinct surface framings per fixture? (More = stronger invariance claim)

Neither Silico nor Langfuse defines this. It's an open design question. A simple v1 metric: `(failure modes tested) / (failure modes catalogued)`.

---

## Proposed Features (Ordered by Impact / Effort)

These emerge from the gaps in the parallel map above. Presented for approval one at a time.

1. **Statistical consistency test** — non-determinism handled via pass_rate ≥ threshold at temp > 0
2. **Model regression runner** — same suite across two model versions, diff the results
3. **Adversarial fixture validation** — deliberately broken variants that must fail (mutation test for AI)
4. **Behavioral coverage metric** — score how many failure modes the current suite tests
5. **Output format contract test** — verify model respects format instructions across framings
6. **Auto-variant generation** — programmatically generate surface framings from a template
7. **Multi-turn entanglement test** — detect when model state leaks across turns
