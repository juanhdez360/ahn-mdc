# Hypotheses

Preregistered before the full run. Source: *When Memory Fails: Characterizing Information
Degradation and Confidence Calibration in AHN* (team research doc).

Central question: **how does information degrade in AHN under compression, and does the
model know when its memory has become unreliable?**

---

## H1 — Non-uniform degradation

AHN does not forget uniformly. Retrieval accuracy declines at different rates for different
information types, with some categories (precise numerical, entity-attribute) degrading
faster than others as compression accumulates.

| | |
| --- | --- |
| Measured by | `ahn.h1_degradation` |
| Test | per-fact-type degradation slopes differ; interaction of fact type × compression pressure |
| Supported if | slope confidence intervals separate across fact types after Holm correction |
| Refuted if | one shared slope fits all types within their intervals |

**Watch out.** The pilot shows extreme spread — `temporal` at 90–100% under high pressure
versus `contradictory` at 0% everywhere. Before reading that as H1 support, verify the
answer matcher scores each type correctly. A scoring bug looks exactly like a large H1 effect.

## H2 — Threshold-like collapse

Degradation is nonlinear in compression pressure: recall stays relatively stable while the
recurrent state has spare capacity, then drops sharply once the state saturates, rather than
declining smoothly throughout.

| | |
| --- | --- |
| Measured by | `ahn.h2_threshold` |
| Test | (a) accuracy drop across the published threshold T; (b) piecewise-at-T fit versus single smooth fit |
| Supported if | the drop at T is large with an interval excluding zero, **and** the piecewise fit beats the smooth one |
| Refuted if | the smooth fit is as good — degradation is gradual, not threshold-like |

**T comes from a published paper, never from our data.** Working value: 50 facts after the
target. The decision record in `h2_threshold.md` is unresolved, so
`config.compression_threshold` raises and the H2 numbers stay blank. Estimating the
changepoint from our own curves would be self-defining the threshold, which the mentor
ruled out — it would force an extra sensitivity-analysis table we have no page budget for.

## H3 — Miscalibration under compression

Expressed confidence does not fully track declining accuracy. Confidence falls more slowly
than accuracy, producing a widening confidence–accuracy gap and, in the strongest case,
confidently incorrect answers on facts the compressed memory has lost.

| | |
| --- | --- |
| Measured by | `ahn.h3_calibration` |
| Test | ECE, Brier and confidently-wrong rate as functions of compression pressure |
| Supported if | the gap (mean confidence − accuracy) grows with pressure and stays positive |
| Refuted if | confidence falls with accuracy — a positive finding either way, since it would show behavioural awareness |

ECE follows Guo et al. 2017 (ICML), equal-width bins. Brier follows Brier 1950. Both
formulas are in the research doc; `ahn.metrics` implements them and cites them inline.
The 500-row pilot is sufficient for interpreting ECE — no need to scale before reading it.

---

## Shared design commitments

These apply to all three hypotheses and are enforced in code, not by discipline.

**Matched trajectories.** For each target fact the wording and the query stay constant while
compression pressure varies. Effects of pressure are estimated within-item, so fact
difficulty cancels out.

**State reset.** Every trajectory is an independent forward pass from a fresh state. Repeated
probing must not alter what a later probe sees.

**Compression pressure is `tokens_after_target`**, in model tokens, and is a *proxy* for
compression pressure rather than a direct measurement of recurrent-state occupancy. Say so
in the paper.

**The exact/recurrent boundary is mechanical.** Below the sliding window the target is still
in the lossless KV cache; at or above it, the target survives only in the compressed state.
The window is read from the checkpoint, not chosen.

**Accuracy band.** Exact-memory accuracy must land in 70–80% (85% acceptable). At 90% or
above the task is already solved by existing models and is not publishable.
