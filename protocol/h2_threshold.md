# H2 — Compression threshold (published anchor)

**Owner:** Juan · **Hypothesis:** H2 · **Status:** `PENDING_VERIFICATION`

Mentor constraint: the threshold must come from a **published paper**, not from us.
Self-defining it would force an extra sensitivity-analysis table, and we only have 6 pages
(NAACL industry track).

Working value: **50 facts after the target**. Source not yet recorded — fill section 4
and flip the status before any H2 number is reported. `ahn.config.compression_threshold`
raises until then.

---

## 1. What the threshold is

H2 claims that retrieval from AHN's compressed memory collapses past a compression
threshold, and that where it sits depends on the recurrent architecture.

The threshold **T** is a point on the compression-pressure axis: the amount of material
the AHN module has to absorb after the target fact before retrieval is expected to fail.
It is *not* an accuracy floor. Accuracy is what we measure on either side of T.

`ahn.h2_threshold` therefore reports, per architecture:

| | |
| --- | --- |
| `acc_below_T` | accuracy at pressures below the published threshold |
| `acc_above_T` | accuracy at pressures at or above it |
| `drop_at_T` | the paired difference, with a bootstrap CI |
| `shape` | whether a fit allowed to break at T beats a single smooth fit |

The contrast is pre-registered at T rather than fitted, so nothing here is tuned to the
data. That is the whole point of taking T from the literature — and it is why the shape
test fixes the break at the published T rather than searching for the best changepoint,
which would be self-defining the threshold through the back door.

## 2. Unit problem: facts vs tokens

The threshold is stated in **facts**, but the AHN module compresses **tokens**. A fact is
a variable number of tokens, so a fact-denominated threshold is not length-controlled —
which is exactly the confound the token axis exists to avoid.

Resolution: T is declared in facts, converted once to tokens with the shared Qwen2.5
tokenizer, and every measurement runs on the token axis.

- `threshold.tokens_per_fact.measured` in `config/experiment.yaml` holds the conversion,
  measured over the pilot's own contexts.
- The pilot suggests roughly 16 tokens per fact: `facts_after_target = 50` came out at
  `token_count = 807`. Treat that as an order-of-magnitude check, not the conversion.
- Report the conversion and its spread in the paper. A reviewer will ask why the x-axis
  changed units between the cited paper and our figure.

## 3. Sanity check against the sliding window

AHN trains Qwen2.5-3B with `--sliding_window 256` (`examples/scripts/train_qwen2.5_3b_ahn_gdn.sh`).
Nothing is compressed until a token leaves that window, so:

- T must be **comfortably above one window**, or the target is still sitting in the lossless
  KV cache and there is no compression to test. All arms, including the no-AHN baseline,
  would sit at ceiling and every curve would be flat.
- At ~16 tokens/fact, 50 facts ≈ 800 tokens ≈ **3.1 windows**. That clears the window and
  lands in genuine recurrent-memory territory.

**Verify the window on the merged checkpoint before running.** `ahn.models.effective_window`
reads it from the model config; the training script value is not automatically what the
released checkpoint reports. If it comes back much larger than 256, T falls back inside the
window and the threshold has to be re-derived.

## 4. Extraction record — fill before locking

Copy the sentence from the paper. Do not paraphrase.

```

```

## 5. Verification checklist

- [ ] Read the PDF, not the abstract or a blog summary.
- [ ] Venue is top-tier (ACL/NAACL/EMNLP/NeurIPS/ICML/ICLR/COLM/TACL/AAAI). Workshop and
      arXiv-only papers do not satisfy the mentor's constraint.
- [ ] The number is stated in the paper, not inferred by us from one of their figures.
- [ ] The unit is recorded, and the fact→token conversion is measured, not assumed.
- [ ] T is above one sliding window on the merged checkpoint.
- [ ] The same T is applied to **all four** arms (Mamba2, DeltaNet, GatedDeltaNet,
      Transformer baseline) — T is a property of the design, never tuned per architecture.
- [ ] The pressure grid brackets T on both sides with at least two points each.
- [ ] BibTeX entry added to the paper repo.

## 6. Interaction with the accuracy targets

Independent constraint from the mentor, applied as an acceptance gate:

| Band | Meaning |
| --- | --- |
| 70–80% | defensible range |
| ≤ 85% | acceptable upper bound |
| ≥ 90% | red flag — the task is already solved by existing models, not publishable |

This band applies to the **exact-memory (control) condition**, where the target is still
inside the window. The recurrent-memory condition is *expected* to fall below it — that
drop is the result. If exact-memory accuracy lands ≥90%, the task is too easy and the
fact/distractor design must be hardened before the full run.
`ahn.report.gate_report` enforces this.

## 7. Decision record

Flip to `LOCKED` only when every checklist box is ticked. Then copy `facts` into
`threshold.facts` in `config/experiment.yaml` and set its status to match.

```yaml
status: PENDING_VERIFICATION
facts: 50                       # working value, unverified
tokens: null                    # measured conversion, filled at runtime
source: null                    # author, short title, venue, year
source_id: null                 # DOI / arXiv id
quote: null
unit_in_paper: null
locked_by: null
locked_on: null
```
