# Open decisions and pending work

Everything that must be settled before the full run. Grouped by whether it blocks other
people. Owners follow the table split: Saadat (T1, core pipeline), Sumiya (T2, calibration),
Youssef (T3, robustness factors), Juan (T4, architectures & H2).

Status: `BLOCKER` · `OPEN` · `DONE`

---

## P0 — Blocks everything

### 1. Verify the target actually leaves the sliding window · `BLOCKER` · Saadat + Juan

The research doc commits to *"the inference sliding window deliberately shortened so that
target facts cross into compressed memory after a short, controlled offset"*. It is not clear
the pilot did this, and the evidence suggests it did not:

- AHN trains Qwen2.5-3B with `--sliding_window 256`, but the merged checkpoint loads through
  a stock `Qwen2ForCausalLM`, which carries Qwen2.5's own (much larger) window.
- The longest pilot prompt is 3,303 tokens. If the effective window is 4k or 32k, **every one
  of the 500 rows was answered from exact attention** and nothing was ever compressed.
- The pilot's own numbers fit that reading: 100% accuracy in-window, and a non-monotonic
  curve afterwards (14% at 25 facts, then 38% at 50, 39% at 100, 32% at 200). Genuine memory
  degradation does not recover as pressure increases; distractor interference does behave
  erratically like this.

**Action.** Print `model.config.sliding_window` and `use_sliding_window` on the merged
checkpoint, confirm the AHN path is active, and shorten the window explicitly at inference.
`ahn.models.load` takes `sliding_window=` and `ahn.report.gate_report` fails the run if the
pressure grid never clears it.

Until this is settled, **no pilot number means anything** — the 500 rows may be measuring
prompt length, not memory.

### 2. Results schema sign-off · `OPEN` · everyone

`src/ahn/schema.py` defines the per-trial contract all three hypotheses read. It was drafted
by one person. Half an hour of review now is cheaper than discovering mid-A40-run that
confidence is on a different scale than H3 expects.

---

## P1 — Blocks a specific hypothesis

| # | Decision | H | Owner | Status |
| --- | --- | --- | --- | --- |
| 3 | **H2 threshold T** from a published paper. Working value 50 facts. Never self-defined — that would force a sensitivity table we have no page budget for. See `h2_threshold.md`. | H2 | Juan | `BLOCKER` |
| 4 | **ECE formula** from a top-venue paper. Guo et al. 2017 (ICML) is already cited in the research doc; confirm bin count and binning strategy against the PDF and freeze them. | H3 | Sumiya | `OPEN` |
| 5 | **CWR threshold** — what confidence counts as *confidently* wrong. Same threshold everywhere. The pilot used 0.5 with no justification. | H3 | Sumiya | `OPEN` |
| 6 | **Confidence definition.** Pilot used full-sequence token probability, which penalises long answers: pilot confidences span 1.8e-06 to 0.99. Decide between sequence probability and length-normalised, apply one everywhere. | H3 | Sumiya | `OPEN` |
| 7 | **Answer matcher per fact type.** Pilot scores `contradictory` at 0% under every condition and `temporal` at 90–100%. Verify this is real degradation and not a scoring bug before it is reported as the H1 effect. | H1 | Saadat | `BLOCKER` |
| 8 | **Fact-type taxonomy frozen** — numerical, temporal, entity-attribute, multi-hop, contradictory (research doc, Table 2). | H1 | Saadat | `DONE` → `config/facts.yaml` |
| 9 | **Random seeds.** How many, fixed across models and conditions. The pilot has 100 distinct `seed` values but one item each, so there is no replication and every clustered interval comes back empty. | all | Youssef | `BLOCKER` |
| 10 | **Distractor density** defined quantitatively, not as low/high labels. | all | Youssef | `OPEN` |
| 11 | **`importance` is currently a no-op.** The pilot tags facts high/low but never changes the text, so the variable cannot explain anything. Either manipulate it in the fact wording or drop it. | H1 | Youssef | `OPEN` |
| 12 | **Architecture configs frozen** — Mamba2 / DeltaNet / GatedDeltaNet matched so architecture is the only difference. | H2 | Juan | `DONE` → `config/experiment.yaml` |
| 13 | **Length-matched controls.** `tokens_after_target` must not be confounded with total prompt length or with the target landing at the start. The pilot puts the target first in every sequence. | all | Saadat | `OPEN` |

---

## P2 — Needed before submission, not before running

| # | Item | Owner | Status |
| --- | --- | --- | --- |
| 14 | Full evaluation set size; all tables must use the same underlying set. | Saadat | `OPEN` |
| 15 | Abstention detection — "I don't know" is not the same as retrieving the wrong thing. Pilot does not measure it. | Sumiya | `OPEN` |
| 16 | Second-stage validation on LongBench / LongBench v2, if the synthetic effect holds. | team | `OPEN` |

---

## Fixed constraints

These came from the mentor and are not up for renegotiation.

**Accuracy targets.** 70–80% defensible, 85% acceptable upper bound, 90%+ signals the task
is already solved by existing models, 95–99% is indefensible at review. Applies to the
exact-memory control condition. Enforced by `ahn.report.gate_report`.

**Venue.** NAACL industry track: 6 pages, October deadline, 10 exhibits (6 tables + 4
graphs). Gautam adds his name only for the industry track, not the main track. The venue
locks once Tables 1–4 are done; Tables 5–10 get framed to fit afterwards. Framing adapts per
venue; numbers, hyperparameters and methodology stay fixed.

**Pilot compute.** 100–150 samples or 10% of data, on Colab GPU. A40 access is capped at two
days, which is not enough to validate a pipeline. Pilot runs verify plumbing and are never
cited as results. The 500-row pilot is sufficient for interpreting ECE.

**Published sources only** for the H2 threshold and the ECE formula — top-venue, and read
from the PDF rather than a summary.
