# When Memory Fails

 Characterizing Information Degradation and Confidence Calibration in Adaptive Hybrid Neural Memory (AHN)

**How does information degrade in AHN under compression, and does the model know when
its memory has become unreliable?**

AHN keeps recent tokens exactly in a sliding window and folds everything older into a
fixed-size recurrent state. Existing evaluations report aggregate accuracy, which says
whether the answer came out right but not *what* was lost, *how* it was lost, or whether
the model can tell. Three hypotheses, one shared pipeline:

| | Claim | Module | Notebook |
| --- | --- | --- | --- |
| **H1** | Degradation is non-uniform across information types | `h1_degradation` | `2_h1_degradation` |
| **H2** | Collapse is threshold-like, not smooth | `h2_threshold` | `3_h2_threshold` |
| **H3** | Confidence does not track the decline | `h3_calibration` | `4_h3_calibration` |

Full statements and test criteria: [`protocol/hypotheses.md`](protocol/hypotheses.md).
Everything still undecided, with owners: [`protocol/open_decisions.md`](protocol/open_decisions.md).

---

## Start here

```bash
uv sync
uv run python -m ahnexp._smoke      # analysis pipeline end to end, no GPU
```

The smoke test fabricates trials with a known degradation shape and runs them through
H1, H2, H3 and the acceptance gates. Run it after touching anything in `src/`.

### GPU run (local Linux or Colab)

Notebook `1_run_experiment` needs CUDA + the [ByteDance AHN](https://github.com/ByteDance-Seed/AHN) repo for weight merging.

**Local (you have GPU):**

```bash
uv sync
git clone --depth 1 https://github.com/ByteDance-Seed/AHN.git vendor/AHN
# then open notebooks/1_run_experiment.ipynb and uncomment the shared pip + local install cell
uv run jupyter notebook notebooks/1_run_experiment.ipynb
```

`config.ahn_repo()` resolves, in order: explicit path → `$AHN_REPO` → Colab
`/content/AHN` → `vendor/AHN`.

**Colab:** clone to `/content/AHN`, use the Colab install block in the same notebook,
restart the session, continue.

Then, in order:

| Notebook | Does | Needs |
| --- | --- | --- |
| `1_run_experiment` | Builds items, runs the grid, writes `outputs/results*.parquet` | CUDA GPU + AHN repo |
| `2_h1_degradation` | Per-fact-type curves and slopes | the parquet |
| `3_h2_threshold` | Drop at T, shape test, architecture comparison | the parquet |
| `4_h3_calibration` | ECE, Brier, CWR, reliability diagrams | the parquet |

Notebooks 2–4 also run without a GPU: set `SOURCE = "synthetic"` to review the analysis,
or `SOURCE = "pilot_csv"` to run against the real pilot data.

`0_pilot_reference` is the original Colab pilot, kept unmodified as provenance for
`data/pilot_raw_results.csv`. Nothing depends on it.

## Layout

```
protocol/     hypotheses, open decisions, the H2 threshold record
config/       experiment.yaml + facts.yaml — every experimental number
src/ahnexp/      the pipeline
notebooks/    one per stage; thin wrappers over src/
data/         the pilot dataset
outputs/      parquet, tables/, figures/ — regenerated, not versioned
```

Inside `src/ahnexp/`:

```
config.py           repo paths, YAML loading, the H2 citation gate
schema.py           per-trial contract shared by all three hypotheses
dataset.py          fact generation, matched trajectories, collision control
models.py           matched arm loading, sliding-window override
evaluate.py         generation, scoring, confidence, the grid runner
stats.py            cluster bootstrap, paired differences, design checks
metrics.py          accuracy, RFR, ECE, Brier, CWR — with their citations
h1_degradation.py   H1
h2_threshold.py     H2 + architecture comparison
h3_calibration.py   H3
report.py           acceptance gates, figures, Markdown tables
```

## Three things wired into the code, not left to discipline

**The accuracy band is a gate.** `report.gate_report` flags exact-memory accuracy outside
70–80%, warns above 85%, and raises a red flag at 90% — at which point existing models
already solve the task and the fact/distractor design has to be hardened before spending
A40 time.

**Compression has to actually happen.** A merged AHN checkpoint loads through a stock
`Qwen2ForCausalLM` carrying Qwen2.5's own, much larger sliding window. If it is not
forced down, the target never leaves exact attention and the run measures prompt length
instead of memory. `models.load` forces it; the `window_is_exceeded` gate fails the run
if no trial cleared it.

## Status

The analysis path is complete and tested. The run path is written but unverified against
a live checkpoint, because the window question above is unresolved.

The 500-row pilot in `data/` exercises the pipeline and trips three gates — it is not
evidence, and [`data/README.md`](data/README.md) explains why in detail.

Two blockers gate everything else: verifying the sliding window (#1) and finding a
published source for T (#3). Both are in
[`protocol/open_decisions.md`](protocol/open_decisions.md).

## Venue

NAACL industry track: 6 pages, October deadline, 10 exhibits (6 tables + 4 graphs).
Framing adapts per venue; numbers, hyperparameters and methodology stay fixed.
