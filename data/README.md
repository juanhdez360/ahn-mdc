# Data

## `pilot_raw_results.csv`

500 trials from the first pilot, produced by `notebooks/0_pilot_reference.ipynb` on
AHN-GatedDeltaNet + Qwen2.5-3B-Instruct. Load it through
`ahn.schema.from_pilot_csv`, which maps the pilot's columns onto the current schema
and converts `facts_after_target` to `tokens_after_target` by differencing each item's
context length against its own zero-pressure probe (≈15.4 tokens per fact).

**Do not cite anything from this file.** It is useful for exercising the analysis code
and for showing the acceptance gates catching real problems. It is not evidence. Four
reasons, in the order they matter:

1. **It is not clear anything was ever compressed.** The longest prompt is 3,303 tokens.
   If the merged checkpoint kept Qwen2.5's own sliding window rather than AHN's 256,
   every row was answered from exact attention. `from_pilot_csv` makes you pass the
   window explicitly precisely because we do not know it. Blocker #1.
2. **Exact-memory accuracy is 100%**, which trips the `RED_FLAG` gate. With one fact and
   no distractors the task is trivial, so the control condition carries no information.
3. **Accuracy is non-monotonic in pressure** — it drops, then partially recovers. Memory
   degradation does not recover; distractor interference does behave this way.
4. **There is no replication.** Each of the 100 items has its own unique `seed`, so
   clustered bootstraps have a single observation per cluster and every interval comes
   back empty. Fixing this is blocker #9.

The confidence column is also degenerate: sequence probability spans 1.8e-06 to 0.99,
with 58% of trials in the lowest ECE bin. `h3_calibration.confidence_health` flags it.

## Run outputs

`outputs/results{suffix}.parquet` is written by `notebooks/1_run_experiment.ipynb` and
is the only input to notebooks 2, 3 and 4. It is regenerated, not versioned.
