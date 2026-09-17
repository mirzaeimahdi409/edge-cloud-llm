# edge-cloud-llm

Thesis prototype: adaptive confidence-driven edge-cloud LLM inference. See
[`CLAUDE.md`](CLAUDE.md) for the full spec — this file only covers how to run
what is here.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`EdgeSimPy` is pulled from GitHub (not on PyPI) via the `edge_sim_py`
dependency in `pyproject.toml`; it needs network access to
`github.com/EdgeSimPy/EdgeSimPy` the first time it's installed.

The edge/cloud models (`Qwen/Qwen2.5-0.5B-Instruct` and
`Qwen/Qwen2.5-1.5B-Instruct`, section 15's default pair) download from the
Hugging Face Hub the first time they're used and are then cached locally.

## Running the tests

```bash
pytest -q                              # everything
pytest -q --ignore=tests/test_model_backend.py --ignore=tests/test_e2e.py
                                        # unit + fake-backend tests only (fast, offline)
```

`test_model_backend.py` and `test_e2e.py` load the real models and are slow
on first run (model download); everything else uses a deterministic fake
backend (`tests/fakes.py`) and runs in a few seconds.

## Running the comparison experiment (step 9)

```bash
PYTHONPATH=src python scripts/run_evaluation.py
```

Runs "always-edge" vs. the threshold policy over a few sample prompts and
writes `results/comparison.csv` and `results/comparison.json` (latency,
tokens generated, switch count, generated text per prompt/policy).

## Running the τ/k sweep (section 11)

```bash
PYTHONPATH=src python scripts/run_sweep.py
```

Runs every combination of threshold τ ∈ {0.2, 0.5, 0.8} and window size
k ∈ {4, 6, 8} over the sample prompts, writing `results/sweep.csv` and
`results/sweep.json`. Override the grid or prompts by calling
`edge_cloud_llm.evaluation.run_sweep(...)` directly.

## Plotting the results (section 13, optional)

```bash
PYTHONPATH=src python scripts/plot_results.py
```

Reads `results/comparison.csv` and `results/sweep.csv` (run the two scripts
above first) and writes PNG charts to `results/plots/`: latency and switch
count for always-edge vs. threshold policy, and a mean-latency /
mean-switch-count heatmap over the τ/k grid.

## Module layout

Mirrors the table in CLAUDE.md section 6:

| Module | Responsibility |
|---|---|
| `model_backend.py` | `ModelBackend` protocol + `EdgeModelAdapter`/`CloudModelAdapter` |
| `confidence.py` | Confidence Extractor (max-prob / entropy / margin, with temperature calibration) |
| `policy.py` | Decision Policy Engine (sliding-window threshold) |
| `communication.py` | Sentence/clause-boundary check used to gate re-prefill switches |
| `fault_tolerance.py` | Cloud call timeout + fallback to edge |
| `observability.py` | Structured per-step event log, dumped to CSV/JSON |
| `middleware.py` | The generation loop that wires all of the above together |
| `edgesim_scenario.py` | Runs the middleware inside an EdgeSimPy network scenario |
| `evaluation.py` | The always-edge vs. threshold-policy comparison and the τ/k sweep (used by `scripts/run_evaluation.py` / `scripts/run_sweep.py`) |

## Known limitations / open decisions (section 15)

- Only `always-edge` is implemented as a comparison baseline so far; the
  literature baselines in CLAUDE.md section 11 (RASOUL, EdgeShard, CE-CoLLM,
  SLED, QoS-Aware Routing, ...) are out of scope for this pass.
- The "cloud" model in `results/*.csv` is just a bigger model on the *same*
  CPU, not a faster/dedicated backend — on this hardware it runs at roughly
  5s/token (benchmarked directly), an order of magnitude slower than the
  edge model. Latency comparisons here should not be read as representative
  of a real edge-vs-cloud hardware trade-off; they mainly exercise the
  switching *logic*, not realistic edge/cloud economics.
- `run_with_timeout` (`fault_tolerance.py`) decides whether to fall back to
  edge based on real wall-clock time. Given the cloud model's ~5s/token cost
  sitting close to the timeout, CPU contention on this machine measurably
  changes which fallback path gets taken between runs — the *same*
  (prompt, τ, k) config produced switch-count-1 runs anywhere from ~6s to
  ~125s across separate script invocations. The switching decision logic
  itself is deterministic (greedy decoding, no randomness); the variance
  comes from the timeout racing real inference time under load.
- Section 11 asks for "several repeated runs to report a confidence
  interval" — `run_comparison`/`run_sweep` currently do a single run per
  config, not repeated trials with mean/CI. Given the run-to-run variance
  above, repeated trials (or a deterministic/mocked latency model instead of
  a real wall-clock timeout) would be needed before treating any single
  `results/*.csv` latency number as a stable estimate.
- No temperature-scaling calibration procedure has been run yet — the
  calibration hook (`ConfidenceExtractor.calibration`) exists but defaults to
  `temperature=1.0` for both backends.
