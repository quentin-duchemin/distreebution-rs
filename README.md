<h1 align="center">DisTreebution·rs</h1>

<p align="center">
  <b>Fast distributional regression forests in Rust</b><br>
  A native-code implementation of the CRPS-RF and PMQRF algorithms from
  <a href="https://github.com/quentin-duchemin/DisTreebution">DisTreebution</a>.
</p>

<p align="center">
  <a href="https://quentin-duchemin.github.io/distreebution-rs"><b>→ Presentation page</b></a>
</p>

---

## What this is

Most regression models predict a single number — the conditional mean. **Distributional
regression** predicts the *entire* conditional distribution of the target, so every forecast
comes with calibrated uncertainty (quantiles, prediction intervals, a full CDF).

DisTreebution builds tree ensembles for exactly this, by splitting on **proper scoring rules**
rather than squared error:

- **PMQRF** — *Pinball Multi-Quantile Regression Forest.* Predicts several quantiles at once by
  minimising a generalised entropy derived from the Weighted Interval Score (a sum of pinball
  losses). All quantile levels share the same splits, so the forecast stays monotone and
  interpretable.
- **CRPS-RF** — *CRPS Regression Forest.* Splits by directly minimising the Continuous Ranked
  Probability Score, a strictly proper scoring rule for full-distribution prediction. It rewards
  forecasts that are both sharp and well-calibrated.

The methodology, theory, and the leave-one-out unbiased estimator of the information gains are
described in **Duchemin & Obozinski (2025)**, *Efficient distributional regression trees learning
algorithms for calibrated non-parametric probabilistic forecasts.* The full method documentation
lives at [quentin-duchemin.github.io/DisTreebution](https://quentin-duchemin.github.io/DisTreebution/).

## Why a Rust implementation

The split-selection procedures are already **O(N log N)** per node — the min–max heaps (PMQRF)
and Fenwick tree (CRPS-RF) maintain the required order statistics incrementally, so the
asymptotic complexity was never the bottleneck. What a pure-Python implementation pays instead is
a large **constant factor**: interpreter overhead, per-sample object allocation, and
cache-unfriendly memory access.

This crate re-implements the core in Rust (via [PyO3](https://pyo3.rs)) and keeps the exact same
algorithm. Concretely, it:

- maintains the same **O(N log N)** heap / Fenwick split search, with identical leaf partitions to
  the Python version on continuous data;
- stores features **column-major** so each per-feature scan stays in cache;
- **releases the GIL** during fitting and parallelises the feature search across cores with
  [rayon](https://github.com/rayon-rs/rayon).

The result is a **10–100× wall-clock speedup** for tree fitting. For a single tree, cost is
**linear in the number of features `d`** and **near-linear in `N`** (the log N factor); a forest
just multiplies by the number of trees and divides by the cores available. Benchmarks and scaling
plots are on the [presentation page](https://quentin-duchemin.github.io/distreebution-rs).

## Install

A prebuilt wheel is included — no Rust toolchain required:

```bash
pip install distreebu_rs-0.2.0-cp312-cp312-manylinux_2_17_x86_64.whl
```

or straight from this repository:

```bash
pip install https://github.com/quentin-duchemin/distreebution-rs/raw/main/distreebu_rs-0.2.0-cp312-cp312-manylinux_2_17_x86_64.whl
```

To build from source (needs a Rust toolchain and [maturin](https://github.com/PyO3/maturin)):

```bash
pip install maturin
cd distreebu_rs
maturin build --release
pip install target/wheels/distreebu_rs-*.whl
```

## Usage

```python
import numpy as np
import distreebu_rs as rs

X = np.random.randn(3000, 40)
y = np.sin(2 * np.pi * X[:, 0]) + 0.3 * np.random.randn(3000)

# CRPS regression tree
tree = rs.RegressionTreeCRPS(max_depth=6, min_samples_split=20)
tree.fit(X.tolist(), y.tolist())

# Route query points to their leaves and read off the pooled training targets
leaves = tree.get_values_leaf(X.tolist(), list(range(len(X))))
```

The package exposes `RegressionTreeCRPS`, `RegressionTreeQuadratic`,
`RegressionTreeQuantile`, and the supporting `FenwickTree` and `MinMaxHeap` data structures, plus
the raw entropy functions `entropies_crps`, `entropies_quadratic`, and
`entropies_multi_quantiles`.

### Drop-in replacement for DisTreebution

The `DisTreebution_rs/` package mirrors the original Python API. Swap the import prefix and keep
the rest of your code unchanged:

```python
# before
from DisTreebution.CRPSRT.RegressionTree import RegressionTree
# after
from DisTreebution_rs.CRPSRT import RegressionTree
```

## Repository layout

```
distreebution-rs/
├── index.html          # source of the presentation / GitHub Pages site
├── distreebu_rs/       # the Rust crate (src/lib.rs, Cargo.toml, pyproject.toml)
├── DisTreebution_rs/   # Python drop-in shims over the compiled module
└── distreebu_rs-0.2.0-…-manylinux_2_17_x86_64.whl
```

## Scope and limits

This port covers the core tree types and their split engines. Not ported (kept in the Python
package): the `WBTree`, the conformalisation / UQ layer, `get_values_leaf_and_groups`, and the
`limit_use_CRPS` hybrid mode. For exactly-tied target values the split can differ from the
reference, because the Python code relies on NumPy's unstable argsort; this port uses a stable,
canonical tiebreak and never triggers on continuous targets.

## Citation

```bibtex



@article{duchemin2026,
	author = {Quentin Duchemin and Guillaume Obozinski},
	title = {Efficient Distributional Regression Trees Learning Algorithms for Calibrated Non-Parametric Probabilistic Forecasts},
	journal = {Journal of Computational and Graphical Statistics},
	volume = {0},
	number = {0},
	pages = {1--17},
	year = {2026},
	publisher = {Taylor \& Francis},
	doi = {10.1080/10618600.2026.2675431},
	URL = { 
		https://doi.org/10.1080/10618600.2026.2675431
	},
	eprint = { 
		https://doi.org/10.1080/10618600.2026.2675431
	}
}

```

## Links

- **Presentation page:** https://quentin-duchemin.github.io/distreebution-rs
- **Method documentation:** https://quentin-duchemin.github.io/DisTreebution/
- **DisTreebution (Python):** https://github.com/quentin-duchemin/DisTreebution

---

<sub>Method & theory © 2025 Q. Duchemin & G. Obozinski · Swiss Data Science Center.</sub>
