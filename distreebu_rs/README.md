# distreebu_rs — Rust backend for DisTreebution

Rust/PyO3 reimplementation of the hot paths of the DisTreebution distributional
regression-tree package. Same numerical results, ~10–100× faster.

## Contents

    distreebu_rs/        Rust crate (compile to a Python wheel)
      src/lib.rs         all data structures + entropy functions + trees
      Cargo.toml         crate manifest (pyo3 + rayon)
      pyproject.toml     maturin build config
    DisTreebution_rs/    pure-Python drop-in shims (thin wrappers over the wheel)

## What is ported

| Component                         | Rust class / function              |
|-----------------------------------|------------------------------------|
| FenwickTree                       | `FenwickTree`                      |
| MinMaxHeap                        | `MinMaxHeap`                       |
| Quadratic entropies               | `entropies_quadratic`              |
| CRPS entropies                    | `entropies_crps`                   |
| Multi-quantile entropies          | `entropies_multi_quantiles`        |
| RegressionTreeQuadratic (RT)      | `RegressionTreeQuadratic`          |
| RegressionTree (CRPS)             | `RegressionTreeCRPS`               |
| RegressionTreeQuantile (QRT)      | `RegressionTreeQuantile`           |

Not ported (kept in pure Python): WBTree, UQ/conformalisation,
`get_values_leaf_and_groups`, and the `limit_use_CRPS` hybrid mode.

## Build

    pip install maturin
    cd distreebu_rs
    maturin build --release
    pip install target/wheels/distreebu_rs-*.whl

## Use directly

    import distreebu_rs as rs
    tree = rs.RegressionTreeCRPS(max_depth=6, min_samples_split=20)
    tree.fit(X_list, y_list)                     # X: list[list[float]], y: list[float]
    leaves = tree.get_values_leaf(Xq_list, idxs) # Xq is the query matrix

## Use as a DisTreebution drop-in

Put `DisTreebution_rs/` on your path and swap the import prefix:

    # from DisTreebution.CRPSRT.RegressionTree import RegressionTree
    from DisTreebution_rs.CRPSRT import RegressionTree

## Performance parallelism note

The feature loop parallelizes with rayon. Set `RAYON_NUM_THREADS` to control it.
If you also parallelize *trees* yourself (e.g. joblib), set `RAYON_NUM_THREADS=1`
to avoid oversubscription.

## Tie-breaking caveat

For datasets with exactly-tied y-values, results can differ from the original
Python because that code uses numpy's unstable argsort (its own output is
tiebreak-dependent). This backend uses a stable, canonical tiebreak. Never
triggers on continuous targets.
