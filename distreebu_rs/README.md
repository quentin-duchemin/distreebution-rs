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

## Install

Two prebuilt wheels are shipped, differing only in the minimum glibc they require.
Pick the one that matches your system (`ldd --version` shows your glibc):

| Wheel | Minimum glibc | Covers |
|-------|---------------|--------|
| `…-manylinux_2_28_x86_64.whl` | **2.28** | Debian 10+, Ubuntu 18.10+, RHEL/CentOS/Alma/Rocky 8+ |
| `…-manylinux_2_17_x86_64.whl` | **2.17** | older still — CentOS 7, Ubuntu 14.04+, most legacy hosts |

    pip install distreebu_rs-0.3.0-cp312-cp312-manylinux_2_28_x86_64.whl
    # or, for maximum reach:
    pip install distreebu_rs-0.3.0-cp312-cp312-manylinux_2_17_x86_64.whl

If you are unsure, the `manylinux_2_17` wheel runs everywhere the `2_28` one does;
the only reason to prefer `2_28` is that it is the tighter, more modern target.

## Build from source

    pip install maturin
    cd distreebu_rs
    maturin build --release                      # targets the host glibc

To reproduce the portable wheels, cross-compile against an older glibc with
[cargo-zigbuild](https://github.com/rust-cross/cargo-zigbuild) (no Docker needed):

    pip install maturin ziglang cargo-zigbuild
    maturin build --release --zig --compatibility manylinux_2_28   # glibc 2.28
    maturin build --release --zig --compatibility manylinux_2_17   # glibc 2.17

    pip install target/wheels/distreebu_rs-*.whl

## Use directly

    import distreebu_rs as rs
    tree = rs.RegressionTreeCRPS(max_depth=6, min_samples_split=20)
    tree.fit(X_list, y_list)                     # X: list[list[float]], y: list[float]
    leaves = tree.get_values_leaf(Xq_list, idxs) # Xq is the query matrix

## Saving and loading models

Every fitted tree can be persisted. Three equivalent routes:

    # 1. save / load to a JSON file on disk
    tree.save("model.json")
    tree = rs.RegressionTreeCRPS.load("model.json")

    # 2. to_json / from_json (in-memory string, e.g. for a database or S3)
    blob = tree.to_json()
    tree = rs.RegressionTreeCRPS.from_json(blob)

    # 3. standard pickle (works for single trees and lists/forests)
    import pickle
    pickle.dump(forest, open("forest.pkl", "wb"))     # forest = list of trees
    forest = pickle.load(open("forest.pkl", "rb"))

The serialised model stores only what prediction needs — the hyper-parameters and
the tree topology (split feature, threshold, child ids, leaf y-values). It does
**not** store the training matrix, so a reloaded model must be queried with an
explicit `X` (the normal call pattern). The format is a small, human-readable JSON
document with a version field for forward compatibility. The same `save`, `load`,
`to_json`, and `from_json` methods are available on `RegressionTreeQuadratic`,
`RegressionTreeCRPS`, and `RegressionTreeQuantile`; the quantile levels are
preserved for `RegressionTreeQuantile`.

## Use as a DisTreebution drop-in

Put `DisTreebution_rs/` on your path and swap the import prefix:

    # from DisTreebution.CRPSRT.RegressionTree import RegressionTree
    from DisTreebution_rs.CRPSRT import RegressionTree

The shim classes also expose `.save(path)` / `.load(path)` and
`.to_json()` / `.from_json(str)`.

## What's new in v0.3

- **Model persistence** — `save`/`load`, `to_json`/`from_json`, and full `pickle`
  support on every tree type (and on lists/forests via pickle). See the section above.
- Two portable wheels shipped: glibc **2.17** and **2.28** builds, both produced by a
  genuine cross-compile (not binary patching).

## Performance history (v0.2)

The v0.2 release focused on speed. Relative to the first port:

1. CRPS rank computation O(n^2) -> O(n log n) (value-bucketed Fenwick).
2. Removed the redundant O(n) left-count scan in the split loop (now O(1)).
3. Column-major feature storage for cache-friendly per-feature scans.
4. rayon-parallel feature loop (nodes >= 512 samples); GIL released during fit.

## Performance parallelism note

The feature loop parallelizes with rayon. Set `RAYON_NUM_THREADS` to control it.
If you also parallelize *trees* yourself (e.g. joblib), set `RAYON_NUM_THREADS=1`
to avoid oversubscription.

## Tie-breaking caveat

For datasets with exactly-tied y-values, results can differ from the original
Python because that code uses numpy's unstable argsort (its own output is
tiebreak-dependent). This backend uses a stable, canonical tiebreak. Never
triggers on continuous targets.
