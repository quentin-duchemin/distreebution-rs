# DisTreebution·rs — project landing page

A single-file GitHub Pages site presenting **CRPS-RF** and **PMQRF**, the Rust
acceleration of the [DisTreebution](https://github.com/quentin-duchemin/DisTreebution)
distributional regression forests.

The whole page is `index.html` — no build step, no dependencies (fonts load from
Google Fonts, everything else is inline).

## Deploy on GitHub Pages

**Option A — root of a repo**
1. Put `index.html` at the root of your repository (or in a `docs/` folder).
2. Repo → **Settings → Pages**.
3. Under *Build and deployment*, set **Source: Deploy from a branch**, pick your
   branch and either `/ (root)` or `/docs`.
4. Save. The site publishes at `https://<user>.github.io/<repo>/`.

**Option B — dedicated `gh-pages` branch**
```bash
git checkout --orphan gh-pages
git rm -rf .
cp /path/to/index.html .
git add index.html && git commit -m "Landing page"
git push origin gh-pages
```
Then set Pages source to the `gh-pages` branch.

## Customising

Everything is driven by CSS custom properties at the top of the `<style>` block:

| Variable | Meaning |
|----------|---------|
| `--q1 … --q5` | the quantile-fan colour spectrum (dark → light) |
| `--signal` | the warm accent used for CTAs and the section index numbers |
| `--ink`, `--paper` | primary text and background |
| `--disp`, `--body`, `--mono` | display, body and monospace font stacks |

The benchmark numbers, results table, and citation are plain HTML — edit them in
place. The hero "quantile fan" is an inline SVG in the `.fan-wrap` block; the band
`<path>` shapes can be tuned directly.

## Attribution

The algorithms, theory, and reference implementation are the work of
**Quentin Duchemin and Guillaume Obozinski** (Swiss Data Science Center), described in
*Efficient distributional regression trees learning algorithms for calibrated
non-parametric probabilistic forecasts* (2025). This page presents a Rust port of
their methods and credits them throughout; update the citation block and links if
details change.
