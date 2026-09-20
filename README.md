# Geopolitical Turning Points and Oil Prices

The baseline is a lag-augmented instrumental-variable local projection with **HC3 standard errors and no HAC correction**. Conventional LP–Newey–West regressions are a separate comparison.

## Run the analysis

Unzip the entire package and preserve its folder structure. From the package directory, use Python 3.12:

```bash
python -m pip install -r requirements.txt
python run_notebook.py
python -m unittest discover -s tests -v
python code/build_paper.py
```

For interactive Jupyter, install `requirements-notebook.txt`, open `Geopolitical_Turning_Points_Replication.ipynb`, and select **Restart Kernel and Run All**. The notebook uses standard Python cells without magics. Its explanations, equations, sample displays and hand calculations precede the complete estimation. No precomputed result is an input; no internet access or credentials are needed after dependency installation.

## Exact baseline

- Treatment: natural logarithm of one plus the raw geopolitical index.
- Instrument: second difference of the **raw** index.
- Outcome: log real Brent or WTI at `t+h`, for `h=0,...,48`.
- Controls: constant; **three price lags**; **two activity lags**; **two oil-production-growth lags**; **two lags of every included geopolitical treatment**.
- Nine separate-category models per price; one joint model containing military conflict, diplomatic tension and nuclear threat.
- Covariance: **HC3**, using structural IV residuals divided by one minus full projected-design leverage, including controls and the constant. No additional HC1 multiplier and no HAC bandwidth.
- Normal critical values for pointwise intervals; asymptotic chi-square Wald tests.

The extra price lag is the specified augmentation. Three lags of every variable are a separate sensitivity. Pointwise HC3 maintains negligible serial covariance of the IV score; the lag pattern is not claimed to prove validity for arbitrary instruments.

## Anticipation and alternative models

Anticipation OLS uses the **same `Z[t+2]`** against outcomes at **`t` and `t+1`**, with origin-t predetermined controls and HC3. Joint tests retain cross-equation covariance using aligned influence rows. Because future curvature contains `G[t]` and `G[t+1]`, the exercise diagnoses conditional association with future-dated curvature rather than isolating only information first revealed in `t+2`.

`inference_comparison.csv` reports matched-sample augmented HC3 and conventional **two-price-lag Newey–West with six-month bandwidth** estimates. Each model has its own coefficient and standard error. `hac_bandwidth.csv` varies bandwidths 4, 12, 24 and 48 for the conventional model. Neither supplies the baseline confidence bands. HC1 is retained as a finite-sample sensitivity on the unchanged augmented model.

Other computations include HC3 first stages and rank diagnostics; formal homoskedastic Sanderson–Windmeijer conditional F statistics for the joint model; category-equality and selected-path tests; event-window omissions; origin splits; matched lag variants; log-curvature contrasts; contemporaneous-production controls; and instrument concentration. Joint lag-augmented IV projections for world industrial production and global oil-production growth provide demand- and physical-supply-channel evidence without being interpreted as a formal mediation decomposition. Simultaneous bands use 1,999 seeded draws: independent month multipliers are the baseline, while 6/12-month blocks are supplementary path-inference sensitivities. Coverage is over 49 horizons of one curve, not over all models.

## Files

| File or directory | Purpose |
|---|---|
| `Geopolitical_Turning_Points_Replication.ipynb` | Equations, explanations, computations and saved outputs |
| `code/analysis.py` | Complete estimation and diagnostic implementation |
| `code/figures.py` | Validated title-free and footnote-free PNG/PDF/SVG exports |
| `code/build_paper.py` | Word article generated from numerical tables, native OMML math |
| `paper/` | Autonomous article with figure notes, tables, expanded references and data appendix |
| `data/monthly_data.csv` | Frozen numerical input |
| `data/source_csv/` | Complete frozen provider histories reconstructing the input merge |
| `data/Geopolitical_Turning_Points_Data.xlsx` | Excel copy, formulas, dictionary and sources |
| `data/input_checksums.json` | Frozen-input checksums |
| `results/` | Recomputed coefficients, diagnostics and metadata |
| `figures/` | Five paper figures, one supplementary mechanism figure, and separate caption metadata |
| `tests/` | Independent algebraic, timing, HC3, AR, data and artifact checks |
| `DATA_SOURCES.md` | Sources, current vintages, transformations and missing observations |
| `VALIDATION.md` | Method and computational validation scope |
| `run_notebook.py` | Sequential cell runner saving native notebook outputs |
| `build_notebook.py` | Transparent notebook source |

Figures show `100*log(1.10)*beta`: log-price points multiplied by 100 for a 10% increase in **one plus** the index. Exact percentage responses in text use `100*expm1(log(1.10)*beta)`.

## Execution boundary

All cells were run sequentially in a fresh Python process, and the package was rerun in a clean directory. A native Jupyter kernel was unavailable; frontend/kernel execution is not claimed. Numerical dependencies are pinned and optional frontend dependencies are listed separately. The included runner is the tested execution route.

Computational agreement establishes implementation consistency, not exclusion or finite-sample coverage. Data remain frozen at the documented endpoints. Updating a vintage requires deliberately updating snapshots and checksums.
