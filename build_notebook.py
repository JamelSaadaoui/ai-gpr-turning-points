"""Build a readable native Jupyter notebook without a notebook-library dependency."""

from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parent
cells = []


def markdown(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)})


def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": textwrap.dedent(source).strip().splitlines(True)})


markdown(r"""
# Geopolitical Turning Points and Oil Prices

## A complete reproducible analysis

This notebook estimates how real oil prices respond to geopolitical variation selected by a second-difference filter. It rebuilds the analysis dataset from the included source snapshots, validates the monthly calendar, estimates lag-augmented instrumental-variable local projections, runs the anticipation and robustness exercises, and exports the manuscript figures and numerical tables. **All estimates are recomputed when the notebook is run. No precomputed result is an input.**

The central empirical distinction is between the **instrument**, the second difference of the raw event index, and the **treatment**, the logarithm of one plus that index. The turning-point filter suppresses smooth movements and emphasizes abrupt accelerations and reversals. Its purpose is to focus identification on variation argued to be more plausibly exogenous; filtering alone is not a test of that identifying restriction.

### How to use this notebook

Unzip the package, install the dependencies in `requirements.txt` and (for a Jupyter frontend) `requirements-notebook.txt`, and open the notebook anywhere inside the extracted package. Select **Restart Kernel and Run All**. The offline command-line alternative is `python run_notebook.py`. Outputs are regenerated in `results/` and `figures/`. The notebook contains no network request and never changes the frozen provider files.

**Reading order:** inspect the data and equations first; the full computation is launched only after a single-regression calculation and the anticipation timing have been made explicit. Subsequent sections display freshly estimated outputs and explain the null hypothesis or comparison behind each exercise.
""")

code(r"""
from pathlib import Path
import os
import sys
import json
import hashlib
import importlib.metadata
import platform

# Set numerical thread counts before importing NumPy for consistent small-matrix speed.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import pandas as pd

def locate_package_root(start):
    # Find the extracted package without assuming the Jupyter working directory.
    start = Path(start).resolve()
    direct = [start, *start.parents]
    nested = []
    for base in direct[:3]:
        if base.is_dir():
            nested.extend(path for path in base.glob("Geopolitical_Turning_Points_Replication*")
                          if path.is_dir())
    candidates = []
    for candidate in [*direct, *nested]:
        if ((candidate / "code" / "analysis.py").is_file() and
                (candidate / "data" / "source_csv").is_dir()):
            candidates.append(candidate)
    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise FileNotFoundError(
            "Could not locate the extracted replication package. Keep the code/ and data/ "
            "folders beside this notebook, then restart the kernel.")
    return unique[0]

ROOT = locate_package_root(Path.cwd())
sys.path.insert(0, str(ROOT / "code"))
from analysis import (EVENTS, PRIMARY, LABELS, HORIZONS, prepare_data,
                      make_design, make_mechanism_design, fit_iv, fit_anticipation,
                      hac_covariance, run_all)
from build_data import build as build_data
from figures import make_all_figures

# A normal Jupyter kernel supplies rich display. The command-line runner also
# captures display() into native notebook outputs, without requiring IPython.
if not globals().get("_REPLICATION_CELL_RUNNER", False):
    try:
        from IPython.display import display, Image
    except ImportError:
        def display(value):
            print(value)
        def Image(filename):
            return Path(filename)

pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda value: f"{value:.5f}")

CONFIG = {
    "draws": 1999,
    "seed": 20260919,
    "maximum_horizon": 48,
    "price_lags": 3,       # Two core lags plus one lag augmentation.
    "control_lags": 2,
    "treatment_lags": 2,
    "anticipation_horizons": [0, 1],
    "anticipation_instrument_lead": 2,
    "shock_log_increment": float(np.log(1.10)),
    "baseline_covariance": "HC3, no HAC",
    "conventional_comparison": "2 price lags and NW6",
}
EVENT_LABELS = {event: LABELS[event] for event in EVENTS}
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
assert tuple(HORIZONS) == tuple(range(CONFIG["maximum_horizon"] + 1))
display(pd.Series(CONFIG, name="Configuration").to_frame())
print("Python", platform.python_version())
""")

markdown(r"""
## 1 Data provenance and complete current-source histories

`data/source_csv/` contains the provider histories used in the paper: AI-GPR event indices, monthly EIA WTI and Brent prices, the BLS all-items CPI, the Baumeister–Hamilton world-industrial-production workbook, and the EIA world crude-oil-production series. The cell below rebuilds the merged dataset directly from those files. Real prices and logged controls are constructed here; no historical segment is spliced, rebased, or extrapolated. The only fill is the unavailable October 2025 CPI level, log-linearly interpolated between the adjacent official observations; nominal Brent and WTI remain observed.

Freezing the downloaded source files makes an exact rerun possible even after providers revise history. The manuscript appendix and provenance file record the URLs, retrieval vintage, transformations, and checksums.
""")

code(r"""
rebuilt = build_data(ROOT)
frozen = pd.read_csv(ROOT / "data" / "monthly_data.csv", parse_dates=["date"])
pd.testing.assert_frame_equal(rebuilt, frozen, check_dtype=False, check_exact=False,
                              rtol=1e-10, atol=1e-10)
print(f"Direct current-source reconstruction verified: {len(frozen)} rows, "
      f"{len(frozen.columns)} analysis columns.")
display(frozen.head())
""")

markdown(r"""
## 2 Preserve calendar time before constructing any lag or lead

`prepare_data` reindexes the input to every calendar month. It does not perform any additional filling. Differences and lags crossing remaining missing values stay missing, and each regression removes only rows that lack one of its required variables. Dropping incomplete rows before constructing lags would falsely make distant observations adjacent; that is explicitly avoided.

The maximum origin date can precede the latest observed price because the controls are published at different dates. A horizon-specific outcome also changes the admissible sample. Every coefficient export therefore reports its sample size and first and last origin months.
""")

code(r"""
data = prepare_data(frozen)
assert data["date"].equals(pd.Series(pd.date_range(data["date"].min(), data["date"].max(), freq="MS"), name="date"))
availability = []
for variable in ["lwti", "lbrent", "lwip", "lgop", *EVENTS]:
    valid = data[variable].notna()
    first, last = data.loc[valid, "date"].min(), data.loc[valid, "date"].max()
    internal = data["date"].between(first, last) & ~valid
    availability.append({"Variable": variable, "First": str(first.date()), "Last": str(last.date()),
                         "Observed": int(valid.sum()), "Internal missing": int(internal.sum())})
display(pd.DataFrame(availability))
gaps = data.loc[data[["lwti", "lbrent", "lwip", "lgop"]].isna().any(axis=1),
                ["date", "lwti", "lbrent", "lwip", "lgop"]]
print("Months with an unavailable macroeconomic observation:")
display(gaps)
""")

markdown(r"""
## 3 The treatment, the instrument, and the identifying variation

For event category $j$, let $G_{j,t}$ denote its raw AI-GPR index. Define

$$x_{j,t}=\log(1+G_{j,t}),\qquad Z_{j,t}=\Delta^2G_{j,t}=G_{j,t}-2G_{j,t-1}+G_{j,t-2}.$$

The second difference is a **curvature filter**: a constant or linear trajectory has zero second difference, while a sudden acceleration, reversal, or return from a short-lived spike receives a large absolute weight. The filter uses all observations; there is no hidden threshold that keeps only selected dates. A sharp event can also produce adjacent positive and negative filtered values.

The economic identifying argument is that this selected turning-point variation operates through the corresponding geopolitical condition, including its consequences for expectations and decisions. The required moment is $E[\widetilde Z_{j,t}u_{t+h}]=0$, conditional on the included controls. The second difference motivates which variation is used; it does not, by itself, prove exclusion or non-anticipation.

Relevance concerns the first-stage relationship between $Z_{j,t}$ and $x_{j,t}$ **after removing the same controls on the same sample**. The baseline does not instrument $G_{j,t}$ with its own second difference while controlling its two raw lags. Instead it combines raw-index curvature with a nonlinear log treatment and log-treatment lags. Consequently, the baseline first stage is not the deterministic raw-level identity. Its partial $R^2$ and robust excluded-instrument test are measured below.
""")

code(r"""
event = "military_conflict"
example = data[["date", event, f"x_{event}", f"z_{event}"]].copy()
example["manual_second_difference"] = data[event] - 2 * data[event].shift(1) + data[event].shift(2)
np.testing.assert_allclose(example[f"z_{event}"], example["manual_second_difference"],
                           rtol=1e-12, atol=1e-12, equal_nan=True)
display(example.head(8))
""")

markdown(r"""
## 4 Exactly what the lag-augmented local projection estimates

For oil-price series $o$ and horizon $h=0,\ldots,48$, the separate-category equation is

$$p^o_{t+h}=\alpha^o_{h,j}+\beta^o_{h,j}x_{j,t}
+\sum_{\ell=1}^{3}\phi^o_{h,j,\ell}p^o_{t-\ell}
+\sum_{\ell=1}^{2}a^o_{h,j,\ell}\log WIP_{t-\ell}
+\sum_{\ell=1}^{2}b^o_{h,j,\ell}\Delta\log Q_{t-\ell}
+\sum_{\ell=1}^{2}c^o_{h,j,\ell}x_{j,t-\ell}+u^o_{t+h,j}.$$

Here $p^o$ is the log real WTI or Brent price, $WIP$ is world industrial production, and $Q$ is world oil production. The endogenous current treatment $x_{j,t}$ is instrumented by $Z_{j,t}$. The outcome is the **future log-price level**, not a cumulative change. The three price lags implement two core lags plus one lag augmentation; the activity, production-growth, and treatment blocks retain two lags.

The first stage is

$$x_{j,t}=\mu_j+\pi_j Z_{j,t}+\Lambda_j'W_{j,t-1}+v_{j,t},$$

where $W$ contains exactly the controls displayed above. The joint specification replaces the scalar current treatment and instrument with the three-component vectors for military conflict, diplomatic tension, and nuclear threats, and includes two lags of all three treatments. It does not include the aggregate index together with its event decomposition.

**Interpretation:** conditional on the identifying assumptions, the coefficient refers to the geopolitical variation selected by these instruments, rather than every possible change in geopolitical risk. A literal binary-treatment complier LATE is not claimed for these continuous indices without further assumptions.
""")

code(r"""
design = make_design(data, outcome="lbrent", events=("military_conflict",), horizon=12)
print("Dependent variable: log real Brent in t+12")
print("Endogenous regressor: log(1 + military_conflict in t)")
print("Excluded instrument: raw military_conflict curvature in t")
print("Included controls:")
print("\n".join(design.control_names))
print(f"N = {len(design.y)}; origin dates {design.dates.min().date()} to {design.dates.max().date()}")
fit = fit_iv(design)
display(pd.DataFrame({"Coefficient": fit.beta,
                      "Primary HC3 standard error": np.sqrt(np.diag(fit.hc3_cov)),
                      "HC1 sensitivity standard error": np.sqrt(np.diag(fit.hc1_cov))},
                     index=["Military conflict, Brent, h=12"]))
""")

markdown(r"""
## 5 A transparent Frisch–Waugh–Lovell calculation

Let $M_W$ project off the constant and included controls on the **same horizon-specific sample**. Denote $\widetilde y=M_Wy$, $\widetilde X=M_WX$, and $\widetilde Z=M_WZ$. Exactly identified IV solves

$$\widehat\beta_h=(\widetilde Z'\widetilde X)^{-1}\widetilde Z'\widetilde y.$$

In the single-treatment case this is simply a ratio of two scalar cross-products. `fit_iv` uses an SVD projection and rescales instrument columns for numerical stability. This changes neither the identifying moment nor the coefficient. Standard errors use the **structural second-stage residual** $\widetilde y-\widetilde X\widehat\beta$, not a first-stage residual. Primary pointwise inference uses **HC3**, with **no HAC correction and no bandwidth** in the augmented model. Let $d_t$ be the leverage from the full projected-regressor design, including the controls and constant. In an exactly identified model its column space equals that of the full instrument matrix. The influence contribution is

$$s_t=(\widetilde Z'\widetilde X)^{-1}\widetilde Z_t\frac{\widehat u_{t+h}}{1-d_t},\qquad \widehat V_{HC3}=\sum_t s_ts_t'.$$

HC3 uses the observation-specific leverage correction, without an additional HC1 factor. This is the second-stage projected-design leverage convention. It is not a naive OLS covariance from regressing outcomes on fitted treatments. HC1 remains a finite-sample sensitivity. The model retains the specified two lags of fundamentals and treatments and three price lags. A separate conventional two-price-lag regression uses Newey–West errors. The maintained pointwise inference condition concerns serial covariance of the IV score; the lag pattern alone is not claimed to prove a theorem for arbitrary instruments.
""")

code(r"""
manual_beta = np.linalg.solve(fit.z_residual.T @ fit.x_residual,
                              fit.z_residual.T @ fit.y_residual)
np.testing.assert_allclose(manual_beta, fit.beta, rtol=1e-11, atol=1e-11)
print("FWL scalar cross-product estimate:", float(manual_beta[0]))
print("Estimator coefficient:", float(fit.beta[0]))
print("Verified equality within 1e-11 numerical tolerance.")
raw_scores = np.linalg.solve(fit.z_residual.T @ fit.x_residual, fit.z_residual.T).T * fit.residuals[:, None]
hc3_scores = raw_scores / (1 - fit.leverage[:, None])
np.testing.assert_allclose(hc3_scores.T @ hc3_scores, fit.hc3_cov, rtol=1e-10, atol=1e-11)
print("HC3 sandwich verified; maximum full-design leverage:", float(fit.leverage.max()))
""")

markdown(r"""
## 6 Anticipation has two horizons and one fixed future instrument date

For each separate category and for the three-category joint specification, estimate

$$p^o_{t+h}=a^o_{h,j}+\theta^o_{h,j}Z_{j,t+2}+\Gamma^o_{h,j}{}'W_{j,t-1}+e^o_{t+h,j},\qquad h\in\{0,1\}.$$

This is an OLS placebo regression, **not** an IV response. The instrument is always dated **$t+2$**. When $h=0$, the oil price is observed two months before the instrument; when $h=1$, it is observed one month before. No control is advanced into the future, and $h\geq2$ is not an anticipation horizon for this exercise. Tests report the individual nulls $\theta_0=0$ and $\theta_1=0$ and a joint zero restriction across the two pre-event coefficients. Failure to reject is supportive timing evidence, not a proof of exclusion.

The dated curvature $Z_{t+2}=G_{t+2}-2G_{t+1}+G_t$ also contains index values already realized at $t$ and $t+1$. Thus this exercise tests pre-date association with the future-dated curvature measure; it does not isolate a wholly new innovation first revealed at $t+2$.

The coefficients use raw instrument units and must not be interpreted using the IV treatment normalization. Anticipation results appear as tables; there is no anticipation response-path figure.
""")

code(r"""
timing = pd.DataFrame({
    "Horizon": [0, 1], "Oil-price date": ["t", "t+1"],
    "Turning-point date": ["t+2", "t+2"],
    "Months before turning point": [2, 1],
    "Controls": ["Origin-t predetermined controls", "Origin-t predetermined controls"]})
display(timing)
for h in CONFIG["anticipation_horizons"]:
    placebo = fit_anticipation(data, "lbrent", ("military_conflict",), h)
    print(f"h={h}: theta={placebo['beta'][0]:.7f}, N={placebo['n']}")
""")

markdown(r"""
## 7 Run the full analysis from the frozen data

The next cell recomputes all coefficients and diagnostics, including 49 horizons for each of nine separate geopolitical indices and two oil prices, the three-treatment joint model, anticipation, and the robustness exercises. The `analysis.py` source contains the complete implementation; no result-reading shortcut is used. The declared multiplier calculation uses 1,999 draws and a fixed seed. Lower draw counts are suitable only for smoke tests, not the reported results.

Files are written only after the calculations have been assembled. `run_metadata.json` records configuration, versions, source hash, and output counts. The output dictionary below stays in memory for all displays and figures that follow.
""")

code(r"""
results = run_all(frozen, RESULTS, draws=CONFIG["draws"], seed=CONFIG["seed"])
display(pd.DataFrame({"Output": list(results), "Rows": [len(value) for value in results.values()]}))
assert len(results["baseline_iv"]) == 2 * 49 * (9 + 3)
assert set(results["anticipation"]["horizon"]) == {0, 1}
assert results["anticipation"]["instrument_lead"].eq(2).all()
""")

markdown(r"""
## 8 Describe the data and generate the publication figures

Summary statistics below use each variable's available observations. They are not represented as a single common regression sample. `endpoints.csv` gives the corresponding coverage, while every regression export contains its own origin dates and sample size.

The figure files contain **no title and no footnote**. Captions and explanatory notes are exported separately in `figures/figure_captions.json` for inclusion in Word. Figures use a common restrained visual design, legible axis labels, and pointwise 90% and 95% heteroskedasticity-robust HC3 intervals. The plot normalization is $100\log(1.10)\widehat\beta_h$: a log-price change multiplied by 100 for a 10% increase in $1+G$, not necessarily a 10% rise in the raw index. It is an approximate percentage-price response, not an exact nonlinear percentage conversion.
""")

code(r"""
display(results["summary_statistics"])
captions = make_all_figures(data, results, FIGURES, EVENT_LABELS)
display(Image(filename=str(FIGURES / "fig01_all_variables.png")))
display(Image(filename=str(FIGURES / "fig02_turning_points.png")))
""")

markdown(r"""
## 9 Separate-category local projections

Each curve answers how the oil-price level changes with the current log geopolitical index when its variation is selected by the raw turning-point instrument and the stated lag controls. Separate-category models do not condition on the other current event categories; the joint model below addresses their overlap. Brent is shown first, with WTI serving as the second benchmark.

Pointwise intervals answer a separate zero-coefficient question at each horizon. A sequence of individually significant points does not by itself establish significance of the whole response path. The joint path tests and simultaneous bands are reported separately.
""")

code(r"""
baseline = results["baseline_iv"]
display(baseline.loc[(baseline["model"] == "single") & baseline["horizon"].isin([0, 12, 24, 48]),
                     ["outcome", "event", "horizon", "n", "beta", "hc3_se", "hc3_p", "hc1_p"]])
display(Image(filename=str(FIGURES / "fig03_brent_separate_irfs.png")))
display(Image(filename=str(FIGURES / "fig03_wti_separate_irfs.png")))
""")

markdown(r"""
## 10 The joint geopolitical specification and relevance diagnostics

The joint specification estimates military-conflict, diplomatic-tension, and nuclear-threat effects together. Each current treatment is instrumented by its own raw curvature series, with the three instruments entering every first-stage equation. Their lagged treatments are included together as controls.

The single-category first-stage statistic measures excluded-instrument relevance after controls. For the three-treatment model, `sw_conditional_first_stage.csv` implements the homoskedastic Sanderson–Windmeijer correction after conditionally instrumenting each exposure on the other two. Canonical correlations and the numerical cross-moment rank provide complementary geometry. The conditional first stages are reported because the principal specification contains three endogenous geopolitical categories.
""")

code(r"""
display(Image(filename=str(FIGURES / "fig04_joint_irfs.png")))
display(results["first_stage"].loc[results["first_stage"]["horizon"].isin([0, 24, 48])])
display(results["joint_identification"].loc[
    (results["joint_identification"]["model"] == "joint") &
    results["joint_identification"]["horizon"].isin([0, 24, 48])])
display(results["sw_conditional_first_stage"].loc[
    results["sw_conditional_first_stage"]["horizon"].isin([0, 12, 24, 48])])
""")

markdown(r"""
## 11 Anticipation estimates and their joint nulls

The table reports both permitted pre-event horizons, the fixed two-month instrument lead, and HC3 and HC1 inference. The two-coefficient joint tests allow for covariance between the $h=0$ and $h=1$ estimates. For the three-category joint specification, an additional six-coefficient test considers all three categories at both anticipation horizons. These are tests about prices **before** the future turning point, not responses after it.
""")

code(r"""
display(results["anticipation"][["model", "outcome", "event", "horizon", "instrument_lead", "n",
                                  "beta", "hc3_p", "hc1_p"]])
display(results["anticipation_joint_tests"])
""")

markdown(r"""
## 12 Robustness exercises and what each one can establish

The exercises below address different concerns; they are not interchangeable tests of instrument validity.

| Exercise | Precisely what changes or is tested | Output |
|---|---|---|
| HC3 versus HC1 | Same augmented regression; different finite-sample heteroskedasticity corrections | `baseline_iv.csv` |
| Augmentation versus Newey–West | Three-price-lag HC3 versus two-price-lag NW6, with each model re-estimated on matched observations | `inference_comparison.csv` |
| Conventional HAC bandwidth sensitivity | Two-price-lag model with 4-, 12-, 24-, and 48-month Bartlett bandwidths | `hac_bandwidth.csv` |
| Equality across categories | Within the joint model, equality rather than separate significance is tested directly | `cross_category_equality.csv` |
| Joint path Wald test | Zero coefficients at the predeclared horizons 0, 6, 12, 24, 36, 48 | `path_wald.csv` |
| Simultaneous bands | Max-$t$ coverage over 49 horizons within one curve; Independent month multipliers baseline; 6/12-month blocks supplementary | `simultaneous_bands.csv` |
| Event-window omission | Re-estimate after removing origin months within one month of listed episodes | `event_drop.csv` |
| Historical origin-date splits | Estimate the long samples ending before 2012 and before 2019 | `origin_splits.csv` |
| Price-lag sensitivity | Two or four price lags, or three lags of every variable, on matched observations | `lag_sensitivity.csv` |
| Log-curvature contrast | Replace raw curvature with curvature of the logged treatment | `log_curvature.csv` |
| Current production control | Add current oil-production growth on a matched sample | `production_control.csv` |
| Cross-moment concentration | Inspect largest raw and residualized first-stage contributions | `leverage.csv` |

The event and split exercises refer to origin dates. At a fixed horizon, the two origin subsamples have disjoint outcome dates, but forecast-error windows can overlap across the cutoff, so their estimates are not treated as independent. Current production growth is a conditional control, not an identified oil-supply shock; including it can remove part of an economic transmission channel.
""")

code(r"""
display(results["path_wald"])
band_summary = results["simultaneous_bands"].groupby(
    ["model", "outcome", "event", "block_length"], as_index=False).agg(
        Draws=("draws", "first"), Critical90=("critical90", "first"), Critical95=("critical95", "first"))
display(band_summary)
display(results["event_drop"][["outcome", "event", "horizon", "episode", "n_removed",
                                "baseline_beta", "beta", "sign_preserved"]])
""")

markdown(r"""
### Direct tests of heterogeneity and alternative conventional inference

An effect being individually significant while another is not does not establish that the two effects differ. `cross_category_equality.csv` therefore reports explicit within-joint-model restrictions, both at individual horizons and jointly across the predeclared horizon grid. Their covariance includes cross-category and, for path tests, cross-horizon dependence. These are tests of equality for the specified normalized log-index changes, not equal physical or historical events.

The Newey–West comparison is estimated separately: two price lags with a six-month Bartlett bandwidth, versus the augmented three-price-lag HC3 model on the same observations. Its coefficient and standard error are exported together. A further grid varies the conventional model bandwidth over 4, 12, 24, and 48 months. It never changes the baseline HC3 bands.
""")

code(r"""
display(results["cross_category_equality"])
display(results["inference_comparison"].query("horizon in [0,12,24,48]"))
display(results["hac_bandwidth"])
""")

markdown(r"""
### Demand and physical-supply channel evidence

The same three-treatment joint design is re-estimated with log world industrial production and global oil-production growth as outcomes. Each outcome receives three lags, while the other fundamentals, real Brent, and the geopolitical treatments receive two. A fall in world activity accompanying a diplomatic turning point would support a demand-side interpretation; a fall in oil-production growth would support a physical-supply interpretation. These are outcome responses under the same instruments, not a formal mediation decomposition, because the oil-price equation is not conditioned on an independently identified mediator.
""")

code(r"""
display(results["mechanism_iv"].loc[
    results["mechanism_iv"]["horizon"].isin([0, 6, 12, 24, 36, 48]),
    ["outcome", "event", "horizon", "n", "beta", "hc3_se", "hc3_p"]])
display(Image(filename=str(FIGURES / "figS01_mechanism_irfs.png")))
""")

markdown(r"""
### Simultaneous uncertainty and robustness interpretation

The simultaneous bands use fixed-design multipliers on **HC3 influence rows**, not a bootstrap that re-estimates models on resampled events. The baseline uses independent Rademacher multipliers for every month (`block_length=1`). Supplementary block lengths of six and twelve months allow multipliers to be shared within contiguous calendar blocks, with a random block origin in each draw. Missing observation scores are zero-filled on the complete calendar. Each curve is adjusted across its 49 horizons; the procedure does not provide family-wise coverage across all event types and both oil prices together.

The origin splits and event omissions are descriptive sensitivity exercises, not formal evidence that coefficients differ. The treatment and control transformations are always computed before sample restrictions, preserving the calendar meaning of each lag and lead.
""")

code(r"""
display(results["origin_splits"].loc[
    results["origin_splits"]["specification"].isin(["pre_2012", "pre_2019"]),
    ["outcome", "event", "horizon", "specification", "n", "beta", "hc3_se"]])
display(results["lag_sensitivity"][["outcome", "event", "horizon", "specification",
                                     "n", "matched_baseline_beta", "beta"]])
display(results["production_control"].loc[
    results["production_control"]["horizon"].isin([0, 12, 24, 48]),
    ["outcome", "event", "horizon", "n", "matched_baseline_beta", "beta", "hc3_se"]])
""")

markdown(r"""
### Why the log-curvature contrast needs a different interpretation

If the instrument is changed to $\Delta^2x_t=x_t-2x_{t-1}+x_{t-2}$ and both treatment lags are controls, residualization gives $M_W\Delta^2x_t=M_Wx_t$. The resulting IV coefficient is exactly the controlled OLS coefficient on the matched sample. This is an algebraic feature of that **alternative**, not a validation exercise for the baseline raw-curvature instrument. It is reported transparently as an OLS-equivalent contrast.

For instrument concentration, the relevant controlled first-stage contribution is $\widetilde Z_t\widetilde x_t$. Large raw values alone do not establish which observations dominate identification after controls. Absolute-contribution shares and the largest dates are reported for both versions to make this difference inspectable.
""")

code(r"""
display(results["log_curvature"][["outcome", "event", "horizon", "matched_baseline_beta", "beta"]])
display(results["leverage"].loc[
    (results["leverage"]["horizon"] == 0) & (results["leverage"]["contribution_type"] == "partialled"),
    ["outcome", "event", "rank", "date", "share_absolute_total", "top5_share_absolute_total"]])
""")

markdown(r"""
## 13 Final consistency checks and the output map

The final checks verify finite baseline estimates and standard errors, complete horizon grids, positive residual sample sizes, the fixed anticipation timing, and the existence of every requested figure format. The source reconstruction check at the start protects the input merge, and the explicit FWL calculation checks the estimator's cross-product implementation.

The independent test suite below checks numerical estimation against separate implementations, input reconstruction, and the exact timing of the design. Reproducibility means matching a frozen-data analysis; it does not establish the empirical identifying assumptions. The notebook deliberately keeps those two questions separate.
""")

code(r"""
assert np.isfinite(results["baseline_iv"][["beta", "hc3_se", "hc1_se"]].to_numpy()).all()
assert results["baseline_iv"]["primary_covariance"].eq("HC3").all()
assert not any("hac" in name.lower() for name in results["baseline_iv"].columns)
for _, group in results["baseline_iv"].groupby(["model", "outcome", "event"]):
    assert group["horizon"].sort_values().tolist() == list(range(49))
    assert (group["n"] > 20).all()
assert results["anticipation"]["horizon"].isin([0, 1]).all()
assert results["anticipation"]["instrument_lead"].eq(2).all()
for stem in captions:
    for extension in ("png", "pdf", "svg"):
        assert (FIGURES / f"{stem}.{extension}").is_file()
    from PIL import Image as PILImage
    with PILImage.open(FIGURES / f"{stem}.png") as png:
        png.verify()  # Check all PNG chunk checksums, not only the file header.
    with PILImage.open(FIGURES / f"{stem}.png") as png:
        png.load()    # Check complete pixel decompression as well.

outputs = pd.DataFrame({"File": [str(path.relative_to(ROOT)) for directory in (RESULTS, FIGURES)
                                   for path in sorted(directory.glob("*"))]})
display(outputs)
print("All notebook assertions passed. Every table and figure was generated in this run.")
""")

code(r"""
import subprocess
validation = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
                            cwd=ROOT, text=True, capture_output=True)
print(validation.stdout)
print(validation.stderr)
validation.check_returncode()
""")

markdown(r"""
## Methodological references and package files

The manuscript provides the full literature review and source bibliography. The local-projection framework follows Jordà (2005), *American Economic Review*, 95(1), 161–182. Lag augmentation is motivated by Montiel Olea and Plagborg-Møller (2021), *Econometrica*, 89(4), 1789–1823; their results should not be read as an unrestricted guarantee for every IV specification. The distinction between equation relevance and conditional strength in multiple-endogenous-variable models follows Sanderson and Windmeijer (2016), *Journal of Econometrics*, 190(2), 212–221. Caldara and Iacoviello (2022), *American Economic Review*, 112(4), 1194–1225, provide the news-based GPR benchmark; the present event categories use the AI-GPR source identified in the manuscript's data appendix.

- [Analysis source](code/analysis.py) and [figure source](code/figures.py).
- [Frozen numerical data](data/monthly_data.csv) and [source snapshots](data/source_csv/).
- [Baseline IV estimates](results/baseline_iv.csv), [first-stage diagnostics](results/first_stage.csv), and [joint identification diagnostics](results/joint_identification.csv).
- [Sanderson–Windmeijer conditional first stages](results/sw_conditional_first_stage.csv).
- [World-activity and oil-production responses](results/mechanism_iv.csv) and [their supplementary figure](figures/figS01_mechanism_irfs.png).
- [Anticipation coefficients](results/anticipation.csv) and [anticipation joint tests](results/anticipation_joint_tests.csv).
- [Summary statistics](results/summary_statistics.csv), [data endpoints](results/endpoints.csv), and [run metadata](results/run_metadata.json).
- [Separate Brent figure](figures/fig03_brent_separate_irfs.png), [separate WTI figure](figures/fig03_wti_separate_irfs.png), [joint figure](figures/fig04_joint_irfs.png), and [external figure captions](figures/figure_captions.json).

All substantive inference should be read together with the manuscript's identifying assumptions, sample definitions, and robustness discussion.
""")

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12", "mimetype": "text/x-python",
                      "codemirror_mode": {"name": "ipython", "version": 3},
                      "pygments_lexer": "ipython3", "file_extension": ".py"},
    "authors": [{"name": "Jamel Saadaoui"}],
}, "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(notebook["cells"]):
    cell["id"] = f"replication-{i:03d}"
path = ROOT / "Geopolitical_Turning_Points_Replication.ipynb"
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {path.name}: {len(cells)} cells")
