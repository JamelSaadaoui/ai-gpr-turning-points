# Econometric and computational validation

## Implemented method

The baseline contains three oil-price lags and two lags of world industrial production, oil-production growth and every geopolitical treatment. It uses HC3, with no HAC correction and no extra HC1 multiplier. Its leverage is from the full projected-regressor design, including controls and the constant; in these exactly identified models, that space equals the full instrument space. The structural 2SLS residual is used, not an OLS residual from fitted treatments. This follows the second-stage leverage convention in the official [ivreg diagnostics documentation](https://zeileis.github.io/ivreg/articles/Diagnostics-for-2SLS-Regression.html).

The economic exclusion assumption and inference conditions are distinct. Expectations, precautionary demand and risk compensation may transmit the geopolitical state. HC3 additionally maintains the relevant score-dependence conditions. Neither curvature nor an extra lag proves those conditions for an arbitrary instrument.

## Independent checks

Coefficients and HC1 covariance are compared with independently assembled samples and QR-based IV calculations. HC3 checks use a separately constructed **full-system 2SLS** estimator that projects the complete regressor matrix on the complete instrument matrix. Across both prices, all nine scalar models and the joint model at horizons 0, 12, 24 and 48 (80 fits), this verifies coefficients, full HC3 covariance, influence rows and leverage.

Additional checks cover full-design OLS HC3 for anticipation, the OLS-HC3 limit of the log-curvature contrast, all-variable and mechanism-outcome lag augmentation, calendar gaps and source hashes, null-specific HC3 Anderson–Rubin inversion, Sanderson–Windmeijer conditional first stages, profiled joint Anderson–Rubin projections, full-covariance category and path tests, multiplier reproducibility, explicit elapsed-month Bartlett calculations, workbook formulas and data, notebook syntax and outputs, and complete PNG/PDF/SVG integrity. The clean-run report records the final test count and result-table reproduction.

The conventional comparison uses two price lags and Newey–West with six-month bandwidth on matched observations. Both its coefficient and its standard error are reported. The bandwidth grid applies to that conventional model. The all-variable three-lag alternative uses HC3. Baseline figures never use HAC errors.

## Interpretation boundaries

- First-stage HC3 Wald statistics measure relevance; joint statistics are not Sanderson–Windmeijer conditional-strength tests. No universal F-greater-than-ten rule is asserted.
- Scalar HC3 Anderson–Rubin confidence sets belong to the separate models, not to individual joint-model coefficients. Exactly identified models have no overidentification test.
- Anticipation uses fixed `Z[t+2]` for outcomes at `t` and `t+1`. Because the regressor includes already realized index values, it is a conditional lead-curvature association diagnostic. All rejections remain reported.
- Independent-month multipliers are baseline path inference under the stacked-score condition. Blocks6/12 are supplementary simultaneous-band dependence sensitivities, not baseline pointwise HAC. Bands are curve-specific fixed-design approximations and are not weak-IV-robust sets.
- Episode omissions concern origin windows. Origin splits are descriptive because forecast-error windows can overlap. Post-2019 horizon48 has only43 origins.
- Current production can be a mediator; conditioning on it is not identification of an oil-supply shock.
- Continuous-index IV responses are local to selected variation, without an automatic binary-complier LATE interpretation.

## Reproduction boundary

All standard Python notebook cells are executed sequentially in a fresh process through the included runner, which saves native outputs. The frozen complete provider histories are the reproduction boundary; `DATA_SOURCES.md` records the current vintages, transformations, checksums, and unavailable observations. The software tests establish implementation consistency, not identifying restrictions or exact confidence-set coverage.
