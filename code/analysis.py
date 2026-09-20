"""Transparent lag-augmented local-projection IV estimates and diagnostics.

All dates are observation-origin months. A horizon h regression uses the oil
price in month t+h, not a cumulative change. Primary inference uses HC3 with
the documented 2+1 price-lag augmentation. HAC belongs to separate comparisons.
No downloads, interpolation, plotting, or hidden global data are required.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import NormalDist
from typing import Sequence

import numpy as np
import pandas as pd


_NORMAL = NormalDist()


def normal_sf(value: float) -> float:
    """Standard-normal survival probability using only the Python standard library."""
    return 0.5 * math.erfc(value / math.sqrt(2.0))


def normal_ppf(probability: float) -> float:
    return _NORMAL.inv_cdf(probability)


def chi2_sf(value: float, degrees: int) -> float:
    """Chi-square survival function via the regularized upper incomplete gamma.

    This stable series/continued-fraction implementation removes SciPy as a
    hidden runtime requirement. It follows the standard Numerical Recipes
    split at x=a+1 and is accurate well beyond the precision displayed here.
    """
    if degrees <= 0:
        raise ValueError("degrees must be positive")
    if value <= 0:
        return 1.0
    a, x = degrees / 2.0, value / 2.0
    eps, tiny, limit = 3e-14, 1e-300, 10000
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(limit):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) <= abs(total) * eps:
                lower = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
                return min(1.0, max(0.0, 1.0 - lower))
        raise ArithmeticError("incomplete-gamma series did not converge")
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    fraction = d
    for i in range(1, limit + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= eps:
            upper = fraction * math.exp(-x + a * math.log(x) - math.lgamma(a))
            return min(1.0, max(0.0, upper))
    raise ArithmeticError("incomplete-gamma fraction did not converge")


def chi2_ppf_df1(probability: float) -> float:
    """Chi-square quantile for one degree of freedom."""
    return normal_ppf((1.0 + probability) / 2.0) ** 2


def chi2_ppf(probability: float, degrees: int) -> float:
    """Chi-square quantile by monotone bisection of ``chi2_sf``."""
    if not 0 < probability < 1:
        raise ValueError("probability must lie strictly between zero and one")
    if degrees == 1:
        return chi2_ppf_df1(probability)
    target = 1.0 - probability
    lower, upper = 0.0, max(1.0, float(degrees))
    while chi2_sf(upper, degrees) > target:
        upper *= 2.0
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if chi2_sf(middle, degrees) > target:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


EVENTS = (
    "gpr_ai", "military_conflict", "diplomatic_tension", "terrorism",
    "civil_war", "nuclear_threat", "coup", "sanctions", "other",
)
PRIMARY = ("military_conflict", "diplomatic_tension", "nuclear_threat")
LABELS = {
    "gpr_ai": "Aggregate AI-GPR", "military_conflict": "Military conflict",
    "diplomatic_tension": "Diplomatic tension", "terrorism": "Terrorism",
    "civil_war": "Civil war", "nuclear_threat": "Nuclear threat",
    "coup": "Coup", "sanctions": "Sanctions", "other": "Other events",
    "lwti": "Log real WTI price", "lbrent": "Log real Brent price",
    "lwip": "Log world industrial production",
    "lgop": "Log global oil production",
    "production_growth": "Global oil production growth",
}
OUTCOMES = ("lwti", "lbrent")
MECHANISM_OUTCOMES = ("lwip", "production_growth")
HORIZONS = tuple(range(49))
WALD_HORIZONS = (0, 6, 12, 24, 36, 48)
CHECK_HORIZONS = (0, 12, 24, 48)
EPISODES = {
    "1990-08-01": "Iraq invades Kuwait",
    "1991-01-01": "Gulf War air campaign",
    "1998-05-01": "Indian and Pakistani nuclear tests",
    "2001-09-01": "September 11 attacks",
    "2003-03-01": "Invasion of Iraq",
    "2006-10-01": "North Korean nuclear test",
    "2011-02-01": "Libyan uprising",
    "2013-07-01": "Egyptian military takeover",
    "2015-11-01": "Paris attacks",
    "2017-07-01": "North Korean missile escalation",
    "2022-02-01": "Russian invasion of Ukraine",
    "2023-10-01": "Israel-Hamas war outbreak",
}


@dataclass
class Design:
    """Complete observations for one LP, with calendar positions retained."""

    y: np.ndarray
    x: np.ndarray
    z: np.ndarray
    controls: np.ndarray
    positions: np.ndarray
    dates: pd.DatetimeIndex
    calendar: pd.DatetimeIndex
    outcome: str
    events: tuple[str, ...]
    horizon: int
    control_names: tuple[str, ...]


@dataclass
class IVFit:
    """Coefficient estimates, covariance matrices, and calendar-aligned scores."""

    beta: np.ndarray
    hc1_cov: np.ndarray
    hac_cov: np.ndarray
    influence: np.ndarray
    residuals: np.ndarray
    x_residual: np.ndarray
    z_residual: np.ndarray
    y_residual: np.ndarray
    controls_rank: int
    hac_lag: int
    n: int
    design: Design
    hc3_cov: np.ndarray
    hc3_influence: np.ndarray
    leverage: np.ndarray


def prepare_data(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Validate input and build all lags on an unbroken monthly calendar.

    Missing values are retained. Calendar months absent from the supplied data
    become missing rows. Input price/production columns are already natural
    logarithms; the frozen input contains the documented log-linear CPI fill
    used to construct real Brent and WTI in October 2025.
    """
    data = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    required = ["date", "lwti", "lbrent", "lwip", "lgop", *EVENTS]
    absent = sorted(set(required).difference(data.columns))
    if absent:
        raise ValueError(f"Required input columns are absent: {absent}")
    data = data[required].copy()
    data["date"] = pd.to_datetime(data["date"]).dt.to_period("M").dt.to_timestamp()
    if data["date"].duplicated().any():
        raise ValueError("Input has duplicate monthly dates.")
    data = data.sort_values("date").set_index("date")
    calendar = pd.date_range(data.index.min(), data.index.max(), freq="MS")
    data = data.reindex(calendar).rename_axis("date").reset_index()
    for name in required[1:]:
        data[name] = pd.to_numeric(data[name], errors="raise")
        if np.isinf(data[name].to_numpy()).any():
            raise ValueError(f"Infinite input values in {name}.")
    for event in EVENTS:
        if (data[event].dropna() < 0).any():
            raise ValueError(f"Event index {event} contains negative levels.")
        data[f"x_{event}"] = np.log1p(data[event])
        data[f"z_{event}"] = data[event].diff().diff()
        data[f"zlog_{event}"] = data[f"x_{event}"].diff().diff()
    data["production_growth"] = data["lgop"].diff()
    lagged = ["lwti", "lbrent", "lwip", "production_growth"]
    lagged += [f"x_{event}" for event in EVENTS]
    additions = {}
    for name in lagged:
        max_lag = 4 if name in OUTCOMES else 3
        for lag in range(1, max_lag + 1):
            additions[f"L{lag}_{name}"] = data[name].shift(lag)
    return pd.concat([data, pd.DataFrame(additions)], axis=1)


def control_names(outcome: str, events: Sequence[str], price_lags: int = 3,
                  current_production: bool = False, other_lags: int = 2) -> list[str]:
    """Baseline: 2 activity, 2 production-growth, 3 price, 2 treatment lags."""
    names = [f"L{lag}_{name}" for name in ("lwip", "production_growth")
             for lag in range(1, other_lags + 1)]
    names += [f"L{lag}_{outcome}" for lag in range(1, price_lags + 1)]
    for event in events:
        names += [f"L{lag}_x_{event}" for lag in range(1, other_lags + 1)]
    if current_production:
        names += ["production_growth"]
    return names


def make_design(data: pd.DataFrame, outcome: str, events: Sequence[str], horizon: int,
                price_lags: int = 3, instrument: str = "raw",
                current_production: bool = False,
                origin_start: str | None = None, origin_end: str | None = None,
                omit_event: str | None = None,
                required_positions: np.ndarray | None = None,
                other_lags: int = 2) -> Design:
    """Select only variables actually used by the specified regression.

    Lags and leads are constructed before any sample exclusions. In particular,
    an event-drop exercise does not spuriously join previously separated months.
    ``required_positions`` enables explicitly matched-sample comparisons.
    """
    events = tuple(events)
    if outcome not in OUTCOMES or not set(events).issubset(EVENTS):
        raise ValueError("Unknown outcome or event.")
    if instrument not in {"raw", "log"}:
        raise ValueError("instrument must be 'raw' or 'log'.")
    names = control_names(outcome, events, price_lags, current_production, other_lags)
    z_prefix = "z_" if instrument == "raw" else "zlog_"
    columns = [*[f"x_{event}" for event in events],
               *[f"{z_prefix}{event}" for event in events], *names]
    frame = data[columns].copy()
    frame["y"] = data[outcome].shift(-horizon)
    valid = frame.notna().all(axis=1)
    if origin_start is not None:
        valid &= data["date"] >= pd.Timestamp(origin_start)
    if origin_end is not None:
        valid &= data["date"] <= pd.Timestamp(origin_end)
    if omit_event is not None:
        center = pd.Timestamp(omit_event)
        low = center - pd.DateOffset(months=1)
        high = center + pd.DateOffset(months=1)
        valid &= ~data["date"].between(low, high)
    if required_positions is not None:
        valid &= data.index.isin(required_positions)
    selected = frame.loc[valid]
    return Design(
        y=selected["y"].to_numpy(float),
        x=selected[[f"x_{event}" for event in events]].to_numpy(float),
        z=selected[[f"{z_prefix}{event}" for event in events]].to_numpy(float),
        controls=np.column_stack([np.ones(len(selected)), selected[names].to_numpy(float)]),
        positions=np.flatnonzero(valid.to_numpy()),
        dates=pd.DatetimeIndex(data.loc[valid, "date"]),
        calendar=pd.DatetimeIndex(data["date"]), outcome=outcome, events=events,
        horizon=horizon, control_names=("constant", *names),
    )


def make_mechanism_design(data: pd.DataFrame, outcome: str, events: Sequence[str], horizon: int,
                          price: str = "lbrent") -> Design:
    """Lag-augmented IV-LP for world activity or oil-production growth.

    The dependent variable receives three lags (two baseline lags plus one lag
    augmentation).  The other oil-market fundamentals and each geopolitical
    treatment receive two lags.  The current geopolitical treatments remain
    endogenous and are instrumented by their raw-index curvatures.
    """
    if outcome not in MECHANISM_OUTCOMES:
        raise ValueError("Unknown mechanism outcome.")
    if price not in OUTCOMES:
        raise ValueError("price must be lbrent or lwti.")
    events = tuple(events)
    names = []
    for variable in ("lwip", "production_growth", price):
        maximum = 3 if variable == outcome else 2
        names.extend(f"L{lag}_{variable}" for lag in range(1, maximum + 1))
    for event in events:
        names.extend(f"L{lag}_x_{event}" for lag in (1, 2))
    frame = data[[*[f"x_{event}" for event in events],
                  *[f"z_{event}" for event in events], *names]].copy()
    frame["y"] = data[outcome].shift(-horizon)
    valid = frame.notna().all(axis=1)
    selected = frame.loc[valid]
    return Design(
        y=selected["y"].to_numpy(float),
        x=selected[[f"x_{event}" for event in events]].to_numpy(float),
        z=selected[[f"z_{event}" for event in events]].to_numpy(float),
        controls=np.column_stack([np.ones(len(selected)), selected[names].to_numpy(float)]),
        positions=np.flatnonzero(valid.to_numpy()),
        dates=pd.DatetimeIndex(data.loc[valid, "date"]),
        calendar=pd.DatetimeIndex(data["date"]), outcome=outcome, events=events,
        horizon=horizon, control_names=("constant", *names),
    )


def residualize(controls: np.ndarray, *arrays: np.ndarray) -> tuple[list[np.ndarray], int]:
    """FWL residualization by SVD on standardized controls, never normal equations."""
    scaled = controls.copy()
    if scaled.shape[1] > 1:
        center = scaled[:, 1:].mean(axis=0)
        scale = scaled[:, 1:].std(axis=0)
        scale[scale == 0] = 1.0
        scaled[:, 1:] = (scaled[:, 1:] - center) / scale
    u, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    threshold = np.finfo(float).eps * max(scaled.shape) * singular[0]
    rank = int(np.sum(singular > threshold))
    q = u[:, :rank]
    return [array - q @ (q.T @ array) for array in arrays], rank


def hac_covariance(influence: np.ndarray, lag: int) -> np.ndarray:
    """Bartlett HAC sum from influence rows on an unbroken monthly calendar.

    Rows for unavailable observations must be zero, not removed. Influence rows
    already include the finite-sample HC1 factor, and sum rather than mean scaling.
    """
    if influence.ndim == 1:
        influence = influence[:, None]
    covariance = influence.T @ influence
    for distance in range(1, min(lag, len(influence) - 1) + 1):
        weight = 1.0 - distance / (lag + 1.0)
        cross = influence[distance:].T @ influence[:-distance]
        covariance += weight * (cross + cross.T)
    return (covariance + covariance.T) / 2.0


def bandwidth(horizon: int) -> int:
    """Legacy capped-bandwidth HAC sensitivity, retained for comparability."""
    return max(4, min(12, horizon + 1))


def projection_leverage(controls: np.ndarray, residual_instruments: np.ndarray) -> np.ndarray:
    """Full projected-regressor leverage, including controls and intercept.

    In the exactly identified case, [C, P_[C,Z] X] spans [C,Z]. Orthogonal
    controls and residual instruments therefore give the second-stage hat
    diagonal without an ill-conditioned full-design inverse. This is the
    stage-2 leverage convention for IV HC3, not OLS on fitted outcomes.
    """
    scaled = controls.copy()
    if scaled.shape[1] > 1:
        scale = scaled[:, 1:].std(axis=0)
        scale[scale == 0] = 1
        scaled[:, 1:] = (scaled[:, 1:] - scaled[:, 1:].mean(axis=0)) / scale
    qc, _ = np.linalg.qr(scaled, mode="reduced")
    qz, _ = np.linalg.qr(residual_instruments, mode="reduced")
    leverage = np.sum(qc * qc, axis=1) + np.sum(qz * qz, axis=1)
    if np.any(leverage >= 1 - 1e-10):
        raise ValueError("Unit leverage prevents finite HC3 inference.")
    return leverage


def fit_iv(design: Design, hac_lag: int | None = None) -> IVFit:
    """Exactly identified IV after stable FWL projection of all included controls.

    Residual instruments are scaled to unit RMS. Solving their cross-moment with
    residual treatments avoids squaring the condition number through X'PzX.
    Covariance uses structural second-stage residuals, not first-stage residuals.
    """
    residuals, controls_rank = residualize(design.controls, design.y, design.x, design.z)
    y_residual, x_residual, z_residual = residuals
    n, k = x_residual.shape
    if n <= controls_rank + k:
        raise ValueError("Insufficient residual degrees of freedom.")
    z_scale = np.sqrt(np.mean(z_residual ** 2, axis=0))
    if np.any(z_scale < 1e-14):
        raise ValueError("Instrument has no variation after the controls.")
    z_standard = z_residual / z_scale
    cross = z_standard.T @ x_residual
    if np.linalg.matrix_rank(cross) != k:
        raise ValueError("Residualized first-stage cross-moment is rank deficient.")
    beta = np.linalg.solve(cross, z_standard.T @ y_residual)
    structural_residuals = y_residual - x_residual @ beta
    # The columns of mapping turn each instrument-score row into beta influence.
    mapping = np.linalg.solve(cross, np.eye(k)).T
    observed_influence = (z_standard * structural_residuals[:, None]) @ mapping
    leverage = projection_leverage(design.controls, z_residual)
    hc3_influence = np.zeros((len(design.calendar), k))
    hc3_influence[design.positions] = observed_influence / (1 - leverage[:, None])
    observed_influence *= np.sqrt(n / (n - controls_rank - k))
    influence = np.zeros((len(design.calendar), k))
    influence[design.positions] = observed_influence
    lag = bandwidth(design.horizon) if hac_lag is None else hac_lag
    return IVFit(
        beta=beta, hc1_cov=influence.T @ influence,
        hac_cov=hac_covariance(influence, lag), influence=influence,
        residuals=structural_residuals, x_residual=x_residual,
        z_residual=z_residual, y_residual=y_residual,
        controls_rank=controls_rank, hac_lag=lag, n=n, design=design,
        hc3_cov=hc3_influence.T @ hc3_influence,
        hc3_influence=hc3_influence, leverage=leverage,
    )


def _wald(beta: np.ndarray, covariance: np.ndarray) -> tuple[float, int, float]:
    """Rank-aware asymptotic chi-square Wald statistic."""
    rank = int(np.linalg.matrix_rank(covariance))
    if rank == 0:
        return np.nan, 0, np.nan
    statistic = float(beta @ np.linalg.pinv(covariance, hermitian=True) @ beta)
    return statistic, rank, chi2_sf(statistic, rank)


def _ols_on_residuals(y: np.ndarray, z: np.ndarray, controls_rank: int,
                      positions: np.ndarray, calendar_length: int,
                      lag: int, controls: np.ndarray) -> dict:
    """Small residualized OLS problem for first stages and reduced forms."""
    n, q = z.shape
    # QR yields a stable inverse map without forming a full-design Gram matrix.
    basis, triangular = np.linalg.qr(z, mode="reduced")
    if np.linalg.matrix_rank(triangular) != q:
        raise ValueError("Rank-deficient excluded instruments.")
    beta = np.linalg.solve(triangular, basis.T @ y)
    residual = y - z @ beta
    derivative = np.linalg.solve(triangular, basis.T).T
    scores = derivative * residual[:, None] * np.sqrt(n / (n - controls_rank - q))
    influence = np.zeros((calendar_length, q))
    influence[positions] = scores
    leverage = projection_leverage(controls, z)
    hc3_influence = np.zeros((calendar_length, q))
    hc3_influence[positions] = derivative * (residual / (1 - leverage))[:, None]
    return {"beta": beta, "residual": residual, "influence": influence,
            "hc3_influence": hc3_influence, "hc3_cov": hc3_influence.T @ hc3_influence,
            "hc1_cov": influence.T @ influence,
            "hac_cov": hac_covariance(influence, lag)}


def first_stage_diagnostics(fit: IVFit) -> tuple[list[dict], dict]:
    """Relevance tests and geometric rank checks, without bogus conditional F.

    In joint models the F statistic tests all excluded instruments in each
    treatment equation. It is NOT a Sanderson-Windmeijer conditional-strength
    statistic and must never be interpreted against conditional-F thresholds.
    """
    design = fit.design
    rows = []
    for column, event in enumerate(design.events):
        x = fit.x_residual[:, column]
        regression = _ols_on_residuals(
            x, fit.z_residual, fit.controls_rank, design.positions,
            len(design.calendar), fit.hac_lag, design.controls,
        )
        statistic, df, pvalue = _wald(regression["beta"], regression["hc1_cov"])
        stat3, df3, p3 = _wald(regression["beta"], regression["hc3_cov"])
        stage_hac4 = hac_covariance(regression["influence"], 4)
        statistic_hac4, df_hac4, pvalue_hac4 = _wald(regression["beta"], stage_hac4)
        partial_r2 = 1 - np.sum(regression["residual"] ** 2) / np.sum(x ** 2)
        rows.append({"event": event, "partial_r2": partial_r2,
                     "excluded_instrument_wald_F_hc3": stat3 / df3,
                     "excluded_instrument_wald_chi2_hc3": stat3,
                     "relevance_p_value_hc3": p3,
                     "excluded_instrument_wald_chi2": statistic,
                     "excluded_instrument_wald_F": statistic / df if df else np.nan,
                     "excluded_instrument_df": df, "relevance_p_value": pvalue,
                     "excluded_instrument_wald_chi2_hac4": statistic_hac4,
                     "excluded_instrument_wald_F_hac4": statistic_hac4 / df_hac4 if df_hac4 else np.nan,
                     "excluded_instrument_df_hac4": df_hac4, "relevance_p_value_hac4": pvalue_hac4,
                     "diagnostic_type": "single-instrument relevance" if len(design.events) == 1
                     else "joint excluded-instrument relevance; not conditional strength"})
    qx, _ = np.linalg.qr(fit.x_residual, mode="reduced")
    qz, _ = np.linalg.qr(fit.z_residual, mode="reduced")
    correlations = np.linalg.svd(qz.T @ qx, compute_uv=False)
    reduced_form = _ols_on_residuals(
        fit.y_residual, fit.z_residual, fit.controls_rank, design.positions,
        len(design.calendar), fit.hac_lag, design.controls,
    )
    ar_stat, ar_df, ar_p = _wald(reduced_form["beta"], reduced_form["hac_cov"])
    horizon_hac_lag = max(4, design.horizon + 1)
    primary_ar_stat, primary_ar_df, primary_ar_p = _wald(
        reduced_form["beta"], hac_covariance(reduced_form["influence"], horizon_hac_lag)
    )
    geometry = {"minimum_canonical_correlation": float(correlations.min()),
                "ar_joint_zero_statistic_hc3": _wald(reduced_form["beta"], reduced_form["hc3_cov"])[0],
                "ar_joint_zero_p_value_hc3": _wald(reduced_form["beta"], reduced_form["hc3_cov"])[2],
                "maximum_canonical_correlation": float(correlations.max()),
                "canonical_correlation_ratio": float(correlations.max() / correlations.min()),
                "first_stage_rank": int(np.linalg.matrix_rank(fit.z_residual.T @ fit.x_residual)),
                "ar_joint_zero_statistic": ar_stat, "ar_df": ar_df,
                "ar_joint_zero_p_value_hac": ar_p,
                "q_hac_horizon": horizon_hac_lag,
                "ar_joint_zero_statistic_hac_horizon": primary_ar_stat,
                "ar_df_hac_horizon": primary_ar_df,
                "ar_joint_zero_p_value_hac_horizon": primary_ar_p}
    return rows, geometry


def sanderson_windmeijer_conditional_f(fit: IVFit) -> list[dict]:
    """Classical Sanderson--Windmeijer conditional first-stage F statistics.

    For each endogenous regressor, the other endogenous regressors are first
    treated as endogenous in a conditional IV regression using the complete
    excluded-instrument set.  Its structural residual is then regressed on all
    excluded instruments.  The raw Wald F is corrected for the ``k-1`` fitted
    endogenous directions exactly as in Sanderson and Windmeijer (2016):
    ``F_SW = F_raw*q/(q-(k-1))``.  The statistic is homoskedastic by
    construction and is therefore reported separately from the HC3
    rank-relevance diagnostic.
    """
    x, z = fit.x_residual, fit.z_residual
    n, q = z.shape
    k = x.shape[1]
    if k < 2 or q < k:
        raise ValueError("Conditional F requires at least two endogenous regressors and q >= k.")
    qz, _ = np.linalg.qr(z, mode="reduced")
    conditional_df = q - (k - 1)
    residual_df = n - fit.controls_rank - q
    rows = []
    for column, event in enumerate(fit.design.events):
        other = np.delete(x, column, axis=1)
        fitted_other = qz @ (qz.T @ other)
        delta = np.linalg.lstsq(fitted_other, x[:, column], rcond=None)[0]
        conditional_residual = x[:, column] - other @ delta
        instrument_slope = np.linalg.lstsq(z, conditional_residual, rcond=None)[0]
        unrestricted_residual = conditional_residual - z @ instrument_slope
        restricted_ssr = float(conditional_residual @ conditional_residual)
        unrestricted_ssr = float(unrestricted_residual @ unrestricted_residual)
        raw_f = ((restricted_ssr - unrestricted_ssr) / q) / (unrestricted_ssr / residual_df)
        corrected_f = raw_f * q / conditional_df
        rows.append({
            "event": event,
            "sw_conditional_F": corrected_f,
            "conditional_df": conditional_df,
            "residual_df": residual_df,
            "raw_wald_F_all_instruments": raw_f,
            "excluded_instruments": q,
            "endogenous_regressors": k,
            "covariance_assumption": "homoskedastic Sanderson-Windmeijer correction",
        })
    return rows


def _joint_ar_hc3_statistic(fit: IVFit, beta0: np.ndarray) -> float:
    """HC3 Anderson--Rubin statistic for a complete joint coefficient null."""
    beta0 = np.asarray(beta0, dtype=float)
    if beta0.shape != fit.beta.shape:
        raise ValueError("beta0 must have one value per endogenous regressor.")
    transformed = fit.y_residual - fit.x_residual @ beta0
    z = fit.z_residual
    inverse = np.linalg.inv(z.T @ z)
    reduced_form = inverse @ (z.T @ transformed)
    residual = transformed - z @ reduced_form
    derivative = (inverse @ z.T).T
    influence = derivative * (residual / (1.0 - fit.leverage))[:, None]
    covariance = influence.T @ influence
    return _wald(reduced_form, covariance)[0]


def _nelder_mead(function, start: np.ndarray, step: np.ndarray,
                 max_iterations: int = 120, tolerance: float = 1e-7) -> tuple[np.ndarray, float]:
    """Small dependency-free Nelder--Mead optimizer for AR nuisance profiling."""
    start = np.asarray(start, dtype=float)
    dimension = len(start)
    simplex = np.vstack([start] + [start + np.eye(dimension)[i] * step[i]
                                   for i in range(dimension)])
    values = np.array([function(point) for point in simplex])
    for _ in range(max_iterations):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        if (np.max(np.abs(simplex[1:] - simplex[0])) < tolerance and
                np.std(values) < tolerance):
            break
        centroid = simplex[:-1].mean(axis=0)
        reflected = centroid + (centroid - simplex[-1])
        reflected_value = function(reflected)
        if values[0] <= reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
            continue
        if reflected_value < values[0]:
            expanded = centroid + 2.0 * (reflected - centroid)
            expanded_value = function(expanded)
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
            continue
        contracted = centroid + 0.5 * (simplex[-1] - centroid)
        contracted_value = function(contracted)
        if contracted_value < values[-1]:
            simplex[-1], values[-1] = contracted, contracted_value
            continue
        simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
        values[1:] = [function(point) for point in simplex[1:]]
    best = int(np.argmin(values))
    return simplex[best], float(values[best])


def _profile_joint_ar(fit: IVFit, focal: int, value: float) -> float:
    """Minimize the joint HC3 AR statistic over the nuisance coefficients."""
    nuisance = np.array([index for index in range(len(fit.beta)) if index != focal])

    def objective(nuisance_values):
        candidate = np.empty_like(fit.beta)
        candidate[focal] = value
        candidate[nuisance] = nuisance_values
        return _joint_ar_hc3_statistic(fit, candidate)

    moment_target = fit.z_residual.T @ (
        fit.y_residual - fit.x_residual[:, focal] * value
    )
    moment_design = fit.z_residual.T @ fit.x_residual[:, nuisance]
    moment_start = np.linalg.lstsq(moment_design, moment_target, rcond=None)[0]
    scale = np.maximum(np.sqrt(np.diag(fit.hc3_cov))[nuisance], 0.02)
    candidates = [
        _nelder_mead(objective, fit.beta[nuisance], scale),
        _nelder_mead(objective, moment_start, scale),
    ]
    return min(candidates, key=lambda result: result[1])[1]


def joint_ar_projection_intervals(fit: IVFit, coverage: float = 0.95) -> list[dict]:
    """Project an HC3 joint Anderson--Rubin region onto each coefficient.

    The complete vector null is tested with ``q`` degrees of freedom.  For each
    reported coefficient value, the nuisance coefficients are profiled out.
    Projection of the accepted joint region produces conservative marginal
    intervals that remain valid under weak identification when the IV moments
    and the HC3 asymptotic approximation are valid.
    """
    if len(fit.beta) < 2:
        raise ValueError("Joint AR projection requires multiple endogenous regressors.")
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between zero and one.")
    degrees = fit.z_residual.shape[1]
    critical = chi2_ppf(coverage, degrees)
    rows = []
    for focal, event in enumerate(fit.design.events):
        center = float(fit.beta[focal])
        initial_step = max(2.0 * float(np.sqrt(fit.hc3_cov[focal, focal])), 0.05)
        boundaries = []
        for direction in (-1.0, 1.0):
            inside = center
            step = initial_step
            outside = center + direction * step
            for _ in range(30):
                if _profile_joint_ar(fit, focal, outside) > critical:
                    break
                inside = outside
                step *= 2.0
                outside = center + direction * step
            else:
                boundaries.append(-np.inf if direction < 0 else np.inf)
                continue
            lower, upper = sorted((inside, outside))
            for _ in range(25):
                middle = (lower + upper) / 2.0
                accepted = _profile_joint_ar(fit, focal, middle) <= critical
                if direction < 0:
                    if accepted:
                        upper = middle
                    else:
                        lower = middle
                else:
                    if accepted:
                        lower = middle
                    else:
                        upper = middle
            boundaries.append((lower + upper) / 2.0)
        rows.append({
            "event": event,
            "beta": center,
            "coverage": coverage,
            "lower": boundaries[0],
            "upper": boundaries[1],
            "contains_zero": bool(boundaries[0] <= 0.0 <= boundaries[1]),
            "joint_test_df": degrees,
            "joint_critical_value": critical,
            "covariance": "HC3",
            "method": "projection of profiled joint Anderson-Rubin region",
        })
    return rows


def coefficient_rows(fit: IVFit, model: str, specification: str = "baseline") -> list[dict]:
    """Tidy coefficients with pointwise 90/95 percent HC1 and HAC intervals."""
    rows = []
    horizon_hac_lag = max(4, fit.design.horizon + 1)
    horizon_hac_covariance = hac_covariance(fit.influence, horizon_hac_lag)
    for column, event in enumerate(fit.design.events):
        beta = float(fit.beta[column])
        row = {"model": model, "specification": specification,
               "outcome": fit.design.outcome, "event": event,
               "horizon": fit.design.horizon, "n": fit.n,
               "origin_start": fit.design.dates.min().date().isoformat(),
               "origin_end": fit.design.dates.max().date().isoformat(),
               "beta": beta, "hac_lag": fit.hac_lag}
        primary_se = float(np.sqrt(max(horizon_hac_covariance[column, column], 0)))
        row.update({"q_hac_horizon": horizon_hac_lag, "se_hac_horizon": primary_se,
                    "p_hac_horizon": float(2 * normal_sf(abs(beta / primary_se))) if primary_se else np.nan})
        for coverage in (90, 95):
            critical = normal_ppf((1 + coverage / 100) / 2)
            row[f"ci{coverage}_low_hac_horizon"] = beta - critical * primary_se
            row[f"ci{coverage}_high_hac_horizon"] = beta + critical * primary_se
        row["primary_covariance"] = "HC3"
        row["maximum_leverage"] = float(fit.leverage.max())
        for label, covariance in [("hc3", fit.hc3_cov), ("hc1", fit.hc1_cov), ("hac", fit.hac_cov)]:
            se = float(np.sqrt(max(covariance[column, column], 0)))
            row[f"{label}_se"] = se
            row[f"{label}_p"] = float(2 * normal_sf(abs(beta / se))) if se else np.nan
            for coverage in (90, 95):
                critical = normal_ppf((1 + coverage / 100) / 2)
                row[f"{label}_{coverage}_low"] = beta - critical * se
                row[f"{label}_{coverage}_high"] = beta + critical * se
        rows.append(row)
    return rows


def fit_anticipation(data: pd.DataFrame, outcome: str, events: Sequence[str],
                     horizon: int) -> dict:
    """Regress p[t+h] on Z[t+2] and origin-t predetermined controls, h=0 or 1.

    The instrument lead remains t+2 for BOTH tested horizons. No control is
    advanced into the future. These are pre-event tests, not future responses.
    """
    if horizon not in (0, 1):
        raise ValueError("Only h=0 and h=1 precede the fixed t+2 turning point.")
    events = tuple(events)
    names = control_names(outcome, events)
    frame = data[names].copy()
    frame["y"] = data[outcome].shift(-horizon)
    for event in events:
        frame[f"future_{event}"] = data[f"z_{event}"].shift(-2)
    valid = frame.notna().all(axis=1)
    frame = frame.loc[valid]
    controls = np.column_stack([np.ones(len(frame)), frame[names].to_numpy(float)])
    z = frame[[f"future_{event}" for event in events]].to_numpy(float)
    residuals, rank = residualize(controls, frame["y"].to_numpy(float), z)
    y_residual, z_residual = residuals
    regression = _ols_on_residuals(
        y_residual, z_residual, rank, np.flatnonzero(valid.to_numpy()), len(data), 4, controls,
    )
    regression.update({"n": len(frame), "horizon": horizon, "events": events,
                       "outcome": outcome, "calendar": pd.DatetimeIndex(data["date"]),
                       "origin_start": data.loc[valid, "date"].min().date().isoformat(),
                       "origin_end": data.loc[valid, "date"].max().date().isoformat()})
    return regression


def path_wald(fits: Sequence[IVFit], model: str) -> list[dict]:
    """Joint zero tests at six declared horizons using aligned influence vectors."""
    selected = [fit for fit in fits if fit.design.horizon in WALD_HORIZONS]
    rows = []
    for column, event in enumerate(selected[0].design.events):
        influence = np.column_stack([fit.influence[:, column] for fit in selected])
        beta = np.array([fit.beta[column] for fit in selected])
        covariance = hac_covariance(influence, 12)
        statistic, df, pvalue = _wald(beta, covariance)
        hc3_scores = np.column_stack([fit.hc3_influence[:, column] for fit in selected])
        primary_statistic, primary_df, primary_pvalue = _wald(beta, hc3_scores.T @ hc3_scores)
        rows.append({"model": model, "outcome": selected[0].design.outcome, "event": event,
                     "horizons": ",".join(map(str, WALD_HORIZONS)), "hac_lag": 12,
                     "wald_chi2": statistic, "df": df, "p_value": pvalue,
                     "primary_covariance": "HC3", "primary_hac_lag": 0, "primary_wald_chi2": primary_statistic,
                     "primary_df": primary_df, "primary_p_value": primary_pvalue})
    return rows


def simultaneous_bands(fits: Sequence[IVFit], model: str, draws: int,
                        block_length: int, seed: int) -> list[dict]:
    """Score block-multiplier bands, simultaneous over 49 horizons per curve.

    Rademacher weights are constant within contiguous calendar blocks. A random
    block origin is used in each draw. Zero-filled missing observation scores
    preserve the true monthly spacing. The method is a fixed-design influence
    approximation, not a re-estimation or event-resampling bootstrap.
    """
    rng = np.random.default_rng(seed)
    total_months = len(fits[0].design.calendar)
    weights = np.empty((draws, total_months))
    calendar_positions = np.arange(total_months)
    for draw in range(draws):
        offset = rng.integers(0, block_length)
        block_id = (calendar_positions + offset) // block_length
        multipliers = rng.choice([-1.0, 1.0], size=block_id.max() + 1)
        weights[draw] = multipliers[block_id]
    rows = []
    for column, event in enumerate(fits[0].design.events):
        influence = np.column_stack([fit.hc3_influence[:, column] for fit in fits])
        deviations = weights @ influence
        standard_error = deviations.std(axis=0, ddof=1)
        if np.any(standard_error <= 0):
            raise ValueError("Degenerate multiplier standard errors.")
        max_statistic = np.max(np.abs(deviations / standard_error), axis=1)
        critical90, critical95 = np.quantile(max_statistic, [0.90, 0.95])
        for index, fit in enumerate(fits):
            beta = float(fit.beta[column])
            rows.append({"model": model, "outcome": fit.design.outcome, "event": event,
                         "horizon": fit.design.horizon, "beta": beta,
                         "block_length": block_length, "draws": draws, "seed": seed,
                         "multiplier_se": standard_error[index],
                         "critical90": critical90, "critical95": critical95,
                         "sim90_low": beta - critical90 * standard_error[index],
                         "sim90_high": beta + critical90 * standard_error[index],
                         "sim95_low": beta - critical95 * standard_error[index],
                         "sim95_high": beta + critical95 * standard_error[index]})
    return rows


def leverage_diagnostics(fit: IVFit) -> list[dict]:
    """Largest absolute first-stage cross-products and concentration shares.

    Raw z*x contributions are descriptive and do not equal the controlled first
    stage. Residualized z*x contributions do equal that stage's cross-moment.
    Absolute shares avoid hiding offsetting positive/negative contributions.
    """
    if len(fit.design.events) != 1:
        raise ValueError("This concentration summary is defined for single-event IV.")
    rows = []
    for kind, contributions in [
        ("raw", fit.design.z[:, 0] * fit.design.x[:, 0]),
        ("partialled", fit.z_residual[:, 0] * fit.x_residual[:, 0]),
    ]:
        absolute = np.abs(contributions)
        order = np.argsort(-absolute)
        total = absolute.sum()
        for rank, position in enumerate(order[:5], start=1):
            rows.append({"outcome": fit.design.outcome, "event": fit.design.events[0],
                         "horizon": fit.design.horizon, "contribution_type": kind,
                         "rank": rank, "date": fit.design.dates[position].date().isoformat(),
                         "signed_contribution": contributions[position],
                         "share_absolute_total": absolute[position] / total,
                         "top5_share_absolute_total": absolute[order[:5]].sum() / total,
                         "top10_share_absolute_total": absolute[order[:10]].sum() / total})
    return rows


def quadratic_acceptance_set(a: float, b: float, c: float) -> dict:
    """Solve a*b0**2 + b*b0 + c <= 0 on the full real line.

    Reporting disconnected or unbounded sets is essential: an AR confidence
    set must not be forced into a conventional finite symmetric interval.
    """
    if not np.isfinite([a, b, c]).all():
        raise ValueError("Quadratic coefficients must be finite.")
    blank = {"lower_1": np.nan, "upper_1": np.nan, "lower_2": np.nan, "upper_2": np.nan}
    if a == 0:
        if b == 0:
            if c <= 0:
                return {**blank, "set_type": "whole_real_line", "lower_1": -np.inf, "upper_1": np.inf}
            return {**blank, "set_type": "empty_set"}
        boundary = -c / b
        if b > 0:
            return {**blank, "set_type": "left_ray", "lower_1": -np.inf, "upper_1": boundary}
        return {**blank, "set_type": "right_ray", "lower_1": boundary, "upper_1": np.inf}
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        if a < 0:
            return {**blank, "set_type": "whole_real_line", "lower_1": -np.inf, "upper_1": np.inf}
        return {**blank, "set_type": "empty_set"}
    root = np.sqrt(max(discriminant, 0))
    # This root formula avoids subtractive cancellation when |b| is large.
    q = -0.5 * (b + np.copysign(root, b))
    if q == 0:
        lower = upper = -b / (2 * a)
    else:
        lower, upper = sorted((q / a, c / q))
    if a > 0:
        return {**blank, "set_type": "bounded_interval", "lower_1": lower, "upper_1": upper}
    if root == 0:
        return {**blank, "set_type": "whole_real_line", "lower_1": -np.inf, "upper_1": np.inf}
    return {**blank, "set_type": "two_unbounded_rays", "lower_1": -np.inf,
            "upper_1": lower, "lower_2": upper, "upper_2": np.inf}


def anderson_rubin_confidence_set(fit: IVFit, coverage: float = 0.95) -> list[dict]:
    """Analytically invert the heteroskedasticity/HAC-robust scalar AR test.

    Regress residualized y and x on residualized z separately. Under beta=b0,
    the reduced-form coefficient is a-b0*pi and its influence is Iy-b0*Ix.
    Acceptance of (a-b0*pi)^2 / Var(a-b0*pi) <= chi2_1(coverage) is a quadratic
    inequality. This is weak-IV-robust asymptotic inference, not an exact test.
    """
    if len(fit.design.events) != 1:
        raise ValueError("Analytic inversion here supports scalar exactly identified models only.")
    if not 0 < coverage < 1:
        raise ValueError("Confidence-set coverage must be strictly between zero and one.")
    arguments = (fit.z_residual, fit.controls_rank, fit.design.positions,
                 len(fit.design.calendar), fit.hac_lag, fit.design.controls)
    reduced_form = _ols_on_residuals(fit.y_residual, *arguments)
    first_stage = _ols_on_residuals(fit.x_residual[:, 0], *arguments)
    a_hat = float(reduced_form["beta"][0])
    pi_hat = float(first_stage["beta"][0])
    scores = np.column_stack([reduced_form["influence"], first_stage["influence"]])
    hc3_scores = np.column_stack([reduced_form["hc3_influence"], first_stage["hc3_influence"]])
    critical = float(chi2_ppf_df1(coverage))
    rows = []
    for covariance_type, covariance in [
        ("HC3", hc3_scores.T @ hc3_scores),
        ("HC1", scores.T @ scores), ("HAC", hac_covariance(scores, fit.hac_lag)),
        ("HAC_horizon", hac_covariance(scores, max(4, fit.design.horizon + 1))),
    ]:
        a = pi_hat ** 2 - critical * covariance[1, 1]
        b = -2 * a_hat * pi_hat + 2 * critical * covariance[0, 1]
        c = a_hat ** 2 - critical * covariance[0, 0]
        confidence_set = quadratic_acceptance_set(a, b, c)
        rows.append({"model": "single", "outcome": fit.design.outcome,
                     "event": fit.design.events[0], "horizon": fit.design.horizon,
                     "category_role": "principal" if fit.design.events[0] in PRIMARY else "exploratory",
                     "n": fit.n, "beta_iv": fit.beta[0], "coverage": coverage,
                     "covariance": covariance_type,
                     "hac_lag": (max(4, fit.design.horizon + 1) if covariance_type == "HAC_horizon"
                                 else fit.hac_lag if covariance_type == "HAC" else 0),
                     "quadratic_a": a, "quadratic_b": b, "quadratic_c": c,
                     "reduced_form_coefficient": a_hat, "first_stage_coefficient": pi_hat,
                     "ar_zero_statistic": a_hat ** 2 / covariance[0, 0],
                     "ar_zero_p_value": chi2_sf(a_hat ** 2 / covariance[0, 0], 1),
                     **confidence_set})
    return rows


def cross_category_equality(fits: Sequence[IVFit]) -> list[dict]:
    """Formal equality of the three joint-model response coefficients.

    These compare equal changes in log(1+index), not equally severe latent
    geopolitical events. Pointwise and selected-path tests are distinguished.
    Pairwise results are reported without pretending to control every comparison.
    """
    if tuple(fits[0].design.events) != PRIMARY:
        raise ValueError("Equality tests require the declared three-event joint model.")
    contrasts = [
        ("all_three_equal", np.array([[1., -1., 0.], [0., 1., -1.]])),
        ("military_minus_diplomatic", np.array([[1., -1., 0.]])),
        ("military_minus_nuclear", np.array([[1., 0., -1.]])),
        ("diplomatic_minus_nuclear", np.array([[0., 1., -1.]])),
    ]
    rows = []
    for fit in fits:
        if fit.design.horizon not in CHECK_HORIZONS:
            continue
        for label, contrast in contrasts:
            difference = contrast @ fit.beta
            horizon_hac_lag = max(4, fit.design.horizon + 1)
            for covariance_type, covariance in [
                ("HC3", fit.hc3_cov),
                ("HC1", fit.hc1_cov), ("HAC", fit.hac_cov),
                ("HAC_horizon", hac_covariance(fit.influence, horizon_hac_lag)),
            ]:
                statistic, df, pvalue = _wald(difference, contrast @ covariance @ contrast.T)
                rows.append({"outcome": fit.design.outcome, "test": label,
                             "horizon": str(fit.design.horizon), "scope": "pointwise",
                             "covariance": covariance_type,
                             "hac_lag": (horizon_hac_lag if covariance_type == "HAC_horizon"
                                         else fit.hac_lag if covariance_type == "HAC" else 0),
                             "difference": float(difference[0]) if len(difference) == 1 else np.nan,
                             "wald_chi2": statistic, "df": df, "p_value": pvalue})
    selected = [fit for fit in fits if fit.design.horizon in WALD_HORIZONS]
    contrast = contrasts[0][1]
    differences = np.concatenate([contrast @ fit.beta for fit in selected])
    scores = np.column_stack([fit.influence @ contrast.T for fit in selected])
    hc3_scores = np.column_stack([fit.hc3_influence @ contrast.T for fit in selected])
    statistic, df, pvalue = _wald(differences, hc3_scores.T @ hc3_scores)
    rows.append({"outcome": selected[0].design.outcome, "test": "all_three_equal",
                 "horizon": ",".join(map(str, WALD_HORIZONS)), "scope": "selected_path",
                 "covariance": "HC3", "hac_lag": 0, "difference": np.nan,
                 "wald_chi2": statistic, "df": df, "p_value": pvalue})
    for lag in (12, 48):
        statistic, df, pvalue = _wald(differences, hac_covariance(scores, lag))
        rows.append({"outcome": selected[0].design.outcome, "test": "all_three_equal",
                     "horizon": ",".join(map(str, WALD_HORIZONS)), "scope": "selected_path",
                     "covariance": "HAC", "hac_lag": lag, "difference": np.nan,
                     "wald_chi2": statistic, "df": df, "p_value": pvalue})
    return rows


def hac_bandwidth_sensitivity(fit: IVFit, model: str) -> list[dict]:
    """Show uncertainty under short and long overlap-aware bandwidth choices."""
    rows = []
    for lag in (4, 12, 24, 48):
        covariance = hac_covariance(fit.influence, lag)
        for column, event in enumerate(fit.design.events):
            se = np.sqrt(max(covariance[column, column], 0))
            beta = fit.beta[column]
            rows.append({"model": model, "outcome": fit.design.outcome, "event": event,
                         "horizon": fit.design.horizon, "n": fit.n, "hac_lag": lag,
                         "beta": beta, "hac_se": se, "hac_p": 2 * normal_sf(abs(beta / se)),
                         "hac_95_low": beta - normal_ppf(.975) * se,
                         "hac_95_high": beta + normal_ppf(.975) * se})
    return rows


def summary_statistics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Available-observation statistics; regression samples are documented separately."""
    variables = ["lwti", "lbrent", "lwip", "lgop", "production_growth"]
    variables += [f"x_{event}" for event in EVENTS] + [f"z_{event}" for event in EVENTS]
    statistics = []
    endpoints = []
    for name in variables:
        valid = data[name].notna()
        series = data.loc[valid, name]
        if name.startswith("x_"):
            label = f"Log(1 + {LABELS[name[2:]]})"
        elif name.startswith("z_"):
            label = f"Second difference: {LABELS[name[2:]]}"
        else:
            label = LABELS[name]
        statistics.append({"variable": name, "label": label, "n": len(series),
                           "mean": series.mean(), "sd": series.std(ddof=1),
                           "min": series.min(), "median": series.median(), "max": series.max()})
        endpoints.append({"variable": name, "first_available": data.loc[valid, "date"].min(),
                          "last_available": data.loc[valid, "date"].max(),
                          "nonmissing": len(series), "missing": int((~valid).sum())})
    return pd.DataFrame(statistics), pd.DataFrame(endpoints)


def run_all(data: str | Path | pd.DataFrame, out_dir: str | Path,
            draws: int = 1999, seed: int = 20260919) -> dict[str, pd.DataFrame]:
    """Run the complete reproducible analysis and write all CSVs plus metadata.

    ``draws=1999`` is the publication configuration. Smaller values are useful
    only for software smoke tests; they must not replace the reported run.
    The return value maps CSV stems to data frames for notebook inspection.
    """
    prepared = prepare_data(data)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = {name: [] for name in (
        "baseline_iv", "first_stage", "joint_identification", "anticipation",
        "anticipation_joint_tests", "path_wald", "simultaneous_bands", "event_drop",
        "origin_splits", "lag_sensitivity", "log_curvature", "production_control", "leverage",
        "cross_category_equality", "ar_confidence_sets", "hac_bandwidth", "inference_comparison",
        "sw_conditional_first_stage", "joint_ar_projection_intervals",
        "mechanism_iv", "mechanism_simultaneous_bands",
    )}
    fit_cache = {}
    configurations = [("single", (event,)) for event in EVENTS] + [("joint", PRIMARY)]
    for outcome in OUTCOMES:
        for model, events in configurations:
            fits = []
            for horizon in HORIZONS:
                fit = fit_iv(make_design(prepared, outcome, events, horizon))
                fits.append(fit)
                fit_cache[(outcome, events, horizon)] = fit
                records["baseline_iv"].extend(coefficient_rows(fit, model))
                stages, geometry = first_stage_diagnostics(fit)
                shared = {"model": model, "outcome": outcome, "horizon": horizon, "n": fit.n}
                records["first_stage"].extend([{**shared, **row} for row in stages])
                records["joint_identification"].append({**shared, "events": ",".join(events), **geometry})
                if model == "joint":
                    records["sw_conditional_first_stage"].extend(
                        [{**shared, **row} for row in sanderson_windmeijer_conditional_f(fit)]
                    )
                    if horizon in CHECK_HORIZONS:
                        records["joint_ar_projection_intervals"].extend(
                            [{**shared, **row} for row in joint_ar_projection_intervals(fit)]
                        )
            if model == "joint" or events[0] in PRIMARY:
                records["path_wald"].extend(path_wald(fits, model))
                for horizon in CHECK_HORIZONS:
                    conventional = fit_iv(make_design(prepared, outcome, events, horizon, price_lags=2))
                    records["hac_bandwidth"].extend(hac_bandwidth_sensitivity(conventional, model))
                for horizon in HORIZONS:
                    conventional_design = make_design(prepared, outcome, events, horizon, price_lags=2)
                    common = np.intersect1d(fits[horizon].design.positions, conventional_design.positions)
                    augmented = fit_iv(make_design(prepared, outcome, events, horizon, required_positions=common))
                    conventional = fit_iv(make_design(prepared, outcome, events, horizon, price_lags=2,
                                                      required_positions=common), hac_lag=6)
                    for column, event in enumerate(events):
                        records["inference_comparison"].append({
                            "model": model, "outcome": outcome, "event": event, "horizon": horizon,
                            "n": augmented.n, "augmented_price_lags": 3, "conventional_price_lags": 2,
                            "augmented_beta": augmented.beta[column],
                            "augmented_hc3_se": np.sqrt(augmented.hc3_cov[column, column]),
                            "conventional_beta": conventional.beta[column],
                            "conventional_hac6_se": np.sqrt(conventional.hac_cov[column, column]),
                            "comparison": "Matched observations; distinct regressions and covariance estimators"})
                for block_length in (1, 6, 12):
                    records["simultaneous_bands"].extend(
                        simultaneous_bands(fits, model, draws, block_length, seed + block_length)
                    )
            if model == "joint":
                records["cross_category_equality"].extend(cross_category_equality(fits))
            else:
                for horizon in CHECK_HORIZONS:
                    records["ar_confidence_sets"].extend(anderson_rubin_confidence_set(fits[horizon]))
            if model == "single":
                for horizon in CHECK_HORIZONS:
                    records["leverage"].extend(leverage_diagnostics(fits[horizon]))

            anticipation_fits = [fit_anticipation(prepared, outcome, events, h) for h in (0, 1)]
            for fit in anticipation_fits:
                for column, event in enumerate(events):
                    row = {"model": model, "outcome": outcome, "event": event,
                           "horizon": fit["horizon"], "instrument_lead": 2, "n": fit["n"],
                           "origin_start": fit["origin_start"], "origin_end": fit["origin_end"],
                           "beta": fit["beta"][column]}
                    for covariance_type in ("hc3", "hc1"):
                        se = np.sqrt(fit[f"{covariance_type}_cov"][column, column])
                        row[f"{covariance_type}_se"] = se
                        row[f"{covariance_type}_p"] = 2 * normal_sf(abs(row["beta"] / se))
                    records["anticipation"].append(row)
            for column, event in enumerate(events):
                scores = np.column_stack([fit["hc3_influence"][:, column] for fit in anticipation_fits])
                coefficients = np.array([fit["beta"][column] for fit in anticipation_fits])
                statistic, df, pvalue = _wald(coefficients, scores.T @ scores)
                records["anticipation_joint_tests"].append({
                    "model": model, "outcome": outcome, "event": event,
                    "null": "theta_h0 = theta_h1 = 0", "covariance": "HC3", "hac_lag": 0,
                    "wald_chi2": statistic, "df": df, "p_value": pvalue})
            if model == "joint":
                scores = np.column_stack([fit["hc3_influence"] for fit in anticipation_fits])
                coefficients = np.concatenate([fit["beta"] for fit in anticipation_fits])
                statistic, df, pvalue = _wald(coefficients, scores.T @ scores)
                records["anticipation_joint_tests"].append({
                    "model": model, "outcome": outcome, "event": "all_three",
                    "null": "all six pre-event coefficients = 0", "covariance": "HC3", "hac_lag": 0,
                    "wald_chi2": statistic, "df": df, "p_value": pvalue})

    # Sensitivity exercises focus on the three focal event types.
    for outcome in OUTCOMES:
        for event in PRIMARY:
            events = (event,)
            for horizon in CHECK_HORIZONS:
                baseline = fit_cache[(outcome, events, horizon)]
                for date, label in EPISODES.items():
                    omitted = fit_iv(make_design(prepared, outcome, events, horizon, omit_event=date))
                    row = coefficient_rows(omitted, "single", "event_window_omitted")[0]
                    row.update({"event_month": date, "episode": label, "omitted_months": "t_event +/- 1",
                                "baseline_beta": baseline.beta[0], "difference": omitted.beta[0] - baseline.beta[0],
                                "n_removed": baseline.n - omitted.n,
                                "sign_preserved": bool(np.sign(omitted.beta[0]) == np.sign(baseline.beta[0]))})
                    records["event_drop"].append(row)
                for label, lower, upper in [
                    ("pre_2008", None, "2007-12-01"), ("post_2008", "2008-01-01", None),
                    ("pre_2012", None, "2011-12-01"), ("post_2012", "2012-01-01", None),
                    ("pre_2019", None, "2018-12-01"), ("post_2019", "2019-01-01", None),
                ]:
                    design = make_design(prepared, outcome, events, horizon, origin_start=lower, origin_end=upper)
                    if len(design.y) <= design.controls.shape[1] + 10:
                        continue
                    fit = fit_iv(design)
                    row = coefficient_rows(fit, "single", label)[0]
                    row["interpretation"] = "Descriptive origin-month split; no independent-subsample difference test"
                    records["origin_splits"].append(row)
                for lags, other_lags, label in [(2, 2, "non_augmented_2_price_lags"),
                                                (4, 2, "extra_4_price_lags"),
                                                (3, 3, "three_lags_all_variables")]:
                    alternative = make_design(prepared, outcome, events, horizon, price_lags=lags, other_lags=other_lags)
                    common = np.intersect1d(baseline.design.positions, alternative.positions)
                    matched_base = fit_iv(make_design(prepared, outcome, events, horizon, required_positions=common))
                    fitted = fit_iv(make_design(prepared, outcome, events, horizon, price_lags=lags,
                                               required_positions=common, other_lags=other_lags))
                    row = coefficient_rows(fitted, "single", label)[0]
                    row.update({"matched_baseline_beta": matched_base.beta[0], "matched_baseline_n": matched_base.n})
                    records["lag_sensitivity"].append(row)
                log_design = make_design(prepared, outcome, events, horizon, instrument="log")
                common = np.intersect1d(baseline.design.positions, log_design.positions)
                matched_base = fit_iv(make_design(prepared, outcome, events, horizon, required_positions=common))
                log_fit = fit_iv(make_design(prepared, outcome, events, horizon, instrument="log", required_positions=common))
                row = coefficient_rows(log_fit, "single", "log_curvature_OLS_equivalent")[0]
                row.update({"matched_baseline_beta": matched_base.beta[0],
                            "interpretation": "Algebraic OLS-equivalent contrast because two treatment lags are controlled"})
                records["log_curvature"].append(row)

            for horizon in HORIZONS:
                conditional_design = make_design(prepared, outcome, events, horizon, current_production=True)
                common = conditional_design.positions
                baseline = fit_iv(make_design(prepared, outcome, events, horizon, required_positions=common))
                conditional = fit_iv(conditional_design)
                row = coefficient_rows(conditional, "single", "current_production_growth_control")[0]
                row.update({"matched_baseline_beta": baseline.beta[0], "matched_baseline_n": baseline.n,
                            "interpretation": "Conditional specification; production growth is not an identified supply shock"})
                records["production_control"].append(row)

    # Mechanism evidence: demand-side world activity and physical oil-supply growth.
    # These are reduced-form outcome responses, not a formal mediation decomposition.
    for outcome in MECHANISM_OUTCOMES:
        fits = []
        for horizon in HORIZONS:
            fit = fit_iv(make_mechanism_design(prepared, outcome, PRIMARY, horizon))
            fits.append(fit)
            records["mechanism_iv"].extend(
                coefficient_rows(fit, "joint", "mechanism_lag_augmented")
            )
        for block_length in (1, 6, 12):
            records["mechanism_simultaneous_bands"].extend(
                simultaneous_bands(fits, "joint", draws, block_length,
                                   seed + 100 + block_length)
            )

    results = {name: pd.DataFrame(rows) for name, rows in records.items()}
    # Keep baseline exports unambiguous: HC3 primary, HC1 finite-sample sensitivity.
    # HAC comparisons are in separate unaugmented-model tables.
    for name in results:
        if name not in ("hac_bandwidth", "inference_comparison"):
            if "covariance" in results[name] and name in ("ar_confidence_sets", "cross_category_equality"):
                results[name] = results[name].loc[results[name].covariance.isin(["HC3", "HC1"])].copy()
            results[name] = results[name].drop(columns=[c for c in results[name] if "hac" in c.lower()])
    results["path_wald"] = results["path_wald"].drop(columns=["wald_chi2", "df", "p_value"])
    results["summary_statistics"], results["endpoints"] = summary_statistics(prepared)
    for name, frame in results.items():
        frame.to_csv(output / f"{name}.csv", index=False, float_format="%.12g")
    canonical_data = prepared[["date", "lwti", "lbrent", "lwip", "lgop", *EVENTS]].to_csv(index=False)
    metadata = {
        "input_sha256_canonical_csv": hashlib.sha256(canonical_data.encode()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "statistical_functions": "Python math/statistics (no SciPy dependency)",
        "calendar_start": prepared.date.min().date().isoformat(),
        "calendar_end": prepared.date.max().date().isoformat(), "calendar_months": len(prepared),
        "outcomes": list(OUTCOMES), "events": list(EVENTS), "joint_events": list(PRIMARY),
        "baseline": "Exactly identified FWL IV with HC3; 2 WIP, 2 production-growth, 3 price, 2 treatment lags",
        "instrument": "second difference of the raw event index",
        "treatment": "natural log(1 + raw event index)",
        "outcome": "natural log real oil-price level at t+h",
        "horizons": list(HORIZONS), "anticipation": "fixed Z[t+2] at h=0,1, with origin-t predetermined controls",
        "primary_inference": "HC3: structural residual divided by one minus full projected-regressor leverage; no HAC bandwidth and no additional HC1 multiplier",
        "hac": "Separate conventional two-price-lag model: HAC6 matched-sample comparison and bandwidths 4,12,24,48",
        "inference_assumption": "HC3 pointwise inference requires negligible serial covariance of the relevant IV score; lag structure alone is not a general proof for arbitrary instruments",
        "path_wald_horizons": list(WALD_HORIZONS), "path_wald_covariance": "Stacked calendar-aligned HC3 influence rows",
        "first_stage_inference": "HC3 excluded-instrument relevance Wald tests; not conditional strength or a universal F>10 rule",
        "hac_bandwidth_sensitivity": [4, 12, 24, 48],
        "ar_confidence_sets": "Analytic scalar Anderson-Rubin inversion at h=0,12,24,48; primary HC3, HC1 sensitivity; 95% asymptotic coverage",
        "sw_conditional_first_stage": "Sanderson-Windmeijer homoskedastic conditional F for each endogenous regressor in the three-category joint model",
        "joint_ar_projection_intervals": "HC3 95% joint Anderson-Rubin region at h=0,12,24,48, projected onto each coefficient after profiling the other two",
        "mechanism_outcomes": "Joint-model lag-augmented IV-LPs for log world industrial production and global oil-production growth; descriptive channel evidence, not formal mediation",
        "cross_category_equality": "Joint-model coefficient equality at h=0,12,24,48 and across six selected horizons; equal log(1+index) changes",
        "multiplier_draws": draws, "multiplier_seed": seed, "multiplier_block_lengths": [1, 6, 12],
        "multiplier_interpretation": "Independent month multipliers baseline; blocks6/12 supplementary simultaneous-path sensitivity, not HAC pointwise inference",
        "missing_data": "October 2025 CPI log-linearly interpolated; no other fill; construct leads/lags on full calendar; complete cases per regression",
        "joint_strength_warning": "Reported joint excluded-instrument F is not a conditional-strength statistic",
        "log_curvature_warning": "Controlling two treatment lags makes log-curvature IV algebraically OLS-equivalent",
        "split_warning": "Origin-split estimates are descriptive; forecast-error windows can overlap across the cutoff despite disjoint fixed-horizon outcome dates",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=20260919)
    arguments = parser.parse_args()
    frames = run_all(arguments.data, arguments.out, arguments.draws, arguments.seed)
    print(f"Completed {len(frames)} output tables in {arguments.out.resolve()}")
