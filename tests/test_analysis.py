"""Independent checks for the frozen-data local-projection replication.

Run from any working directory with:
    python -m unittest discover -s tests -v

The reference calculations deliberately use QR residualization and direct
calendar-pair summation rather than the implementation under test.
"""

from pathlib import Path
import hashlib
import json
import sys
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from statistics import NormalDist


PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "data"
sys.path.insert(0, str(PACKAGE / "code"))
import analysis

norm = NormalDist()


def independent_sample(data, outcome, categories, horizon, anticipation=False):
    """Build the equation explicitly, independently of the engine's helper."""
    frame = pd.DataFrame({"date": data.date,
                          "y": data[outcome].shift(-horizon)})
    controls = []
    for variable, values, lags in [
        ("wip", data.lwip, (1, 2)),
        ("oil_growth", data.lgop.diff(), (1, 2)),
        ("price", data[outcome], (1, 2, 3)),
    ]:
        for lag in lags:
            name = f"{variable}_{lag}"
            frame[name] = values.shift(lag)
            controls.append(name)
    for category in categories:
        treatment = np.log1p(data[category])
        instrument = data[category] - 2 * data[category].shift(1) + data[category].shift(2)
        if not anticipation:
            frame[f"x_{category}"] = treatment
        frame[f"z_{category}"] = instrument.shift(-2) if anticipation else instrument
        for lag in (1, 2):
            name = f"{category}_{lag}"
            frame[name] = treatment.shift(lag)
            controls.append(name)
    frame = frame.dropna()
    exog = np.column_stack([np.ones(len(frame)), frame[controls]])
    instruments = frame[[f"z_{c}" for c in categories]].to_numpy(float)
    treatments = (instruments.copy() if anticipation else
                  frame[[f"x_{c}" for c in categories]].to_numpy(float))
    return frame, frame.y.to_numpy(float), treatments, instruments, exog


def independent_iv(y, treatments, instruments, exog):
    """Exactly identified IV via QR/FWL, including full-equation HC1 d.f."""
    basis, _ = np.linalg.qr(exog, mode="reduced")
    residual = lambda values: values - basis @ (basis.T @ values)
    y_res = residual(y)
    x_res = residual(treatments)
    z_res = residual(instruments)
    cross = z_res.T @ x_res
    beta = np.linalg.solve(cross, z_res.T @ y_res)
    errors = y_res - x_res @ beta
    influence = np.linalg.solve(cross, z_res.T).T * errors[:, None]
    correction = len(y) / (len(y) - exog.shape[1] - treatments.shape[1])
    covariance = influence.T @ influence * correction
    return beta, covariance, influence * np.sqrt(correction)



def independent_hc3(y, treatments, instruments, exog):
    """Full-system projected-design 2SLS sandwich, no FWL helper calls."""
    c = exog.copy()
    c[:, 1:] = (c[:, 1:] - c[:, 1:].mean(axis=0)) / c[:, 1:].std(axis=0)
    d = np.column_stack([c, treatments])
    w = np.column_stack([c, instruments])
    q, _ = np.linalg.qr(w, mode="reduced")
    projected = q @ (q.T @ d)
    inverse = np.linalg.pinv(projected)
    coefficients = inverse @ y
    errors = y - d @ coefficients
    hat = np.sum(projected * inverse.T, axis=1)
    full_scores = inverse.T * (errors / (1 - hat))[:, None]
    k = treatments.shape[1]
    scores = full_scores[:, -k:]
    return coefficients[-k:], scores.T @ scores, scores, hat


def independent_calendar_hac(influence, dates, bandwidth):
    """Direct pair summation by elapsed months, not compressed-row position."""
    months = pd.DatetimeIndex(dates).to_period("M").asi8
    covariance = influence.T @ influence
    for lag in range(1, bandwidth + 1):
        later, earlier = np.nonzero(months[:, None] - months[None, :] == lag)
        cross = influence[later].T @ influence[earlier]
        covariance += (1 - lag / (bandwidth + 1)) * (cross + cross.T)
    return covariance


class FrozenDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DATA / "monthly_data.csv", parse_dates=["date"])

    def test_source_checksums(self):
        expected = json.loads((DATA / "input_checksums.json").read_text())
        for relative, digest in expected.items():
            with self.subTest(file=relative):
                actual = hashlib.sha256((DATA / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, digest)

    def test_complete_monthly_calendar(self):
        expected = pd.date_range("1990-01-01", "2026-08-01", freq="MS")
        self.assertEqual(len(self.data), 440)
        self.assertFalse(self.data.date.duplicated().any())
        pd.testing.assert_index_equal(pd.DatetimeIndex(self.data.date), expected,
                                      check_names=False)

    def test_documented_october_2025_cpi_interpolation(self):
        row = self.data.set_index("date").loc["2025-10-01"]
        self.assertTrue(np.isfinite(row.lwti))
        self.assertTrue(np.isfinite(row.lbrent))
        merged = pd.read_csv(DATA / "current_source_merge.csv", parse_dates=["date"])
        october = merged.set_index("date").loc["2025-10-01"]
        expected = np.sqrt(324.245 * 325.063)
        self.assertAlmostEqual(october.cpi_all_items_sa, expected, places=9)
        self.assertTrue(bool(october.cpi_interpolated))

    def test_documented_data_endpoints(self):
        endpoints = {"lwti": "2026-08-01", "lbrent": "2026-08-01",
                     "lwip": "2026-06-01", "lgop": "2026-02-01",
                     "gpr_ai": "2026-08-01"}
        for variable, last in endpoints.items():
            with self.subTest(variable=variable):
                actual = self.data.loc[self.data[variable].notna(), "date"].max()
                self.assertEqual(actual, pd.Timestamp(last))

    def test_frozen_gpr_snapshot_is_used_without_revision(self):
        source = pd.read_csv(DATA / "source_csv" / "ai_gpr_event_types_snapshot.csv",
                             parse_dates=["Date"])
        source = source.rename(columns={"Date": "date", "GPR_AI": "gpr_ai"})
        compare = self.data.merge(source, on="date", validate="one_to_one",
                                  suffixes=("_analysis", "_source"))
        for variable in source.columns.drop("date"):
            with self.subTest(variable=variable):
                np.testing.assert_allclose(compare[variable + "_analysis"],
                                           compare[variable + "_source"],
                                           rtol=1e-14, atol=1e-14)
                self.assertTrue((compare[variable + "_analysis"] >= 0).all())

    def test_complete_source_histories_rebuild_analysis_data(self):
        from build_data import build
        rebuilt = build(PACKAGE)
        pd.testing.assert_frame_equal(rebuilt, self.data, check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-10)


class EstimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(DATA / "monthly_data.csv", parse_dates=["date"])
        cls.prepared = analysis.prepare_data(cls.raw)

    def test_raw_level_curvature_and_log_treatment(self):
        for event in analysis.EVENTS:
            with self.subTest(event=event):
                expected = self.raw[event] - 2 * self.raw[event].shift(1) + self.raw[event].shift(2)
                np.testing.assert_allclose(self.prepared[f"z_{event}"], expected,
                                           rtol=1e-12, atol=1e-12, equal_nan=True)
                np.testing.assert_allclose(self.prepared[f"x_{event}"],
                                           np.log1p(self.raw[event]), rtol=1e-14)

    def test_exact_lag_augmentation(self):
        actual = analysis.control_names("lbrent", ("nuclear_threat",))
        expected = ["L1_lwip", "L2_lwip", "L1_production_growth", "L2_production_growth",
                    "L1_lbrent", "L2_lbrent", "L3_lbrent",
                    "L1_x_nuclear_threat", "L2_x_nuclear_threat"]
        self.assertEqual(actual, expected)
        design = analysis.make_design(self.prepared, "lbrent", ("nuclear_threat",), 24)
        self.assertEqual(design.controls.shape[1], 10)
        self.assertEqual(design.control_names, ("constant", *expected))

    def test_mechanism_lag_augmentation(self):
        activity = analysis.make_mechanism_design(
            self.prepared, "lwip", analysis.PRIMARY, 24)
        production = analysis.make_mechanism_design(
            self.prepared, "production_growth", analysis.PRIMARY, 24)
        self.assertIn("L3_lwip", activity.control_names)
        self.assertNotIn("L3_production_growth", activity.control_names)
        self.assertIn("L3_production_growth", production.control_names)
        self.assertNotIn("L3_lwip", production.control_names)
        self.assertIn("L2_lbrent", activity.control_names)
        self.assertNotIn("L3_lbrent", activity.control_names)
        self.assertEqual(activity.events, analysis.PRIMARY)

    def test_design_selects_only_required_variables(self):
        horizon_zero = analysis.make_design(self.prepared, "lbrent", ("gpr_ai",), 0)
        horizon_one = analysis.make_design(self.prepared, "lbrent", ("gpr_ai",), 1)
        gap = pd.Timestamp("2025-10-01")
        self.assertIn(gap, horizon_zero.dates)
        self.assertIn(gap, horizon_one.dates)
        self.assertEqual(len(horizon_zero.y), 432)
        self.assertEqual(len(horizon_one.y), 432)

    def test_all_single_category_estimates_against_independent_qr(self):
        for outcome in analysis.OUTCOMES:
            for event in analysis.EVENTS:
                for horizon in (0, 1, 6, 12, 24, 48):
                    with self.subTest(outcome=outcome, event=event, horizon=horizon):
                        frame, y, x, z, controls = independent_sample(
                            self.raw, outcome, (event,), horizon)
                        expected, covariance, _ = independent_iv(y, x, z, controls)
                        design = analysis.make_design(self.prepared, outcome, (event,), horizon)
                        actual = analysis.fit_iv(design)
                        pd.testing.assert_index_equal(design.dates, pd.DatetimeIndex(frame.date),
                                                      check_names=False)
                        np.testing.assert_allclose(actual.beta, expected, rtol=1e-9, atol=1e-10)
                        np.testing.assert_allclose(actual.hc1_cov, covariance, rtol=1e-9, atol=1e-11)
                        self.assertEqual(actual.n, len(frame))

    def test_joint_model_influence_and_covariance_mapping(self):
        for outcome in analysis.OUTCOMES:
            for horizon in (0, 1, 12, 24, 48):
                with self.subTest(outcome=outcome, horizon=horizon):
                    frame, y, x, z, controls = independent_sample(
                        self.raw, outcome, analysis.PRIMARY, horizon)
                    expected, covariance, influence = independent_iv(y, x, z, controls)
                    design = analysis.make_design(self.prepared, outcome, analysis.PRIMARY, horizon)
                    actual = analysis.fit_iv(design)
                    np.testing.assert_allclose(actual.beta, expected, rtol=1e-9, atol=1e-10)
                    np.testing.assert_allclose(actual.hc1_cov, covariance, rtol=1e-9, atol=1e-11)
                    np.testing.assert_allclose(actual.influence[design.positions], influence,
                                               rtol=1e-8, atol=1e-10)
                    missing = np.setdiff1d(np.arange(len(self.raw)), design.positions)
                    self.assertTrue((actual.influence[missing] == 0).all())

    def test_calendar_hac_against_independent_elapsed_month_pairs(self):
        for horizon in (1, 12, 24):
            with self.subTest(horizon=horizon):
                frame, y, x, z, controls = independent_sample(
                    self.raw, "lbrent", analysis.PRIMARY, horizon)
                _, _, influence = independent_iv(y, x, z, controls)
                actual = analysis.fit_iv(analysis.make_design(
                    self.prepared, "lbrent", analysis.PRIMARY, horizon), hac_lag=12)
                expected = independent_calendar_hac(influence, frame.date, 12)
                np.testing.assert_allclose(actual.hac_cov, expected, rtol=1e-9, atol=1e-11)

    def test_calendar_hac_does_not_close_missing_months(self):
        # The two scores are two calendar months apart, so HAC lag 1 has no cross term.
        influence = np.array([[1.0], [0.0], [2.0]])
        np.testing.assert_allclose(analysis.hac_covariance(influence, 1), [[5.0]])
        np.testing.assert_allclose(analysis.hac_covariance(influence[[0, 2]], 1), [[7.0]])

    def test_instrument_scaling_invariance(self):
        design = analysis.make_design(self.prepared, "lbrent", analysis.PRIMARY, 24)
        original = analysis.fit_iv(design)
        scaled = analysis.fit_iv(replace(design, z=design.z * np.array([20.0, 0.3, -2.0])))
        np.testing.assert_allclose(scaled.beta, original.beta, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(scaled.hc1_cov, original.hc1_cov, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(scaled.hac_cov, original.hac_cov, rtol=1e-10, atol=1e-11)

    def test_treatment_units_scale_coefficients_and_uncertainty(self):
        design = analysis.make_design(self.prepared, "lbrent", analysis.PRIMARY, 24)
        original = analysis.fit_iv(design)
        scales = np.array([2.0, 3.0, 4.0])
        scaled = analysis.fit_iv(replace(design, x=design.x * scales))
        np.testing.assert_allclose(scaled.beta, original.beta / scales, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(scaled.hc1_cov, original.hc1_cov / np.outer(scales, scales),
                                   rtol=1e-10, atol=1e-11)

    def test_fixed_t_plus_two_anticipation_at_both_pre_event_horizons(self):
        for outcome in analysis.OUTCOMES:
            for events in [("nuclear_threat",), analysis.PRIMARY]:
                for horizon in (0, 1):
                    with self.subTest(outcome=outcome, events=events, horizon=horizon):
                        frame, y, x, z, controls = independent_sample(
                            self.raw, outcome, events, horizon, anticipation=True)
                        expected, covariance, influence = independent_iv(y, x, z, controls)
                        expected_hac = independent_calendar_hac(influence, frame.date, 4)
                        actual = analysis.fit_anticipation(self.prepared, outcome, events, horizon)
                        self.assertEqual(actual["n"], len(frame))
                        np.testing.assert_allclose(actual["beta"], expected, rtol=1e-9, atol=1e-10)
                        np.testing.assert_allclose(actual["hc1_cov"], covariance, rtol=1e-9, atol=1e-11)
                        np.testing.assert_allclose(actual["hac_cov"], expected_hac, rtol=1e-9, atol=1e-11)

    def test_anticipation_excludes_event_month_and_later(self):
        for horizon in (-1, 2, 12, 48):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                analysis.fit_anticipation(self.prepared, "lbrent", ("gpr_ai",), horizon)

    def test_missing_input_month_is_reinserted_before_differencing(self):
        removed = self.raw[self.raw.date.ne("2020-03-01")]
        prepared = analysis.prepare_data(removed).set_index("date")
        self.assertEqual(len(prepared), 440)
        self.assertTrue(prepared.loc["2020-03-01", "gpr_ai"] !=
                        prepared.loc["2020-03-01", "gpr_ai"])
        self.assertTrue(pd.isna(prepared.loc["2020-04-01", "z_gpr_ai"]))
        self.assertTrue(pd.isna(prepared.loc["2020-04-01", "L1_x_gpr_ai"]))

    def test_input_validation(self):
        duplicate = pd.concat([self.raw, self.raw.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            analysis.prepare_data(duplicate)
        invalid = self.raw.copy()
        invalid.loc[0, "gpr_ai"] = -1
        with self.assertRaises(ValueError):
            analysis.prepare_data(invalid)

    def test_known_numerical_reference_points(self):
        checks = [
            ("diplomatic_tension", 24, 413, -0.3079175431594433, 0.10187162714052052),
            ("nuclear_threat", 24, 413, 0.09801169198627598, 0.04833558884522053),
            ("military_conflict", 0, 432, 0.06363256369960153, 0.0272332685555403),
            ("gpr_ai", 1, 432, 0.05107132455127668, 0.08328552184914145),
        ]
        for event, horizon, n, beta, se in checks:
            with self.subTest(event=event, horizon=horizon):
                fit = analysis.fit_iv(analysis.make_design(self.prepared, "lbrent", (event,), horizon))
                self.assertEqual(fit.n, n)
                self.assertAlmostEqual(fit.beta[0], beta, places=10)
                self.assertAlmostEqual(np.sqrt(fit.hc1_cov[0, 0]), se, places=10)

    def test_log_curvature_contrast_is_algebraically_ols_equivalent(self):
        design = analysis.make_design(self.prepared, "lbrent", ("nuclear_threat",), 24,
                                      instrument="log")
        fitted = analysis.fit_iv(design)
        expected = np.linalg.lstsq(np.column_stack([design.controls, design.x]),
                                    design.y, rcond=None)[0][-1]
        self.assertAlmostEqual(fitted.beta[0], expected, places=10)

    def test_first_stage_relevance_statistics_match_independent_ols(self):
        for events in [("nuclear_threat",), analysis.PRIMARY]:
            with self.subTest(events=events):
                design = analysis.make_design(self.prepared, "lbrent", events, 24)
                fit = analysis.fit_iv(design)
                rows, geometry = analysis.first_stage_diagnostics(fit)
                for column, row in enumerate(rows):
                    beta, covariance, _ = independent_iv(
                        design.x[:, column], design.z, design.z, design.controls)
                    statistic = float(beta @ np.linalg.solve(covariance, beta))
                    self.assertAlmostEqual(row["excluded_instrument_wald_chi2"], statistic, places=7)
                    self.assertAlmostEqual(row["excluded_instrument_wald_F"], statistic / len(events), places=7)
                    self.assertGreaterEqual(row["partial_r2"], 0)
                    self.assertLessEqual(row["partial_r2"], 1)
                self.assertGreaterEqual(geometry["minimum_canonical_correlation"], 0)
                self.assertLessEqual(geometry["maximum_canonical_correlation"], 1)
                self.assertEqual(geometry["first_stage_rank"], len(events))
                if len(events) > 1:
                    self.assertTrue(all("not conditional strength" in row["diagnostic_type"]
                                        for row in rows))

    def test_anderson_rubin_joint_zero_test_matches_reduced_form(self):
        for events in [("nuclear_threat",), analysis.PRIMARY]:
            with self.subTest(events=events):
                design = analysis.make_design(self.prepared, "lbrent", events, 24)
                fit = analysis.fit_iv(design)
                _, geometry = analysis.first_stage_diagnostics(fit)
                beta, _, influence = independent_iv(design.y, design.z, design.z, design.controls)
                covariance = independent_calendar_hac(influence, design.dates, fit.hac_lag)
                expected = float(beta @ np.linalg.solve(covariance, beta))
                self.assertAlmostEqual(geometry["ar_joint_zero_statistic"], expected, places=9)
                self.assertEqual(geometry["ar_df"], len(events))

    def test_sanderson_windmeijer_conditional_f_matches_definition(self):
        fit = analysis.fit_iv(analysis.make_design(
            self.prepared, "lbrent", analysis.PRIMARY, 24))
        rows = analysis.sanderson_windmeijer_conditional_f(fit)
        x, z = fit.x_residual, fit.z_residual
        qz, _ = np.linalg.qr(z, mode="reduced")
        q, k = z.shape[1], x.shape[1]
        residual_df = fit.n - fit.controls_rank - q
        for column, row in enumerate(rows):
            other = np.delete(x, column, axis=1)
            delta = np.linalg.lstsq(qz @ (qz.T @ other), x[:, column], rcond=None)[0]
            conditional_residual = x[:, column] - other @ delta
            unrestricted = conditional_residual - z @ np.linalg.lstsq(
                z, conditional_residual, rcond=None)[0]
            expected = ((conditional_residual @ conditional_residual - unrestricted @ unrestricted)
                        / (q - (k - 1))) / ((unrestricted @ unrestricted) / residual_df)
            self.assertAlmostEqual(row["sw_conditional_F"], expected, places=9)
            self.assertEqual(row["conditional_df"], 1)

    def test_joint_ar_projection_contains_2sls_and_has_correct_boundaries(self):
        fit = analysis.fit_iv(analysis.make_design(
            self.prepared, "lbrent", analysis.PRIMARY, 24))
        rows = analysis.joint_ar_projection_intervals(fit)
        critical = analysis.chi2_ppf(.95, len(analysis.PRIMARY))
        self.assertAlmostEqual(critical, 7.814727903251179, places=9)
        for column, row in enumerate(rows):
            self.assertLessEqual(row["lower"], fit.beta[column])
            self.assertGreaterEqual(row["upper"], fit.beta[column])
            for boundary in (row["lower"], row["upper"]):
                profiled = analysis._profile_joint_ar(fit, column, boundary)
                self.assertAlmostEqual(profiled, critical, places=4)

    def test_score_bootstrap_is_seeded_and_bands_are_ordered(self):
        fits = [analysis.fit_iv(analysis.make_design(
            self.prepared, "lbrent", ("nuclear_threat",), horizon)) for horizon in range(49)]
        first = pd.DataFrame(analysis.simultaneous_bands(fits, "single", 99, 6, 12345))
        second = pd.DataFrame(analysis.simultaneous_bands(fits, "single", 99, 6, 12345))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 49)
        self.assertTrue((first.sim95_low <= first.sim90_low).all())
        self.assertTrue((first.sim90_low <= first.beta).all())
        self.assertTrue((first.beta <= first.sim90_high).all())
        self.assertTrue((first.sim90_high <= first.sim95_high).all())

    def test_quadratic_confidence_set_topologies(self):
        cases = [
            (1, 0, -1, "bounded_interval", -1, 1, np.nan, np.nan),
            (-1, 0, 1, "two_unbounded_rays", -np.inf, -1, 1, np.inf),
            (1, 0, 1, "empty_set", np.nan, np.nan, np.nan, np.nan),
            (-1, 0, -1, "whole_real_line", -np.inf, np.inf, np.nan, np.nan),
            (0, 2, -4, "left_ray", -np.inf, 2, np.nan, np.nan),
            (0, -2, 4, "right_ray", 2, np.inf, np.nan, np.nan),
            (0, 0, 0, "whole_real_line", -np.inf, np.inf, np.nan, np.nan),
            (0, 0, 1, "empty_set", np.nan, np.nan, np.nan, np.nan),
            (1, -2, 1, "bounded_interval", 1, 1, np.nan, np.nan),
            (-1, 2, -1, "whole_real_line", -np.inf, np.inf, np.nan, np.nan),
        ]
        for a, b, c, kind, lo1, hi1, lo2, hi2 in cases:
            with self.subTest(coefficients=(a, b, c)):
                actual = analysis.quadratic_acceptance_set(a, b, c)
                self.assertEqual(actual["set_type"], kind)
                np.testing.assert_allclose(
                    [actual[k] for k in ("lower_1", "upper_1", "lower_2", "upper_2")],
                    [lo1, hi1, lo2, hi2], equal_nan=True)

    def test_nearly_linear_quadratic_retains_unbounded_domain_topology(self):
        # A tiny nonzero quadratic term still matters sufficiently far from zero.
        actual = analysis.quadratic_acceptance_set(1e-14, 0, -1)
        self.assertEqual(actual["set_type"], "bounded_interval")
        np.testing.assert_allclose([actual["lower_1"], actual["upper_1"]], [-1e7, 1e7])
        extreme = analysis.quadratic_acceptance_set(1, 1e14, -1)
        self.assertEqual(extreme["set_type"], "bounded_interval")
        np.testing.assert_allclose([extreme["lower_1"], extreme["upper_1"]], [-1e14, 1e-14])

    def test_quadratic_solution_membership_matches_inequality(self):
        rng = np.random.default_rng(20260919)
        for _ in range(100):
            a, b, c = rng.normal(size=3)
            confidence = analysis.quadratic_acceptance_set(a, b, c)
            for value in np.linspace(-10, 10, 41):
                belongs = ((confidence["lower_1"] <= value <= confidence["upper_1"]) or
                           (confidence["lower_2"] <= value <= confidence["upper_2"]))
                self.assertEqual(belongs, bool(a * value ** 2 + b * value + c <= 0))

    def test_ar_confidence_boundaries_equal_direct_null_regression(self):
        for outcome in analysis.OUTCOMES:
            for event in ("diplomatic_tension", "nuclear_threat", "sanctions", "coup"):
                for horizon in (0, 24, 48):
                    design = analysis.make_design(self.prepared, outcome, (event,), horizon)
                    fitted = analysis.fit_iv(design)
                    for row in analysis.anderson_rubin_confidence_set(fitted):
                        with self.subTest(outcome=outcome, event=event, horizon=horizon,
                                          covariance=row["covariance"]):
                            critical = analysis.chi2_ppf_df1(row["coverage"])
                            boundaries = [row[key] for key in ("lower_1", "upper_1", "lower_2", "upper_2")
                                          if np.isfinite(row[key])]
                            probes = [0.0, fitted.beta[0]]
                            for boundary in boundaries:
                                reduced_form, hc1_cov, influence = independent_iv(
                                    design.y - boundary * design.x[:, 0],
                                    design.z, design.z, design.controls)
                                covariance = (independent_calendar_hac(influence, design.dates, row["hac_lag"])
                                              if row["hac_lag"] else hc1_cov)
                                if row["covariance"] == "HC3":
                                    _, covariance, _, _ = independent_hc3(
                                        design.y - boundary * design.x[:, 0], design.z, design.z, design.controls)
                                statistic = float(reduced_form @ np.linalg.solve(covariance, reduced_form))
                                self.assertAlmostEqual(statistic, critical, places=7)
                                margin = max(0.005, abs(boundary) * 0.05)
                                probes += [boundary - margin, boundary + margin]
                            for value in probes:
                                reduced_form, hc1_cov, influence = independent_iv(
                                    design.y - value * design.x[:, 0],
                                    design.z, design.z, design.controls)
                                covariance = (independent_calendar_hac(influence, design.dates, row["hac_lag"])
                                              if row["hac_lag"] else hc1_cov)
                                if row["covariance"] == "HC3":
                                    _, covariance, _, _ = independent_hc3(
                                        design.y - value * design.x[:, 0], design.z, design.z, design.controls)
                                statistic = float(reduced_form @ np.linalg.solve(covariance, reduced_form))
                                belongs = ((row["lower_1"] <= value <= row["upper_1"]) or
                                           (row["lower_2"] <= value <= row["upper_2"]))
                                self.assertEqual(belongs, bool(statistic <= critical))

    def test_ar_inversion_coverage_and_multivariate_validation(self):
        fitted = analysis.fit_iv(analysis.make_design(self.prepared, "lbrent", ("nuclear_threat",), 24))
        intervals90 = analysis.anderson_rubin_confidence_set(fitted, coverage=0.90)
        intervals95 = analysis.anderson_rubin_confidence_set(fitted, coverage=0.95)
        for narrow, wide in zip(intervals90, intervals95):
            self.assertLessEqual(wide["lower_1"], narrow["lower_1"])
            self.assertGreaterEqual(wide["upper_1"], narrow["upper_1"])
        joint = analysis.fit_iv(analysis.make_design(self.prepared, "lbrent", analysis.PRIMARY, 24))
        with self.assertRaises(ValueError):
            analysis.anderson_rubin_confidence_set(joint)
        for invalid in (0, 1, -0.1, 1.1):
            with self.subTest(coverage=invalid), self.assertRaises(ValueError):
                analysis.anderson_rubin_confidence_set(fitted, coverage=invalid)

    def test_cross_category_equality_uses_full_covariance(self):
        contrasts = {
            "all_three_equal": np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]),
            "military_minus_diplomatic": np.array([[1.0, -1.0, 0.0]]),
            "military_minus_nuclear": np.array([[1.0, 0.0, -1.0]]),
            "diplomatic_minus_nuclear": np.array([[0.0, 1.0, -1.0]]),
        }
        fits = [analysis.fit_iv(analysis.make_design(self.prepared, "lbrent", analysis.PRIMARY, h))
                for h in analysis.WALD_HORIZONS]
        references = {}
        for fitted in fits:
            frame, y, x, z, controls = independent_sample(
                self.raw, "lbrent", analysis.PRIMARY, fitted.design.horizon)
            beta, hc1_cov, influence = independent_iv(y, x, z, controls)
            references[fitted.design.horizon] = (frame, beta, hc1_cov, influence)
        for row in analysis.cross_category_equality(fits):
            if row["scope"] != "pointwise":
                continue
            with self.subTest(horizon=row["horizon"], test=row["test"], covariance=row["covariance"]):
                frame, beta, covariance, influence = references[int(row["horizon"])]
                if row["hac_lag"]:
                    covariance = independent_calendar_hac(influence, frame.date, row["hac_lag"])
                if row["covariance"] == "HC3":
                    fitted = next(f for f in fits if f.design.horizon == int(row["horizon"]))
                    d = fitted.design
                    _, covariance, _, _ = independent_hc3(d.y, d.x, d.z, d.controls)
                contrast = contrasts[row["test"]]
                difference = contrast @ beta
                contrast_covariance = contrast @ covariance @ contrast.T
                expected = float(difference @ np.linalg.solve(contrast_covariance, difference))
                self.assertAlmostEqual(row["wald_chi2"], expected, places=8)
                self.assertEqual(row["df"], len(contrast))
                self.assertAlmostEqual(row["p_value"], analysis.chi2_sf(expected, len(contrast)), places=11)

    def test_cross_category_selected_path_preserves_cross_horizon_covariance(self):
        contrast = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        fits = [analysis.fit_iv(analysis.make_design(self.prepared, "lbrent", analysis.PRIMARY, h))
                for h in analysis.WALD_HORIZONS]
        scores = []
        differences = []
        for fitted in fits:
            frame, y, x, z, controls = independent_sample(
                self.raw, "lbrent", analysis.PRIMARY, fitted.design.horizon)
            beta, _, influence = independent_iv(y, x, z, controls)
            padded = np.zeros((len(self.raw), 2))
            positions = pd.DatetimeIndex(self.raw.date).get_indexer(frame.date)
            padded[positions] = influence @ contrast.T
            scores.append(padded)
            differences.extend(contrast @ beta)
        scores = np.column_stack(scores)
        differences = np.asarray(differences)
        for row in analysis.cross_category_equality(fits):
            if row["scope"] != "selected_path":
                continue
            with self.subTest(hac_lag=row["hac_lag"]):
                covariance = independent_calendar_hac(scores, self.raw.date, row["hac_lag"])
                if row["covariance"] == "HC3":
                    corrected = []
                    for fitted in fits:
                        d = fitted.design
                        _, _, observed, _ = independent_hc3(d.y, d.x, d.z, d.controls)
                        padded = np.zeros((len(self.raw), 2))
                        padded[d.positions] = observed @ contrast.T
                        corrected.append(padded)
                    corrected = np.column_stack(corrected)
                    covariance = corrected.T @ corrected
                expected = float(differences @ np.linalg.solve(covariance, differences))
                self.assertAlmostEqual(row["wald_chi2"], expected, places=7)
                self.assertEqual(row["df"], 12)
                self.assertAlmostEqual(row["p_value"], analysis.chi2_sf(expected, 12), places=11)

    def test_bandwidth_sensitivity_grid_matches_calendar_hac(self):
        for events in [("nuclear_threat",), analysis.PRIMARY]:
            design = analysis.make_design(self.prepared, "lbrent", events, 24)
            fitted = analysis.fit_iv(design)
            _, _, influence = independent_iv(design.y, design.x, design.z, design.controls)
            rows = analysis.hac_bandwidth_sensitivity(fitted, "single" if len(events) == 1 else "joint")
            self.assertEqual({row["hac_lag"] for row in rows}, {4, 12, 24, 48})
            for row in rows:
                with self.subTest(events=events, event=row["event"], lag=row["hac_lag"]):
                    covariance = independent_calendar_hac(influence, design.dates, row["hac_lag"])
                    column = events.index(row["event"])
                    se = np.sqrt(covariance[column, column])
                    self.assertAlmostEqual(row["hac_se"], se, places=10)
                    self.assertAlmostEqual(row["hac_p"], 2 * analysis.normal_sf(abs(fitted.beta[column] / se)), places=10)
                    self.assertAlmostEqual(row["hac_95_low"], fitted.beta[column] - norm.inv_cdf(.975) * se, places=10)
                    self.assertAlmostEqual(row["hac_95_high"], fitted.beta[column] + norm.inv_cdf(.975) * se, places=10)

    def test_primary_hc3_matches_full_system_2sls(self):
        for outcome in analysis.OUTCOMES:
            for events in [(event,) for event in analysis.EVENTS] + [analysis.PRIMARY]:
                for h in (0, 12, 24, 48):
                    d = analysis.make_design(self.prepared, outcome, events, h)
                    fit = analysis.fit_iv(d)
                    beta, covariance, influence, leverage = independent_hc3(d.y, d.x, d.z, d.controls)
                    np.testing.assert_allclose(fit.beta, beta, rtol=1e-8, atol=1e-9)
                    np.testing.assert_allclose(fit.hc3_cov, covariance, rtol=1e-8, atol=1e-10)
                    np.testing.assert_allclose(fit.leverage, leverage, rtol=1e-8, atol=1e-10)
                    np.testing.assert_allclose(fit.hc3_influence[d.positions], influence, rtol=1e-7, atol=1e-9)
                    for col, row in enumerate(analysis.coefficient_rows(fit, "joint" if len(events)>1 else "single")):
                        self.assertEqual(row["primary_covariance"], "HC3")
                        self.assertAlmostEqual(row["hc3_se"], np.sqrt(covariance[col,col]), places=9)
                        for coverage in (90,95):
                            z = norm.inv_cdf((1+coverage/100)/2)
                            self.assertAlmostEqual(row[f"hc3_{coverage}_low"], beta[col]-z*np.sqrt(covariance[col,col]), places=8)

    def test_hc3_anticipation_matches_full_design_ols(self):
        for outcome in analysis.OUTCOMES:
            for events in [("nuclear_threat",), analysis.PRIMARY]:
                for h in (0,1):
                    frame, y, x, z, c = independent_sample(self.raw, outcome, events, h, anticipation=True)
                    _, expected, _, _ = independent_hc3(y,x,z,c)
                    actual = analysis.fit_anticipation(self.prepared,outcome,events,h)
                    np.testing.assert_allclose(actual["hc3_cov"],expected,rtol=1e-8,atol=1e-11)

    def test_hc3_reduces_to_ols_hc3_when_instruments_equal_regressors(self):
        d=analysis.make_design(self.prepared,"lbrent",("nuclear_threat",),24,instrument="log")
        fit=analysis.fit_iv(d)
        _, expected, _, _=independent_hc3(d.y,d.x,d.x,d.controls)
        np.testing.assert_allclose(fit.hc3_cov,expected,rtol=1e-8,atol=1e-10)

    def test_three_lags_all_variables_sensitivity_has_correct_controls(self):
        d=analysis.make_design(self.prepared,"lbrent",analysis.PRIMARY,12,other_lags=3)
        for name in ["lwip","production_growth",*[f"x_{e}" for e in analysis.PRIMARY]]:
            self.assertIn(f"L3_{name}",d.control_names)
        self.assertEqual(len(d.control_names),19)

    def test_hc3_first_stage_and_joint_zero_match_full_design(self):
        for events in [("military_conflict",), analysis.PRIMARY]:
            d=analysis.make_design(self.prepared,"lbrent",events,24)
            fit=analysis.fit_iv(d)
            rows, geometry=analysis.first_stage_diagnostics(fit)
            for column,row in enumerate(rows):
                b,v,_,_=independent_hc3(d.x[:,column],d.z,d.z,d.controls)
                expected=float(b @ np.linalg.solve(v,b))
                self.assertAlmostEqual(row["excluded_instrument_wald_chi2_hc3"],expected,places=6)
            b,v,_,_=independent_hc3(d.y,d.z,d.z,d.controls)
            expected=float(b @ np.linalg.solve(v,b))
            self.assertAlmostEqual(geometry["ar_joint_zero_statistic_hc3"],expected,places=8)

    def test_published_baseline_has_no_hac_and_comparison_has_two_lags(self):
        result=PACKAGE / "results/baseline_iv.csv"
        if not result.exists():
            self.skipTest("Run estimation before checking its exported specification.")
        baseline=pd.read_csv(result)
        self.assertTrue(baseline.primary_covariance.eq("HC3").all())
        self.assertFalse(any("hac" in c.lower() for c in baseline))
        comparisons=pd.read_csv(PACKAGE / "results/inference_comparison.csv")
        self.assertTrue(comparisons.augmented_price_lags.eq(3).all())
        self.assertTrue(comparisons.conventional_price_lags.eq(2).all())
        for event,h in [("military_conflict",0),("nuclear_threat",24)]:
            d=analysis.make_design(self.prepared,"lbrent",(event,),h)
            conventional=analysis.make_design(self.prepared,"lbrent",(event,),h,price_lags=2,
                                              required_positions=d.positions)
            b,_,influence=independent_iv(conventional.y,conventional.x,conventional.z,conventional.controls)
            v=independent_calendar_hac(influence,conventional.dates,6)
            row=comparisons.query("model=='single' and outcome=='lbrent' and event==@event and horizon==@h").iloc[0]
            self.assertEqual(row.n,len(conventional.y))
            self.assertAlmostEqual(row.conventional_beta,b[0],places=10)
            self.assertAlmostEqual(row.conventional_hac6_se,np.sqrt(v[0,0]),places=10)

    def test_internal_hac_comparison_statistics(self):
        for events in [("nuclear_threat",), analysis.PRIMARY]:
            design = analysis.make_design(self.prepared, "lbrent", events, 24)
            fitted = analysis.fit_iv(design)
            rows, geometry = analysis.first_stage_diagnostics(fitted)
            for column, row in enumerate(rows):
                with self.subTest(events=events, column=column):
                    beta, _, influence = independent_iv(design.x[:, column],
                                                        design.z, design.z, design.controls)
                    covariance = independent_calendar_hac(influence, design.dates, 4)
                    expected = float(beta @ np.linalg.solve(covariance, beta))
                    self.assertAlmostEqual(row["excluded_instrument_wald_chi2_hac4"], expected, places=7)
                    self.assertAlmostEqual(row["excluded_instrument_wald_F_hac4"], expected / len(events), places=7)
                    self.assertEqual(row["excluded_instrument_df_hac4"], len(events))
                    self.assertAlmostEqual(row["relevance_p_value_hac4"], analysis.chi2_sf(expected, len(events)), places=11)
            beta, _, influence = independent_iv(design.y, design.z, design.z, design.controls)
            covariance = independent_calendar_hac(influence, design.dates, 25)
            expected = float(beta @ np.linalg.solve(covariance, beta))
            self.assertEqual(geometry["q_hac_horizon"], 25)
            self.assertAlmostEqual(geometry["ar_joint_zero_statistic_hac_horizon"], expected, places=8)
            self.assertEqual(geometry["ar_df_hac_horizon"], len(events))
            self.assertAlmostEqual(geometry["ar_joint_zero_p_value_hac_horizon"],
                                   analysis.chi2_sf(expected, len(events)), places=11)

    def test_primary_selected_path_zero_test_uses_stacked_hc3(self):
        fits = [analysis.fit_iv(analysis.make_design(self.prepared, "lbrent", analysis.PRIMARY, h))
                for h in analysis.WALD_HORIZONS]
        references = []
        for fitted in fits:
            design = fitted.design
            beta, _, influence, _ = independent_hc3(design.y, design.x, design.z, design.controls)
            padded = np.zeros((len(self.raw), 3))
            padded[design.positions] = influence
            references.append((beta, padded))
        for column, row in enumerate(analysis.path_wald(fits, "joint")):
            with self.subTest(event=row["event"]):
                beta = np.array([reference[0][column] for reference in references])
                influence = np.column_stack([reference[1][:, column] for reference in references])
                covariance = influence.T @ influence
                expected = float(beta @ np.linalg.solve(covariance, beta))
                self.assertEqual(row["primary_hac_lag"], 0)
                self.assertEqual(row["primary_df"], 6)
                self.assertAlmostEqual(row["primary_wald_chi2"], expected, places=7)
                self.assertAlmostEqual(row["primary_p_value"], analysis.chi2_sf(expected, 6), places=11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
