"""Build the analysis data from complete current-source histories.

The script performs no splicing, rebasing, or extrapolation.  It fills the
single unavailable October 2025 CPI observation by log-linear interpolation
between the official September and November 2025 levels, records that month in
the merged source table, and writes the analysis-ready monthly CSV.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source_csv"
EVENTS = (
    "gpr_ai", "military_conflict", "diplomatic_tension", "terrorism",
    "civil_war", "nuclear_threat", "coup", "sanctions", "other",
)


def read_eia_monthly(path: Path, value_name: str) -> pd.DataFrame:
    tables = json.loads(path.read_text(encoding="utf-8"))
    rows = tables[8]
    months = {name: number for number, name in enumerate(
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
         "Sep", "Oct", "Nov", "Dec"), start=1)}
    values = []
    for row in rows[1:]:
        if not str(row[0]).strip().isdigit():
            continue
        year = int(row[0])
        for label, raw in zip(rows[0][1:], row[1:]):
            if str(raw).strip() not in {"", "-", "--", "NA"}:
                values.append({"date": pd.Timestamp(year, months[label], 1),
                               value_name: float(raw)})
    result = pd.DataFrame(values).sort_values("date").reset_index(drop=True)
    if result["date"].duplicated().any():
        raise ValueError(f"Duplicate EIA months in {path.name}")
    return result


def read_bls_cpi(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", dtype=str)
    raw.columns = raw.columns.str.strip()
    for column in raw.columns:
        raw[column] = raw[column].str.strip()
    raw = raw.loc[(raw["series_id"] == "CUSR0000SA0") &
                  raw["period"].str.fullmatch(r"M(0[1-9]|1[0-2])")].copy()
    raw["date"] = pd.to_datetime(raw["year"] + "-" + raw["period"].str[1:] + "-01")
    raw["cpi_all_items_sa"] = pd.to_numeric(raw["value"], errors="coerce")
    result = raw[["date", "cpi_all_items_sa"]].sort_values("date").reset_index(drop=True)
    target = pd.Timestamp("2025-10-01")
    if result.loc[result["date"].eq(target), "cpi_all_items_sa"].notna().any():
        raise ValueError("October 2025 CPI is no longer missing; review interpolation rule")
    adjacent = result.set_index("date").loc[
        [pd.Timestamp("2025-09-01"), pd.Timestamp("2025-11-01")],
        "cpi_all_items_sa",
    ]
    if adjacent.isna().any():
        raise ValueError("Adjacent CPI observations required for interpolation are missing")
    interpolated = float(np.exp(np.log(adjacent).mean()))
    result["cpi_interpolated"] = False
    result.loc[result["date"].eq(target), "cpi_all_items_sa"] = interpolated
    result.loc[result["date"].eq(target), "cpi_interpolated"] = True
    return result.dropna(subset=["cpi_all_items_sa"]).reset_index(drop=True)


def read_wip(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="world_IP", header=None, usecols=[0, 1],
                        names=["date", "world_industrial_production"])
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["world_industrial_production"] = pd.to_numeric(
        raw["world_industrial_production"], errors="coerce")
    raw = raw.dropna().sort_values("date").reset_index(drop=True)
    return raw


def read_oil_production(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    raw = pd.DataFrame(rows[1:], columns=["date", "world_crude_oil_production_tbpd"])
    raw["date"] = pd.to_datetime(raw["date"] + "-01")
    raw["world_crude_oil_production_tbpd"] = pd.to_numeric(
        raw["world_crude_oil_production_tbpd"], errors="raise")
    return raw.sort_values("date").reset_index(drop=True)


def read_gpr(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["Date"]).rename(
        columns={"Date": "date", "GPR_AI": "gpr_ai"})
    return raw[["date", *EVENTS]].sort_values("date").reset_index(drop=True)


def build(root: Path = ROOT) -> pd.DataFrame:
    source = root / "data" / "source_csv"
    frames = [
        read_eia_monthly(source / "eia_wti_monthly.json", "wti_usd_per_barrel"),
        read_eia_monthly(source / "eia_brent_monthly.json", "brent_usd_per_barrel"),
        read_bls_cpi(source / "bls_cpi_allitems.txt"),
        read_wip(source / "world_industrial_production.xlsx"),
        read_oil_production(source / "world_crude_oil_production.json"),
        read_gpr(source / "ai_gpr_event_types_snapshot.csv"),
    ]
    calendar = pd.DataFrame({"date": pd.date_range("1990-01-01", "2026-08-01", freq="MS")})
    merged = calendar
    for frame in frames:
        if frame["date"].duplicated().any():
            raise ValueError("A source contains duplicate months")
        merged = merged.merge(frame, on="date", how="left", validate="one_to_one")
    merged["lwti"] = np.log(merged["wti_usd_per_barrel"] /
                             (merged["cpi_all_items_sa"] / 100.0))
    merged["lbrent"] = np.log(merged["brent_usd_per_barrel"] /
                               (merged["cpi_all_items_sa"] / 100.0))
    merged["lwip"] = np.log(merged["world_industrial_production"])
    merged["lgop"] = np.log(merged["world_crude_oil_production_tbpd"])
    analysis = merged[["date", "lwti", "lbrent", "lwip", "lgop", *EVENTS]].copy()

    data_dir = root / "data"
    merged.to_csv(data_dir / "current_source_merge.csv", index=False, float_format="%.12g")
    analysis.to_csv(data_dir / "monthly_data.csv", index=False, float_format="%.12g")
    checksums = {}
    for path in sorted(source.iterdir()):
        if path.is_file():
            checksums[str(path.relative_to(data_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    checksums["current_source_merge.csv"] = hashlib.sha256(
        (data_dir / "current_source_merge.csv").read_bytes()).hexdigest()
    checksums["monthly_data.csv"] = hashlib.sha256(
        (data_dir / "monthly_data.csv").read_bytes()).hexdigest()
    (data_dir / "input_checksums.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    return analysis


if __name__ == "__main__":
    data = build()
    print(f"Built {len(data)} monthly rows: {data.date.min().date()} to {data.date.max().date()}")
