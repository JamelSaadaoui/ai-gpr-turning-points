"""Synchronize the companion workbook after completing real oil prices.

The workbook is patched at the OOXML level so formula cells and their cached
values remain intact.  Only the two October 2025 real-price cells and the
corresponding dictionary descriptions are changed.
"""

from pathlib import Path
import tempfile
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "data" / "Geopolitical_Turning_Points_Data.xlsx"
DATA = ROOT / "data" / "monthly_data.csv"
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
ET.register_namespace("x", NS)


def cell(root, address):
    found = root.find(f".//{{{NS}}}c[@r='{address}']")
    if found is None:
        raise ValueError(f"Workbook cell {address} not found")
    return found


def set_number(root, address, value):
    target = cell(root, address)
    target.attrib.pop("t", None)
    value_node = target.find(f"{{{NS}}}v")
    if value_node is None:
        value_node = ET.SubElement(target, f"{{{NS}}}v")
    value_node.text = format(float(value), ".15g")


def set_text(root, address, value):
    target = cell(root, address)
    target.set("t", "str")
    value_node = target.find(f"{{{NS}}}v")
    if value_node is None:
        value_node = ET.SubElement(target, f"{{{NS}}}v")
    value_node.text = value


def main():
    data = pd.read_csv(DATA, parse_dates=["date"])
    row = data.loc[data.date.eq(pd.Timestamp("2025-10-01"))].iloc[0]
    with zipfile.ZipFile(BOOK, "r") as source:
        sheet1 = ET.fromstring(source.read("xl/worksheets/sheet1.xml"))
        sheet3 = ET.fromstring(source.read("xl/worksheets/sheet3.xml"))
        set_number(sheet1, "B431", row.lwti)
        set_number(sheet1, "C431", row.lbrent)
        suffix = " October 2025 CPI is log-linearly interpolated; nominal oil prices remain observed."
        set_text(sheet3, "D3", "Complete EIA monthly WTI history divided by BLS CUSR0000SA0; through August 2026." + suffix)
        set_text(sheet3, "D4", "Complete EIA monthly Brent history divided by BLS CUSR0000SA0; through August 2026." + suffix)
        set_text(sheet3, "C35", "Only October 2025 CPI is interpolated")
        set_text(sheet3, "D35", "Real prices are complete. The unavailable October 2025 CPI is log-linearly interpolated between September and November; WIP ends June 2026 and production ends February 2026.")
        set_text(sheet3, "C41", "No splice, rebasing or extrapolation; one documented CPI interpolation")
        replacements = {
            "xl/worksheets/sheet1.xml": ET.tostring(sheet1, encoding="utf-8", xml_declaration=True),
            "xl/worksheets/sheet3.xml": ET.tostring(sheet3, encoding="utf-8", xml_declaration=True),
        }
        with tempfile.NamedTemporaryFile(dir=BOOK.parent, suffix=".xlsx", delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, replacements.get(item.filename, source.read(item.filename)))
    temporary.replace(BOOK)
    print(BOOK)


if __name__ == "__main__":
    main()
