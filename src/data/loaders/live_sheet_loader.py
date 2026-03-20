"""
LiveSheet extractor for Japan merit order workbook.

This module converts the workbook into standardized tabular inputs that can be
used by the fundamental pricing engine without requiring Excel or openpyxl.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import base64
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.settings import JapanRegion


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"a": MAIN_NS, "r": REL_NS}

THERMAL_TECHS = {"gas", "coal", "oil"}

REGION_TO_MARKET_SIDE = {
    JapanRegion.HOKKAIDO.value: "East",
    JapanRegion.TOHOKU.value: "East",
    JapanRegion.TOKYO.value: "East",
    JapanRegion.CHUBU.value: "West",
    JapanRegion.HOKURIKU.value: "West",
    JapanRegion.KANSAI.value: "West",
    JapanRegion.CHUGOKU.value: "West",
    JapanRegion.SHIKOKU.value: "West",
    JapanRegion.KYUSHU.value: "West",
    JapanRegion.OKINAWA.value: "West",
}

TECH_NORMALIZATION = {
    "Battery storage": "battery_storage",
    "Biomass": "biomass",
    "Coal": "coal",
    "Gas": "gas",
    "Geothermal": "geothermal",
    "Hydro": "hydro",
    "Interconnector": "interconnector",
    "Nuclear": "nuclear",
    "Oil": "oil",
    "Pumped hydro": "pumped_hydro",
    "Solar PV": "solar",
    "Wind": "wind",
}

PLANT_TECH_NORMALIZATION = {
    "Battery storage": "battery_storage",
    "Biomass": "biomass",
    "Coal": "coal",
    "Gas": "gas",
    "Geothermal": "geothermal",
    "Hydro": "hydro",
    "Interconnector": "interconnector",
    "Nuclear": "nuclear",
    "Oil": "oil",
    "Pumped hydro": "pumped_hydro",
    "Solar PV": "solar",
    "Wind": "wind",
}


def _col_to_num(col: str) -> int:
    value = 0
    for char in col:
        if char.isalpha():
            value = value * 26 + (ord(char.upper()) - 64)
    return value


def _excel_serial_to_timestamp(value: object) -> pd.Timestamp:
    if value in (None, "", "nan"):
        return pd.NaT
    try:
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
    except (TypeError, ValueError):
        return pd.to_datetime(value, errors="coerce")


@dataclass
class StandardizedFundamentalData:
    plant_master: pd.DataFrame
    monthly_demand: pd.DataFrame
    dashboard_inputs: pd.DataFrame
    monthly_market_inputs: pd.DataFrame
    contract_calendar: pd.DataFrame


class LiveSheetExtractor:
    """Extract standardized tables from the merit order workbook."""

    def __init__(
        self,
        workbook_path: str | Path,
        contract_calendar_path: Optional[str | Path] = None,
    ):
        self.workbook_path = Path(workbook_path)
        self.contract_calendar_path = Path(contract_calendar_path) if contract_calendar_path else None
        self._shared_strings: List[str] = []
        self._sheet_targets: Dict[str, str] = {}
        self._sheet_cache: Dict[str, Dict[int, Dict[str, object]]] = {}

    def extract_standardized_data(self, region: JapanRegion) -> StandardizedFundamentalData:
        self._load_workbook_metadata()

        plant_master = self.extract_plant_master()
        monthly_demand = self.extract_monthly_demand()
        dashboard_inputs = self.extract_dashboard_inputs()
        monthly_market_inputs = self.build_monthly_market_inputs(region, dashboard_inputs)
        contract_calendar = self.extract_contract_calendar()

        return StandardizedFundamentalData(
            plant_master=plant_master,
            monthly_demand=monthly_demand,
            dashboard_inputs=dashboard_inputs,
            monthly_market_inputs=monthly_market_inputs,
            contract_calendar=contract_calendar,
        )

    def save_standardized_data(
        self,
        data: StandardizedFundamentalData,
        output_dir: str | Path,
        region: JapanRegion,
    ) -> Dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = {
            "plant_master": output_path / "plant_master.csv",
            "monthly_demand": output_path / "monthly_demand.csv",
            "dashboard_inputs": output_path / "dashboard_inputs.csv",
            "monthly_market_inputs": output_path / f"monthly_market_inputs_{region.value}.csv",
            "contract_calendar": output_path / "contract_calendar.csv",
        }

        data.plant_master.to_csv(files["plant_master"], index=False)
        data.monthly_demand.to_csv(files["monthly_demand"], index=False)
        data.dashboard_inputs.to_csv(files["dashboard_inputs"], index=False)
        data.monthly_market_inputs.to_csv(files["monthly_market_inputs"], index=False)
        data.contract_calendar.to_csv(files["contract_calendar"], index=False)

        return files

    def extract_plant_master(self) -> pd.DataFrame:
        rows = self._load_sheet_rows("Plant list")
        records: List[Dict[str, object]] = []

        for row_num in sorted(rows):
            if row_num < 11:
                continue
            row = rows[row_num]
            plant_id = row.get("B")
            if plant_id in (None, ""):
                continue

            technology = PLANT_TECH_NORMALIZATION.get(str(row.get("D")).strip())
            if technology is None:
                continue

            record = {
                "plant_id": str(plant_id).strip(),
                "capacity_mw": float(row.get("C", 0) or 0),
                "technology": technology,
                "efficiency": float(row.get("E", 0) or 0),
                "start_date": _excel_serial_to_timestamp(row.get("F")),
                "end_date": _excel_serial_to_timestamp(row.get("G")),
                "market_side": str(row.get("H", "")).strip(),
                "region": str(row.get("I", "")).strip().lower(),
            }
            records.append(record)

        return pd.DataFrame(records)

    def extract_monthly_demand(self) -> pd.DataFrame:
        rows = self._load_sheet_rows("Default inputs")
        date_columns = self._extract_month_columns(rows[17], start_col="CR", end_col="FR")
        records: List[Dict[str, object]] = []

        for row_num in sorted(rows):
            if row_num < 21:
                continue
            row = rows[row_num]
            region = str(row.get("CN", "")).strip()
            contract_type = str(row.get("CO", "")).strip()
            weather_year = row.get("CP")
            if not region or not contract_type or weather_year in (None, ""):
                continue

            for col, month in date_columns:
                value = row.get(col)
                if value in (None, ""):
                    continue
                records.append(
                    {
                        "market_side": region,
                        "contract_type": contract_type,
                        "weather_year": int(float(weather_year)),
                        "month": month,
                        "demand_mw": float(value),
                    }
                )

        return pd.DataFrame(records)

    def extract_dashboard_inputs(self) -> pd.DataFrame:
        rows = self._load_sheet_rows("Dashboard")
        date_columns = self._extract_month_columns(rows[66], start_col="J", end_col="CJ")
        records: List[Dict[str, object]] = []

        for row_num in range(147, 184):
            row = rows.get(row_num, {})
            property_name = str(row.get("B", "")).strip()
            unit = str(row.get("C", "")).strip()
            source = str(row.get("E", "")).strip()
            region = str(row.get("F", "")).strip()
            contract_type = str(row.get("G", "")).strip()
            weather_year = row.get("H")
            technology = TECH_NORMALIZATION.get(str(row.get("I", "")).strip(), "")
            if not property_name or not region or weather_year in (None, ""):
                continue

            for col, month in date_columns:
                value = row.get(col)
                if value in (None, ""):
                    continue
                records.append(
                    {
                        "property": property_name.lower().replace("&", "and"),
                        "unit": unit,
                        "source_scope": source,
                        "region": region.lower(),
                        "contract_type": contract_type,
                        "weather_year": int(float(weather_year)),
                        "technology": technology,
                        "month": month,
                        "value": float(value),
                    }
                )

        return pd.DataFrame(records)

    def build_monthly_market_inputs(
        self,
        region: JapanRegion,
        dashboard_inputs: pd.DataFrame,
    ) -> pd.DataFrame:
        selected_region = region.value
        market_side = REGION_TO_MARKET_SIDE[selected_region].lower()
        region_inputs = dashboard_inputs[dashboard_inputs["region"] == selected_region].copy()
        if region_inputs.empty:
            available = sorted(dashboard_inputs["region"].dropna().unique().tolist())
            raise ValueError(
                f"Workbook cache does not contain dashboard inputs for region '{selected_region}'. "
                f"Available cached regions: {available}"
            )

        demand = region_inputs[region_inputs["property"] == "demand"][["month", "contract_type", "weather_year", "value"]]
        demand = demand.rename(columns={"value": "demand_mw"})

        base = demand.copy()
        base["region"] = selected_region
        base["market_side"] = REGION_TO_MARKET_SIDE[selected_region]

        property_sources = {
            "availability": region_inputs[region_inputs["property"] == "availability"].copy(),
            "generation cost": dashboard_inputs[
                (dashboard_inputs["region"] == market_side)
                & (dashboard_inputs["property"] == "generation cost")
            ].copy(),
            "vo&m": dashboard_inputs[
                (dashboard_inputs["region"] == market_side)
                & (dashboard_inputs["property"] == "vo&m")
            ].copy(),
        }

        for property_name, prefix in (
            ("availability", "availability"),
            ("generation cost", "cost_jpy_mwh"),
            ("vo&m", "vom_jpy_mwh"),
        ):
            subset = property_sources[property_name]
            if subset.empty:
                continue
            pivoted = subset.pivot_table(
                index=["month", "contract_type", "weather_year"],
                columns="technology",
                values="value",
                aggfunc="first",
            )
            pivoted = pivoted.rename(columns=lambda col: f"{prefix}_{col}")
            base = base.merge(
                pivoted.reset_index(),
                on=["month", "contract_type", "weather_year"],
                how="left",
            )

        return base.sort_values("month").reset_index(drop=True)

    def extract_contract_calendar(self) -> pd.DataFrame:
        if not self.contract_calendar_path:
            return pd.DataFrame()

        df = pd.read_csv(self.contract_calendar_path, encoding="cp932")
        column_map = {
            "限月/Contract Month": "contract_month",
            "ベースロード電力 取引単位/Contract unit of Baseload Electricity(kWh)": "baseload_contract_unit_kwh",
            "ベースロード電力 暦日数/Calendar days of Baseload Electricity (days)": "baseload_calendar_days",
            "ベースロード電力 取引開始日/First Trading Day of Baseload Electricity(YYYYMMDD)": "baseload_first_trading_day",
            "ベースロード電力 取引最終日/Last Trading Day of Baseload Electricity(YYYYMMDD)": "baseload_last_trading_day",
            "ベースロード電力 最終決済日/Final Settlement Day of Baseload Electricity(YYYYMMDD)": "baseload_final_settlement_day",
            "日中ロード電力 取引単位/Contract unit of Peakload Electricity(kWh)": "peakload_contract_unit_kwh",
            "日中ロード電力 平日数/Weekdays of Peakload Electricity (days)": "peakload_weekdays",
            "日中ロード電力 取引開始日/First Trading Day of Peakload Electricity(YYYYMMDD)": "peakload_first_trading_day",
            "日中ロード電力 取引最終日/Last Trading Day of Peakload Electricity(YYYYMMDD)": "peakload_last_trading_day",
            "日中ロード電力 最終決済日/Final Settlement Day of Peakload Electricity(YYYYMMDD)": "peakload_final_settlement_day",
            "取引の対象となる期間の開始日(週間物取引)/Start Day of the Period covered by the Trading(Weekly Contracts)(YYYYMMDD)": "weekly_period_start",
            "取引の対象となる期間の終了日(週間物取引)/End Day of the Period covered by the Trading(Weekly Contracts)(YYYYMMDD)": "weekly_period_end",
            "備考及びTOCOM非営業日以外の休日相当の日/Notes and Days Treated as non-weekdays other than TOCOM non-business days": "notes",
        }
        df = df.rename(columns=column_map)

        date_cols = [
            "baseload_first_trading_day",
            "baseload_last_trading_day",
            "baseload_final_settlement_day",
            "peakload_first_trading_day",
            "peakload_last_trading_day",
            "peakload_final_settlement_day",
            "weekly_period_start",
            "weekly_period_end",
        ]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].astype(str), format="%Y%m%d", errors="coerce")

        if "contract_month" in df.columns:
            contract_month_str = df["contract_month"].astype(str)
            monthly_mask = contract_month_str.str.fullmatch(r"\d{6}")
            df["contract_month_start"] = pd.NaT
            df.loc[monthly_mask, "contract_month_start"] = pd.to_datetime(
                contract_month_str[monthly_mask],
                format="%Y%m",
                errors="coerce",
            )

        return df

    def _load_workbook_metadata(self) -> None:
        if self._sheet_targets:
            return

        with ZipFile(BytesIO(self._read_workbook_bytes())) as workbook:
            if "xl/sharedStrings.xml" in workbook.namelist():
                shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
                self._shared_strings = []
                for string_item in shared_root:
                    texts = [node.text or "" for node in string_item.iter(f"{{{MAIN_NS}}}t")]
                    self._shared_strings.append("".join(texts))

            workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
            rel_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
            rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root}

            for sheet in workbook_root.find("a:sheets", NS):
                rel_id = sheet.attrib[f"{{{REL_NS}}}id"]
                self._sheet_targets[sheet.attrib["name"]] = f"xl/{rel_map[rel_id]}"

    def _load_sheet_rows(self, sheet_name: str) -> Dict[int, Dict[str, object]]:
        if sheet_name in self._sheet_cache:
            return self._sheet_cache[sheet_name]

        if not self._sheet_targets:
            self._load_workbook_metadata()

        rows_by_number: Dict[int, Dict[str, object]] = {}
        with ZipFile(BytesIO(self._read_workbook_bytes())) as workbook:
            root = ET.fromstring(workbook.read(self._sheet_targets[sheet_name]))
            sheet_data = root.find("a:sheetData", NS)

            for row in sheet_data.findall("a:row", NS):
                row_number = int(row.attrib.get("r", "0"))
                values: Dict[str, object] = {}
                for cell in row.findall("a:c", NS):
                    cell_ref = cell.attrib.get("r", "")
                    col = "".join(ch for ch in cell_ref if ch.isalpha())
                    values[col] = self._parse_cell_value(cell)
                rows_by_number[row_number] = values

        self._sheet_cache[sheet_name] = rows_by_number
        return rows_by_number

    def _read_workbook_bytes(self) -> bytes:
        try:
            return self.workbook_path.read_bytes()
        except PermissionError:
            command = (
                "$path = @'\n"
                f"{self.workbook_path}\n"
                "'@.Trim(); "
                "$fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open, "
                "[System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite); "
                "try { "
                "$bytes = New-Object byte[] $fs.Length; "
                "[void]$fs.Read($bytes, 0, $fs.Length); "
                "[Convert]::ToBase64String($bytes) "
                "} finally { $fs.Dispose() }"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                check=True,
            )
            return base64.b64decode(result.stdout.strip())

    def _parse_cell_value(self, cell: ET.Element) -> object:
        value_node = cell.find("a:v", NS)
        if value_node is None:
            return ""
        raw_value = value_node.text
        if cell.attrib.get("t") == "s":
            return self._shared_strings[int(raw_value)]
        try:
            if raw_value is None:
                return ""
            if "." in raw_value:
                return float(raw_value)
            return int(raw_value)
        except ValueError:
            return raw_value

    def _extract_month_columns(
        self,
        header_row: Dict[str, object],
        start_col: str,
        end_col: str,
    ) -> List[Tuple[str, pd.Timestamp]]:
        start = _col_to_num(start_col)
        end = _col_to_num(end_col)
        month_columns: List[Tuple[str, pd.Timestamp]] = []
        for col, value in header_row.items():
            col_num = _col_to_num(col)
            if start <= col_num <= end and value not in (None, ""):
                month = _excel_serial_to_timestamp(value).normalize()
                month_columns.append((col, month))
        month_columns.sort(key=lambda item: item[1])
        return month_columns
