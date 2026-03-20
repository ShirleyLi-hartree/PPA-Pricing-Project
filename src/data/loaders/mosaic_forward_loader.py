"""
Loader for historical Japan power forward curves from the Mosaic curves API.

This standardizes curve panels into a tabular format that can be joined with
monthly theoretical pricing outputs for forward validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://curves-api.mosaic.hartreepartners.com/api/v1"


@dataclass(frozen=True)
class ForwardCurveSpec:
    alias: str
    region: str
    load_shape: str
    tenor: str
    source: str = "mosaic_curves_api"


TOKYO_BASELOAD_MONTHLY = ForwardCurveSpec(
    alias="JAPAN-BASE-POWER-MONTH-TOKYO-FOBM",
    region="tokyo",
    load_shape="baseload",
    tenor="monthly",
)


class MosaicForwardCurveLoader:
    """Fetch and standardize historical forward curve panels from Mosaic."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")

    def fetch_panel(
        self,
        curve_spec: ForwardCurveSpec,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch raw panel data from the API."""
        url = (
            f"{self.base_url}/evalPanel"
            f"?alias={curve_spec.alias}&start_date={start_date}&end_date={end_date}"
        )
        session = requests.Session()
        # Ignore broken local proxy settings in sandboxed environments.
        session.trust_env = False
        response = session.get(url, timeout=120, verify=False)
        response.raise_for_status()
        raw_data = response.json()
        if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict) and "result" in raw_data[0]:
            raw_data = raw_data[0]["result"]
        return pd.DataFrame(raw_data)

    def standardize_panel(
        self,
        raw_panel: pd.DataFrame,
        curve_spec: ForwardCurveSpec,
    ) -> pd.DataFrame:
        """Standardize raw API output into a historical forward curve table."""
        if raw_panel.empty:
            return pd.DataFrame(
                columns=[
                    "observation_date",
                    "delivery_month",
                    "contract",
                    "rank",
                    "forward_price_jpy_kwh",
                    "region",
                    "load_shape",
                    "tenor",
                    "source",
                    "source_alias",
                ]
            )

        df = raw_panel.copy()
        df["observation_date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["delivery_month"] = pd.to_datetime(df["contract"].astype(str), format="%Y%m", errors="coerce")
        df["contract"] = df["contract"].astype(str)
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
        df["forward_price_jpy_kwh"] = pd.to_numeric(df["value"], errors="coerce")
        df["region"] = curve_spec.region
        df["load_shape"] = curve_spec.load_shape
        df["tenor"] = curve_spec.tenor
        df["source"] = curve_spec.source
        df["source_alias"] = curve_spec.alias
        return df[
            [
                "observation_date",
                "delivery_month",
                "contract",
                "rank",
                "forward_price_jpy_kwh",
                "region",
                "load_shape",
                "tenor",
                "source",
                "source_alias",
            ]
        ].sort_values(["observation_date", "delivery_month"]).reset_index(drop=True)

    def fetch_and_standardize(
        self,
        curve_spec: ForwardCurveSpec,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        raw_panel = self.fetch_panel(curve_spec, start_date, end_date)
        return self.standardize_panel(raw_panel, curve_spec)

    def save_standardized_panel(
        self,
        standardized_panel: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        standardized_panel.to_csv(output, index=False)
        return output
