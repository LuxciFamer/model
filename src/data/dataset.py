"""Dataset assembly: features, runoff target (observed or synthetic), leakage-free splits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import p

FEAT_CSV = "data/processed/xinshui_daily_features.csv"


def load_features() -> pd.DataFrame:
    return pd.read_csv(p(FEAT_CSV), parse_dates=["date"], index_col="date")


def _to_mm(df: pd.DataFrame, area_km2: float, prefix: str) -> pd.Series:
    if f"{prefix}_mm" in df.columns:
        return df[f"{prefix}_mm"].astype(float)
    if f"{prefix}_m3s" in df.columns:
        return df[f"{prefix}_m3s"].astype(float) * 86.4 / area_km2
    raise ValueError(f"runoff file needs column {prefix}_mm or {prefix}_m3s")


def load_runoff(meta: dict, cfg: dict) -> dict:
    """Return {'source','resolution','daily': Series|None,'monthly': Series|None}."""
    area = float(meta["gauge"]["drainage_area_km2"])
    paths = meta["paths"]
    if p(paths["observed_daily"]).exists():
        d = pd.read_csv(p(paths["observed_daily"]), parse_dates=["date"]).set_index("date")
        return {"source": "observed", "resolution": "daily", "daily": _to_mm(d, area, "q_obs"),
                "monthly": None, "area_km2": area}
    if p(paths["observed_monthly"]).exists():
        m = pd.read_csv(p(paths["observed_monthly"]))
        m["date"] = pd.to_datetime(m["year_month"] + "-01")
        m = m.set_index("date")
        q = _to_mm(m, area, "q_obs")  # mean daily depth (mm/d) of the month
        return {"source": "observed", "resolution": "monthly", "daily": None, "monthly": q,
                "area_km2": area}
    if cfg.get("allow_synthetic") and p(paths["synthetic_daily"]).exists():
        d = pd.read_csv(p(paths["synthetic_daily"]), parse_dates=["date"]).set_index("date")
        return {"source": "SYNTHETIC", "resolution": "daily", "daily": _to_mm(d, area, "q_obs"),
                "monthly": None, "area_km2": area}
    raise FileNotFoundError(
        "No P0 runoff found. Put data/external/xinshui_observed_runoff.csv "
        "(date,q_obs_m3s) or the monthly file; or enable synthetic twin data.")


def load_swat(meta: dict, index: pd.DatetimeIndex) -> pd.Series | None:
    path = p(meta["paths"]["swat_simulated"])
    if not path.exists():
        return None
    d = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return _to_mm(d, float(meta["gauge"]["drainage_area_km2"]), "q_swat").reindex(index)


def period_mask(index: pd.DatetimeIndex, period) -> np.ndarray:
    a, b = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    return np.asarray((index >= a) & (index <= b))


class FeatureScaler:
    """Z-score scaler fitted ONLY on calibration rows."""

    def fit(self, X: np.ndarray, mask: np.ndarray):
        self.mean_ = X[mask].mean(axis=0)
        self.scale_ = X[mask].std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_


def monthly_index(index: pd.DatetimeIndex) -> np.ndarray:
    ym = index.year * 12 + index.month
    return (ym - ym.min()).astype(int)
