"""Fetch and prepare CAMELS-US / USGS 01542810 daily hydrometeorological dataset.

Data sources:
  1. USGS NWIS Daily Values API (parameterCd=00060, discharge in cfs)
  2. Open-Meteo Historical Archive (ERA5-Land daily meteorological forcings & FAO-56 ET0)
Unit conversions verified against API metadata:
  - Q_mm/d = Q_cfs * 2.4465755455 / area_km2
  - wind_speed_10m_mean (km/h) -> u10 (m/s) by / 3.6
  - relative_humidity_2m_mean (%) -> RH ([0, 1]) by / 100.0
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from src.utils import p, save_json

PUBLIC_FEATURES = [
    "P", "Tmax", "Tmin", "Tmean", "T_range", "Rs", "u10", "RH", "VPD", "ET0",
    "API_3", "API_7", "API_15", "API_30", "PmE_7", "PmE_30", "sin_doy", "cos_doy",
]


def _fetch_json(url: str, retries: int = 4) -> dict:
    last_err = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "xinshui-pinn-hydro/0.1"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (2 ** k))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def _es(t: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def lyne_hollick_bfi(q: np.ndarray, alpha: float = 0.925, passes: int = 3) -> float:
    """Estimate Baseflow Index via 3-pass Lyne-Hollick digital filter."""
    bf = q.copy().astype(float)
    for p_idx in range(passes):
        arr = bf if p_idx % 2 == 0 else bf[::-1]
        qf = np.zeros_like(arr)
        qb = np.zeros_like(arr)
        qb[0] = arr[0]
        for i in range(1, len(arr)):
            qf[i] = alpha * qf[i - 1] + 0.5 * (1 + alpha) * (arr[i] - arr[i - 1])
            qf[i] = max(0.0, min(qf[i], arr[i]))
            qb[i] = arr[i] - qf[i]
        bf = qb if p_idx % 2 == 0 else qb[::-1]
    return float(np.sum(np.clip(bf, 0.0, q)) / max(float(np.sum(q)), 1e-12))


def fetch_and_prepare_camels(cfg: dict) -> tuple[pd.DataFrame, dict]:
    ws = cfg["watershed"]
    sid = ws["site_id"]
    area_km2 = float(ws["area_km2"])
    start_date = cfg["split"]["warmup"][0]
    end_date = cfg["split"]["test"][1]

    out_dir = p("data/public_camels")
    out_dir.mkdir(parents=True, exist_ok=True)
    usgs_csv = out_dir / f"{sid}_usgs_daily.csv"
    meteo_csv = out_dir / f"{sid}_openmeteo_daily.csv"

    if usgs_csv.exists():
        df_q = pd.read_csv(usgs_csv, parse_dates=["date"])
    else:
        u1 = (
            f"https://waterservices.usgs.gov/nwis/dv/?format=json&sites={sid}"
            f"&startDT={start_date}&endDT={end_date}&parameterCd=00060"
        )
        raw_q = _fetch_json(u1)
        vals = raw_q["value"]["timeSeries"][0]["values"][0]["value"]
        rows = [{"date": v["dateTime"][:10], "q_cfs": float(v["value"])} for v in vals]
        df_q = pd.DataFrame(rows)
        df_q["date"] = pd.to_datetime(df_q["date"])
        df_q.to_csv(usgs_csv, index=False)

    if meteo_csv.exists():
        df_m = pd.read_csv(meteo_csv, parse_dates=["date"])
    else:
        u2 = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={ws['lat']}&longitude={ws['lon']}"
            f"&start_date={start_date}&end_date={end_date}"
            "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "shortwave_radiation_sum,wind_speed_10m_mean,relative_humidity_2m_mean,et0_fao_evapotranspiration"
            "&timezone=UTC"
        )
        raw_m = _fetch_json(u2)
        d = raw_m["daily"]
        df_m = pd.DataFrame(d).rename(columns={"time": "date"})
        df_m["date"] = pd.to_datetime(df_m["date"])
        df_m.to_csv(meteo_csv, index=False)

    df = df_m.merge(df_q, on="date", how="inner").sort_values("date").set_index("date")
    expected_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    assert len(df) == expected_days, f"Expected {expected_days} days, got {len(df)}"
    assert not df.isna().any().any(), "Unexpected NaNs in raw public data"

    out = pd.DataFrame(index=df.index)
    out["P"] = df["precipitation_sum"].astype(float)
    out["Tmax"] = df["temperature_2m_max"].astype(float)
    out["Tmin"] = df["temperature_2m_min"].astype(float)
    out["Tmean"] = df["temperature_2m_mean"].astype(float)
    out["T_range"] = out["Tmax"] - out["Tmin"]
    out["Rs"] = df["shortwave_radiation_sum"].astype(float)
    out["u10"] = df["wind_speed_10m_mean"].astype(float) / 3.6
    out["RH"] = np.clip(df["relative_humidity_2m_mean"].astype(float) / 100.0, 0.0, 1.0)
    es = 0.5 * (_es(out["Tmax"].values) + _es(out["Tmin"].values))
    out["VPD"] = np.maximum(es * (1.0 - out["RH"].values), 0.0)
    out["ET0"] = np.maximum(df["et0_fao_evapotranspiration"].astype(float), 0.0)
    out["q_cfs"] = df["q_cfs"].astype(float)
    out["Q_obs"] = out["q_cfs"] * 2.4465755455 / area_km2

    P_arr = out["P"].values
    for k in [3, 7, 15, 30]:
        api = np.zeros_like(P_arr)
        for i in range(1, k + 1):
            api[i:] += (0.85 ** (i - 1)) * P_arr[:-i]
        out[f"API_{k}"] = api
    pme = out["P"] - out["ET0"]
    out["PmE_7"] = pme.rolling(7, min_periods=1).sum().shift(1).fillna(0.0)
    out["PmE_30"] = pme.rolling(30, min_periods=1).sum().shift(1).fillna(0.0)
    doy = out.index.dayofyear.values
    out["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)

    assert not out[PUBLIC_FEATURES + ["Q_obs"]].isna().any().any()
    out.round(6).to_csv(out_dir / f"{sid}_daily_dataset.csv")

    stats = out[PUBLIC_FEATURES + ["Q_obs"]].describe().T
    stats["skew"] = out[PUBLIC_FEATURES + ["Q_obs"]].skew()
    stats.to_csv(out_dir / "env_factor_stats.csv")

    cal_mask = (out.index >= cfg["split"]["calib"][0]) & (out.index <= cfg["split"]["calib"][1])
    Xc = out.loc[cal_mask, PUBLIC_FEATURES]
    yc = out.loc[cal_mask, "Q_obs"].values
    mi = mutual_info_regression(Xc.values, yc, random_state=0)
    rel_rows = []
    for f_name, m_val in zip(PUBLIC_FEATURES, mi):
        rel_rows.append({
            "factor": f_name,
            "pearson": float(np.corrcoef(Xc[f_name].values, yc)[0, 1]),
            "spearman": float(pd.Series(Xc[f_name].values).corr(pd.Series(yc), method="spearman")),
            "mutual_info": float(m_val),
        })
    pd.DataFrame(rel_rows).sort_values("mutual_info", ascending=False).to_csv(
        out_dir / "factor_target_relevance.csv", index=False
    )
    out[PUBLIC_FEATURES].corr(method="spearman").to_csv(out_dir / "factor_spearman_matrix.csv")

    ann = out.groupby(out.index.year).agg(
        P=("P", "sum"), ET0=("ET0", "sum"), Tmean=("Tmean", "mean"), Q_obs=("Q_obs", "sum")
    )
    ann["RC"] = ann["Q_obs"] / ann["P"]
    ann.index.name = "year"

    ny = len(out) / 365.25
    summary = {
        "site_id": sid,
        "name": ws["name"],
        "lat": ws["lat"],
        "lon": ws["lon"],
        "area_km2": area_km2,
        "area_sqmi": ws["area_sqmi"],
        "CN0": ws["CN0"],
        "n_days": int(len(out)),
        "start_date": str(out.index[0].date()),
        "end_date": str(out.index[-1].date()),
        "P_mm_yr": float(out["P"].sum() / ny),
        "ET0_mm_yr": float(out["ET0"].sum() / ny),
        "Q_obs_mm_yr": float(out["Q_obs"].sum() / ny),
        "runoff_coeff": float(out["Q_obs"].sum() / out["P"].sum()),
        "aridity_index": float(out["ET0"].sum() / out["P"].sum()),
        "bfi_lyne_hollick": lyne_hollick_bfi(out["Q_obs"].values),
        "annual_summary": ann.round(3).reset_index().to_dict(orient="records"),
    }
    save_json(summary, "data/public_camels/watershed_summary.json")
    return out, summary
