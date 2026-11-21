"""Basin forcing from the 43 SWAT weather stations.

* Voronoi (Thiessen) area weights clipped to the basin polygon (EPSG:32649)
* Station-level FAO-56 Penman-Monteith ET0 (Allen et al., 1998) and VPD
* Storm heterogeneity factors, antecedent precipitation indices, seasonality
NOTE: `hmd` is a fraction in [0, 1] (asserted), NOT a percentage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import MultiPoint, Point
from shapely.ops import voronoi_diagram

from src.utils import p


def voronoi_weights(stations: pd.DataFrame, basin_geom) -> pd.Series:
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    x, y = tr.transform(stations["longitude"].values, stations["latitude"].values)
    pts = MultiPoint([Point(a, b) for a, b in zip(x, y)])
    env = basin_geom.envelope.buffer(50_000)
    cells = voronoi_diagram(pts, envelope=env)
    w = np.zeros(len(stations))
    for cell in cells.geoms:
        inter = cell.intersection(basin_geom)
        if inter.is_empty:
            continue
        for i, (a, b) in enumerate(zip(x, y)):
            if cell.contains(Point(a, b)):
                w[i] += inter.area
                break
    w = w / w.sum()
    return pd.Series(w, index=stations["station_id"].values, name="weight")


def _e0(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def fao56_et0(tmax, tmin, rs, u2, rh, lat_deg, z, doy):
    """FAO-56 Penman-Monteith reference ET (mm/d). All inputs broadcastable arrays."""
    tmean = 0.5 * (tmax + tmin)
    es = 0.5 * (_e0(tmax) + _e0(tmin))
    ea = np.clip(rh, 0.0, 1.0) * es
    delta = 4098 * _e0(tmean) / (tmean + 237.3) ** 2
    pres = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26
    gamma = 0.000665 * pres
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec)
                                            + np.cos(phi) * np.cos(dec) * np.sin(ws))
    rso = (0.75 + 2e-5 * z) * ra
    rns = 0.77 * rs
    ratio = np.clip(rs / np.maximum(rso, 1e-6), 0.25, 1.0)
    sigma = 4.903e-9
    rnl = sigma * 0.5 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) \
        * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * ratio - 0.35)
    rn = rns - rnl
    num = 0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)
    den = delta + gamma * (1 + 0.34 * u2)
    return np.maximum(num / den, 0.0), es - ea


def build_forcing(meta: dict) -> tuple[pd.DataFrame, dict]:
    paths = meta["paths"]
    wx = pd.read_csv(p(paths["weather"]), parse_dates=["date"])
    st = pd.read_csv(p(paths["stations"]))
    assert wx["hmd"].between(0, 1).all(), "hmd must be a fraction in [0,1]"
    assert not wx[["pcp", "tmp_max", "tmp_min", "slr", "wnd", "hmd"]].isna().any().any()

    basin = gpd.read_file(p(paths["basin"]))
    geom = basin.geometry.union_all()
    w = voronoi_weights(st, geom)

    wx = wx.merge(st[["station_id", "elevation_m"]], on="station_id", how="left")
    doy = wx["date"].dt.dayofyear.values
    # SWAT wind is used as ~2 m wind (SWAT convention); no height conversion applied
    et0, vpd = fao56_et0(wx["tmp_max"].values, wx["tmp_min"].values, wx["slr"].values,
                         wx["wnd"].values, wx["hmd"].values, wx["latitude"].values,
                         wx["elevation_m"].values, doy)
    wx["et0"] = et0
    wx["vpd"] = vpd
    wx["w"] = wx["station_id"].map(w).fillna(0.0)

    vars_ = {"pcp": "P", "tmp_max": "Tmax", "tmp_min": "Tmin", "slr": "Rs", "wnd": "u2",
             "hmd": "RH", "et0": "ET0", "vpd": "VPD"}
    wsum = wx.groupby("date")["w"].sum()
    df = pd.DataFrame(index=wsum.index)
    for c, name in vars_.items():
        df[name] = (wx[c] * wx["w"]).groupby(wx["date"]).sum() / wsum
    df["Tmean"] = 0.5 * (df["Tmax"] + df["Tmin"])

    # storm heterogeneity: stations inside basin (w>0) split into elevation terciles
    inside = st[st["station_id"].map(w) > 0].copy()
    q1, q2 = inside["elevation_m"].quantile([1 / 3, 2 / 3])
    grp = {sid: ("P_low" if e <= q1 else "P_mid" if e <= q2 else "P_up")
           for sid, e in zip(inside["station_id"], inside["elevation_m"])}
    pin = wx[wx["station_id"].isin(grp)].copy()
    pin["grp"] = pin["station_id"].map(grp)
    piv = pin.pivot_table(index="date", columns="grp", values="pcp", aggfunc="mean")
    for g in ["P_up", "P_mid", "P_low"]:
        df[g] = piv[g]
    pall = pin.pivot_table(index="date", columns="station_id", values="pcp")
    df["P_max_sta"] = pall.max(axis=1)
    df["P_std_sta"] = pall.std(axis=1)
    df["f_rain"] = (pall >= 1.0).mean(axis=1)

    # antecedent indices and water deficit
    P = df["P"].values
    for k in [3, 7, 15, 30]:
        api = np.zeros_like(P)
        for i in range(1, k + 1):
            api[i:] += 0.85 ** (i - 1) * P[:-i]
        df[f"API_{k}"] = api
    pme = df["P"] - df["ET0"]
    df["PmE_7"] = pme.rolling(7, min_periods=1).sum().shift(1).fillna(0.0)
    df["PmE_30"] = pme.rolling(30, min_periods=1).sum().shift(1).fillna(0.0)
    d = df.index.dayofyear.values
    df["sin_doy"] = np.sin(2 * np.pi * d / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * d / 365.25)
    df.index.name = "date"

    info = {"station_weights": w.to_dict(), "n_stations_inside": int((w > 0).sum()),
            "weight_sum": float(w.sum()),
            "elev_tercile_bounds_m": [float(q1), float(q2)],
            "wind_height_note": "SWAT wnd used directly as u2 (no height conversion)"}
    return df, info


FEATURES_ALL = ["P", "Tmax", "Tmin", "Tmean", "Rs", "u2", "RH", "VPD", "ET0",
                "P_up", "P_mid", "P_low", "P_max_sta", "P_std_sta", "f_rain",
                "API_3", "API_7", "API_15", "API_30", "PmE_7", "PmE_30", "sin_doy", "cos_doy"]
STORM_FEATURES = ["P_up", "P_mid", "P_low", "P_max_sta", "P_std_sta", "f_rain"]
