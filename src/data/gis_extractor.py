"""Static physiographic attributes and physics priors from the SWAT GIS layers.

Reads ONLY from Xinshui_SWAT_Inputs/projected (EPSG:32649) and never writes there.
Outputs: slope/elevation/TWI statistics, LULC & soil-texture fractions, area-weighted
SCS curve number CN0 (USDA-NRCS TR-55 style table) and Saxton & Rawls (2006)
soil-hydraulic priors (Ks, AWC, Smax0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling

from src.utils import p

# USDA texture -> hydrologic soil group (NRCS NEH-630 ch.7 texture guidance)
TEXTURE_HSG = {"S": "A", "LS": "A", "SL": "B", "L": "B", "SIL": "B", "SI": "B",
               "SCL": "C", "CL": "D", "SICL": "D", "SC": "D", "SIC": "D", "C": "D"}
# Texture-class centroid sand / clay fractions (USDA texture triangle)
TEXTURE_SAND_CLAY = {"S": (0.92, 0.03), "LS": (0.82, 0.06), "SL": (0.65, 0.10),
                     "L": (0.40, 0.20), "SIL": (0.20, 0.15), "SI": (0.07, 0.06),
                     "SCL": (0.60, 0.27), "CL": (0.32, 0.34), "SICL": (0.10, 0.34),
                     "SC": (0.52, 0.42), "SIC": (0.07, 0.47), "C": (0.20, 0.60)}
# CN2 (AMC II) by SWAT land-use code and HSG (TR-55 tables, representative cover/condition)
CN_TABLE = {"FRSD": (30, 55, 70, 77), "RNGB": (35, 56, 70, 77), "PAST": (49, 69, 79, 84),
            "AGRL": (67, 78, 85, 89), "URBN": (77, 85, 90, 92), "BARR": (77, 86, 91, 94),
            "WATR": (98, 98, 98, 98), "WETL": (78, 78, 78, 78)}
HSG_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def saxton_rawls(sand: float, clay: float, om: float = 1.5) -> dict:
    """Saxton & Rawls (2006) pedotransfer functions. sand/clay as fractions, OM in %."""
    S, C, OM = sand, clay, om
    t1500 = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
    th1500 = t1500 + (0.14 * t1500 - 0.02)
    t33 = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
    th33 = t33 + (1.283 * t33 ** 2 - 0.374 * t33 - 0.015)
    ts33 = 0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * S * OM - 0.027 * C * OM - 0.584 * S * C + 0.078
    ths33 = ts33 + (0.636 * ts33 - 0.107)
    ths = th33 + ths33 - 0.097 * S + 0.043
    B = (np.log(1500) - np.log(33)) / (np.log(th33) - np.log(th1500))
    lam = 1.0 / B
    ks = 1930.0 * (ths - th33) ** (3 - lam)  # mm/h
    return {"theta_wp": th1500, "theta_fc": th33, "theta_sat": ths, "Ks_mmh": ks}


def _read_masked(path, geoms):
    with rasterio.open(path) as src:
        arr, tr = rio_mask(src, geoms, crop=True, filled=False)
        prof = src.profile.copy()
        prof.update(transform=tr, height=arr.shape[1], width=arr.shape[2])
        res = src.res
    return arr[0], prof, res


def _horn_slope_deg(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    zp = np.pad(z, 1, mode="edge")
    a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    d, f = zp[1:-1, :-2], zp[1:-1, 2:]
    g, h, i = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def _d8_twi(z: np.ndarray, valid: np.ndarray, cell: float) -> np.ndarray:
    """Approximate TWI = ln(a / tan b) with D8 accumulation (no pit filling)."""
    H, W = z.shape
    zz = np.where(valid, z, np.inf)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    zp = np.pad(zz, 1, constant_values=np.inf)
    best_drop = np.zeros_like(z, dtype=float)
    best_k = np.full(z.shape, -1, dtype=int)
    for k, (di, dj) in enumerate(offs):
        nb = zp[1 + di:1 + di + H, 1 + dj:1 + dj + W]
        dist = cell * (np.sqrt(2) if di and dj else 1.0)
        drop = np.where(np.isfinite(nb), (zz - nb) / dist, -np.inf)
        upd = drop > best_drop
        best_drop[upd] = drop[upd]
        best_k[upd] = k
    flat = np.flatnonzero(valid.ravel())
    order = flat[np.argsort(-z.ravel()[flat])]
    acc = np.where(valid, 1.0, 0.0).ravel()
    bk = best_k.ravel()
    for idx in order.tolist():
        k = bk[idx]
        if k < 0:
            continue
        di, dj = offs[k]
        j = idx + di * W + dj
        acc[j] += acc[idx]
    acc = acc.reshape(H, W)
    slope = np.maximum(best_drop, 1e-4)
    twi = np.log(acc * cell / slope)
    return np.where(valid, twi, np.nan)


def extract_static(meta: dict) -> dict:
    paths = meta["paths"]
    basin = gpd.read_file(p(paths["basin"]))
    geoms = list(basin.geometry)
    area_poly = float(basin.geometry.area.sum() / 1e6)

    # ---- DEM ----
    dem, dprof, dres = _read_masked(p(paths["dem"]), geoms)
    z = dem.filled(np.nan).astype(float)
    nod = dprof.get("nodata")
    if nod is not None:
        z[z == nod] = np.nan
    valid = np.isfinite(z)
    slope = _horn_slope_deg(np.where(valid, z, np.nanmean(z)), dres[0], dres[1])
    slope = np.where(valid, slope, np.nan)
    # TWI on 3x aggregated DEM (~90 m) for tractability
    f = 3
    H, W = (z.shape[0] // f) * f, (z.shape[1] // f) * f
    zc = z[:H, :W].reshape(H // f, f, W // f, f)
    with np.errstate(all="ignore"):
        zc = np.nanmean(zc, axis=(1, 3))
    twi = _d8_twi(zc, np.isfinite(zc), dres[0] * f)
    zv = z[valid]
    out = {
        "basin_polygon_area_km2": area_poly,
        "dem_resolution_m": float(dres[0]),
        "elev_mean_m": float(zv.mean()), "elev_min_m": float(zv.min()), "elev_max_m": float(zv.max()),
        "elev_p05_m": float(np.percentile(zv, 5)), "elev_p10_m": float(np.percentile(zv, 10)),
        "elev_p50_m": float(np.percentile(zv, 50)), "elev_p90_m": float(np.percentile(zv, 90)),
        "elev_p95_m": float(np.percentile(zv, 95)),
        "slope_mean_deg": float(np.nanmean(slope)), "slope_p50_deg": float(np.nanmedian(slope)),
        "slope_gt15deg_frac": float(np.nanmean(slope[valid] > 15)),
        "twi_mean": float(np.nanmean(twi)), "twi_p50": float(np.nanmedian(twi)),
        "twi_method": "D8 on 3x-aggregated DEM, no pit filling (approximation)",
    }

    # ---- rivers ----
    try:
        riv = gpd.read_file(p(paths["rivers"]))
        out["river_length_km_geom"] = float(riv.geometry.length.sum() / 1e3)
        if "channel_slope" in riv.columns:
            out["channel_slope"] = float(riv["channel_slope"].iloc[0])
    except Exception:  # noqa: BLE001
        pass

    # ---- LULC ----
    lulc_lk = pd.read_csv(p(paths["lulc_lookup"]))
    code_map = dict(zip(lulc_lk.iloc[:, 0].astype(int), lulc_lk.iloc[:, 1].astype(str).str.strip()))
    lulc, lprof, _ = _read_masked(p(paths["lulc"]), geoms)
    lmask = ~np.ma.getmaskarray(lulc)
    lv = np.asarray(lulc.filled(0))
    lnod = lprof.get("nodata")
    if lnod is not None:
        lmask &= lv != lnod
    # WorldCover raw classes may be 10..100; map unknown classes to nearest lookup key
    lulc_code = np.full(lv.shape, "", dtype=object)
    for k, v in code_map.items():
        lulc_code[lv == k] = v
    # WorldCover 95 (mangrove) / 100 (moss) -> treat as RNGB; 70 (snow) -> BARR
    lulc_code[lv == 100] = "RNGB"
    lulc_code[lv == 95] = "WETL"
    lulc_code[lv == 70] = "BARR"
    known = lmask & (lulc_code != "")
    codes, cnt = np.unique(lulc_code[known], return_counts=True)
    out["lulc_fractions"] = {str(c): float(n / cnt.sum()) for c, n in zip(codes, cnt)}
    out["lulc_raw_classes"] = {int(k): int(v) for k, v in zip(*np.unique(lv[lmask], return_counts=True))}

    # ---- Soil (reprojected onto LULC grid, nearest) ----
    soil_lk = pd.read_csv(p(paths["soil_lookup"]))
    smap = dict(zip(soil_lk.iloc[:, 0].astype(int), soil_lk.iloc[:, 1].astype(str).str.strip()))
    with rasterio.open(p(paths["soil"])) as ssrc:
        soil_on_lulc = np.full(lv.shape, -9999, dtype=np.int32)
        reproject(source=rasterio.band(ssrc, 1), destination=soil_on_lulc,
                  src_transform=ssrc.transform, src_crs=ssrc.crs,
                  dst_transform=lprof["transform"], dst_crs=lprof["crs"],
                  resampling=Resampling.nearest, dst_nodata=-9999)
        snod = ssrc.nodata
    smask = soil_on_lulc != -9999
    if snod is not None:
        smask &= soil_on_lulc != snod
    tex = np.full(lv.shape, "", dtype=object)
    for k, v in smap.items():
        tex[soil_on_lulc == k] = v
    tvalid = known & smask & (tex != "")
    tcodes, tcnt = np.unique(tex[tvalid], return_counts=True)
    tfrac = {str(c): float(n / tcnt.sum()) for c, n in zip(tcodes, tcnt)}
    out["soil_texture_fractions"] = tfrac

    # ---- CN0 (pixel-wise LULC x HSG) ----
    cn = np.full(lv.shape, np.nan)
    for lu, row in CN_TABLE.items():
        for t, hsg in TEXTURE_HSG.items():
            sel = tvalid & (lulc_code == lu) & (tex == t)
            cn[sel] = row[HSG_IDX[hsg]]
    out["CN0"] = float(np.nanmean(cn))
    out["CN0_p10"] = float(np.nanpercentile(cn, 10))
    out["CN0_p90"] = float(np.nanpercentile(cn, 90))
    hsg_frac = {}
    for t, fr in tfrac.items():
        hsg_frac[TEXTURE_HSG.get(t, "?")] = hsg_frac.get(TEXTURE_HSG.get(t, "?"), 0) + fr
    out["hsg_fractions"] = hsg_frac

    # ---- Saxton-Rawls priors (texture-fraction weighted) ----
    sand = sum(TEXTURE_SAND_CLAY[t][0] * fr for t, fr in tfrac.items() if t in TEXTURE_SAND_CLAY)
    clay = sum(TEXTURE_SAND_CLAY[t][1] * fr for t, fr in tfrac.items() if t in TEXTURE_SAND_CLAY)
    sr = {t: saxton_rawls(*TEXTURE_SAND_CLAY[t]) for t in tfrac if t in TEXTURE_SAND_CLAY}
    w = {t: tfrac[t] for t in sr}
    ks = float(np.exp(sum(np.log(sr[t]["Ks_mmh"]) * w[t] for t in sr)))  # geometric mean
    fc = sum(sr[t]["theta_fc"] * w[t] for t in sr)
    wp = sum(sr[t]["theta_wp"] * w[t] for t in sr)
    ths = sum(sr[t]["theta_sat"] * w[t] for t in sr)
    root_depth_mm = 1000.0  # assumed effective root-zone depth for Loess soils
    out.update({
        "sand_frac": sand, "clay_frac": clay, "silt_frac": 1 - sand - clay,
        "Ks0_mmh": ks, "theta_fc": fc, "theta_wp": wp, "theta_sat": ths,
        "AWC0_mm": float((fc - wp) * root_depth_mm),
        "Smax0_mm": float((fc - wp) * root_depth_mm),
        "root_depth_assumed_mm": root_depth_mm,
        "saxton_rawls_OM_pct": 1.5,
    })
    return out
