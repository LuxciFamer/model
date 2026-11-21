"""Phase 1: GIS static attributes, 43-station basin forcing + FAO-56 ET0, factor statistics,
and (if no observations exist and allowed) the SYNTHETIC twin runoff for smoke testing."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_selection import mutual_info_regression  # noqa: E402

from src.data.gis_extractor import extract_static  # noqa: E402
from src.data.meteo_et0 import FEATURES_ALL, build_forcing  # noqa: E402
from src.data.synthetic import make_twin  # noqa: E402
from src.data.dataset import FEAT_CSV, load_runoff  # noqa: E402
from src.utils import load_cfg, load_meta, p, save_json  # noqa: E402


def main():
    meta, cfg = load_meta(), load_cfg()
    (p("data/processed")).mkdir(parents=True, exist_ok=True)
    (p("data/external")).mkdir(parents=True, exist_ok=True)

    print("[1/4] GIS static attributes ...")
    static = extract_static(meta)
    print(f"      CN0={static['CN0']:.1f}  Smax0={static['Smax0_mm']:.0f} mm  "
          f"slope={static['slope_mean_deg']:.1f} deg  LULC={static['lulc_fractions']}")

    print("[2/4] Basin forcing & FAO-56 ET0 ...")
    df, winfo = build_forcing(meta)
    static["weather"] = winfo
    assert len(df) == 3287 and not df[FEATURES_ALL].isna().any().any(), "feature table incomplete"
    ann = df.groupby(df.index.year).agg(P=("P", "sum"), ET0=("ET0", "sum"), T=("Tmean", "mean"))
    static["annual_climate"] = ann.round(2).reset_index().to_dict(orient="records")
    print(ann.round(1).to_string())

    obs_exists = p(meta["paths"]["observed_daily"]).exists() or p(meta["paths"]["observed_monthly"]).exists()
    if not obs_exists:
        if not cfg.get("allow_synthetic"):
            raise SystemExit("P0 observed runoff missing and allow_synthetic=false -> stop.")
        print("[3/4] !!! No observed runoff: generating SYNTHETIC twin runoff (smoke test only) ...")
        syn, sinfo = make_twin(df, static["CN0"], meta["gauge"]["drainage_area_km2"])
        p(meta["paths"]["synthetic_daily"]).parent.mkdir(parents=True, exist_ok=True)
        syn.to_csv(p(meta["paths"]["synthetic_daily"]), index=False)
        save_json(sinfo, "data/synthetic/SYNTHETIC_twin_info.json")
        print(f"      runoff coefficient={sinfo['runoff_coefficient_2016_2023']:.3f}, "
              f"mean={sinfo['mean_runoff_mm_per_yr']:.1f} mm/yr")
    run = load_runoff(meta, cfg)
    static["runoff_source"] = run["source"]; static["runoff_resolution"] = run["resolution"]

    print("[4/4] Factor statistics / correlation / mutual information ...")
    df.round(6).to_csv(p(FEAT_CSV))
    stats = df[FEATURES_ALL].describe().T
    stats["skew"] = df[FEATURES_ALL].skew()
    stats.to_csv(p("data/processed/env_factor_stats.csv"))
    if run["daily"] is not None:
        q = run["daily"].reindex(df.index)
        cal = (df.index >= cfg["split"]["calib"][0]) & (df.index <= cfg["split"]["calib"][1]) & q.notna().values
        X, y = df.loc[cal, FEATURES_ALL], q[cal].values
        rows = []
        mi = mutual_info_regression(X.values, y, random_state=0)
        for f, m in zip(FEATURES_ALL, mi):
            rows.append({"factor": f, "pearson": float(np.corrcoef(X[f], y)[0, 1]),
                         "spearman": float(pd.Series(X[f].values).corr(pd.Series(y), method="spearman")),
                         "mutual_info": float(m)})
        pd.DataFrame(rows).sort_values("mutual_info", ascending=False).to_csv(
            p("data/processed/factor_target_relevance.csv"), index=False)
    df[FEATURES_ALL].corr(method="spearman").to_csv(p("data/processed/factor_spearman_matrix.csv"))
    save_json(static, "data/processed/xinshui_static_attributes.json")
    print("done ->", FEAT_CSV)


if __name__ == "__main__":
    main()
