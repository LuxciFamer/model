"""Publication-grade 300 DPI PNG & SVG figures for Xinshui River Meta-PINN-Hydro."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import seaborn as sns

from src.physics.loess_hydro_np import PARAM_NAMES
from src.utils import p

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def _save(fig, stem: str, is_synthetic: bool = False):
    out_dir = p("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_synthetic:
        fig.text(
            0.5, -0.028,
            "注意：当前径流目标为合成孪生验证序列（SYNTHETIC TWIN SMOKE TEST），用于流程验证而非真实水文站率定结论",
            ha="center", va="top", fontsize=8.5, color="#B22222", style="italic",
        )
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_fig01_study_area(meta: dict, cfg: dict, df_feat: pd.DataFrame, pred_df: pd.DataFrame, is_syn: bool):
    basin = gpd.read_file(p(meta["paths"]["basin"]))
    rivers = gpd.read_file(p(meta["paths"]["rivers"]))
    st = pd.read_csv(p(meta["paths"]["stations"]))
    st_gdf = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["longitude"], st["latitude"]), crs="EPSG:4326").to_crs("EPSG:32649")

    with rasterio.open(p(meta["paths"]["dem"])) as src:
        scale = 6
        arr = src.read(1, out_shape=(1, src.height // scale, src.width // scale), resampling=Resampling.bilinear).astype(float)
        nod = src.nodata
        if nod is not None:
            arr[arr == nod] = np.nan
        arr[(arr <= 1.0) | (arr > 5000)] = np.nan
        b = src.bounds
        extent = [b.left / 1e3, b.right / 1e3, b.bottom / 1e3, b.top / 1e3]

    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1, 1], hspace=0.28, wspace=0.22)
    ax_map = fig.add_subplot(gs[:, 0])
    im = ax_map.imshow(arr, extent=extent, origin="upper", cmap="terrain", alpha=0.88)
    basin_km = basin.copy(); basin_km.geometry = basin_km.geometry.scale(1e-3, 1e-3, origin=(0, 0))
    riv_km = rivers.copy(); riv_km.geometry = riv_km.geometry.scale(1e-3, 1e-3, origin=(0, 0))
    st_km = st_gdf.copy(); st_km.geometry = st_km.geometry.scale(1e-3, 1e-3, origin=(0, 0))
    basin_km.boundary.plot(ax=ax_map, color="black", linewidth=1.6, label="流域边界")
    riv_km.plot(ax=ax_map, color="#1f77b4", linewidth=1.3, label="河网干支流")
    st_km.plot(ax=ax_map, color="#d62728", markersize=22, marker="^", edgecolor="white", linewidth=0.5, label="SWAT气象格点(43)")
    cbar = fig.colorbar(im, ax=ax_map, shrink=0.78, pad=0.03)
    cbar.set_label("高程 (m)")
    ax_map.set_title("昕水河流域地形、河网与43个气象站分布 (UTM 49N)", fontweight="bold")
    ax_map.set_xlabel("东距 Easting (km)"); ax_map.set_ylabel("北距 Northing (km)")
    ax_map.legend(loc="lower right", frameon=True, fontsize=8.5)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)
    ax1.bar(df_feat.index, df_feat["P"], width=2.0, color="#2b8cbe", alpha=0.65, label="面平均降雨 P")
    ax1_r.plot(df_feat.index, df_feat["ET0"], color="#e6550d", linewidth=0.7, alpha=0.8, label="FAO-56 ET0")
    ax1.invert_yaxis()
    ax1.set_ylabel("降雨 P (mm/d)", color="#2b8cbe")
    ax1_r.set_ylabel("潜在蒸散发 ET0 (mm/d)", color="#e6550d")
    ax1.set_title("2015-2023年流域强迫与径流过程及数据切分", fontweight="bold")

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    q_col = "obs" if "obs" in pred_df and pred_df["obs"].notna().any() else "Meta-PINN-Hydro"
    ax2.plot(pred_df.index, pred_df[q_col], color="#08519c", linewidth=0.9, label="目标径流 (mm/d)")
    sp = cfg["split"]
    cols = [("warmup", "#bdbdbd", "预热期"), ("calib", "#ccebc5", "率定期"),
            ("valid", "#fed9a6", "验证期"), ("test", "#fbb4ae", "独立测试期(含2021秋汛)")]
    for k, c, lbl in cols:
        t0, t1 = pd.Timestamp(sp[k][0]), pd.Timestamp(sp[k][1])
        ax2.axvspan(t0, t1, color=c, alpha=0.25, label=lbl)
    ax2.set_ylabel("径流深 (mm/d)"); ax2.set_xlabel("日期")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.legend(loc="upper left", ncol=3, fontsize=8, frameon=True)
    _save(fig, "fig01_xinshui_study_area_and_forcing", is_syn)


def plot_fig02_env_factors(corr_df: pd.DataFrame, rel_df: pd.DataFrame | None, grad_df: pd.DataFrame, is_syn: bool):
    key_cols = ["P", "Tmean", "ET0", "VPD", "Rs", "RH", "P_max_sta", "P_std_sta", "f_rain",
                "API_3", "API_15", "PmE_30", "sin_doy", "cos_doy"]
    cols = [c for c in key_cols if c in corr_df.columns]
    sub = corr_df.loc[cols, cols]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.30})
    sns.heatmap(sub, ax=axes[0], cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    axes[0].set_title("主要环境因子 Spearman 相关矩阵", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=40)

    merged = grad_df.copy()
    if rel_df is not None and len(rel_df) > 0:
        merged = merged.merge(rel_df[["factor", "mutual_info"]], on="factor", how="left")
    else:
        merged["mutual_info"] = 0.0
    top = merged.head(14).iloc[::-1]
    y = np.arange(len(top))
    h = 0.38
    axes[1].barh(y + h / 2, top["grad_importance_pct"], height=h, xerr=top["grad_importance_std"],
                 color="#2b8cbe", label="PINN 梯度敏感性 |∂Q/∂X| (%)")
    mi_pct = top["mutual_info"].values / max(float(merged["mutual_info"].sum()), 1e-12) * 100.0
    axes[1].barh(y - h / 2, mi_pct, height=h, color="#fdae6b", label="与径流互信息 MI 占比 (%)")
    axes[1].set_yticks(y); axes[1].set_yticklabels(top["factor"])
    axes[1].set_xlabel("相对重要性占比 (%)")
    axes[1].set_title("环境因子互信息与PINN梯度归因排序 (Top 14)", fontweight="bold")
    axes[1].legend(loc="lower right", frameon=True, fontsize=8.5)
    _save(fig, "fig02_env_factors_diagnostics", is_syn)


def plot_fig03_architecture(static: dict, is_syn: bool):
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, title, body, fc="#eef5fc", ec="#2b5c8f"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                      facecolor=fc, edgecolor=ec, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=ec)
        ax.text(x + w / 2, y + (h - 3.0) / 2, body, ha="center", va="center", fontsize=8.2, color="#222222", linespacing=1.35)

    def arrow(x1, y1, x2, y2, lbl="", color="#333333"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=2, shrinkB=2))
        if lbl:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, lbl, ha="center", va="bottom", fontsize=7.8, color=color)

    ax.text(50, 60.2, "Meta-PINN-Hydro 黄土高原混合产汇流物理信息神经网络架构", ha="center", va="top", fontsize=12.5, fontweight="bold")
    box(2, 34, 21, 21, "多源环境因子 X_t (23维)",
        f"• 43站加权强迫: P, T, ET0, VPD\n• 暴雨异质性: P_max, σ_P, f_rain\n• 前期记忆: API_3~30, P-ET0\n• GIS先验: CN0={static['CN0']:.1f}, Smax0={static['Smax0_mm']:.0f}mm",
        fc="#f0f7ee", ec="#2e6f40")
    box(2, 7, 21, 20, "外环 II 智能优化引擎",
        "• 单目标对比: PSO / GWO / SSA\n• 双目标寻优: NSGA-II (膝点)\n  f1 = 1 - KGE_val\n  f2 = |WBE_val| (%)",
        fc="#fff4e6", ec="#b35900")
    box(29, 34, 34, 21, "可微黄土混合产流物理核 (PyTorch ODE)",
        "1. 平滑积雪-融雪 dS0/dt (应对冬季-24.6℃)\n2. 可微 SCS-CN 超渗快径流 Q_inf (δ_cn动态调制)\n3. 土壤水-蒸散发-基流/蓄满 dS1/dt (δ_et调制)\n4. 双线性水库汇流 (快Sf + 慢Ss) → Q_phys",
        fc="#edf4fb", ec="#1f5a8a")
    box(29, 7, 34, 20, "物理引导 GRU 时序分支",
        "• 输入拼接: [X_t, 状态 S0, S1/Smax, Sf, Ss]\n• 动态参数调制头: Δα_et(t), ΔCN(t) 有界修正\n• 受限对数残差头: r_t ∈ (-γ, +γ)\n  捕捉黄土下垫面高阶非线性偏差",
        fc="#f5effb", ec="#5e3c99")
    box(69, 20, 29, 35, "物理融合输出与复合损失",
        "最终径流:\n  Q_hat = Q_phys · exp(γ·tanh(r_t)) ≥ 0\n\n复合损失 L_total:\n  L_data (0.65 NSE + 0.35 LogMSE)\n+ λ_wb · L_WB (30d+全序列闭合)\n+ λ_ode · L_ODE (退水通量一致)\n+ λ_mono · L_mono (无雨退水单调)\n+ λ_res · L_res (抑制黑箱接管)",
        fc="#fdf2f2", ec="#9e2a2b")

    arrow(23, 45, 29, 45, "强迫+先验")
    arrow(23, 39, 29, 21, "X_t")
    arrow(43, 34, 43, 27, "状态反馈", "#1f5a8a")
    arrow(49, 27, 49, 34, "δ调制", "#5e3c99")
    arrow(63, 44, 69, 44, "Q_phys")
    arrow(63, 18, 69, 28, "r_t")
    arrow(23, 16, 69, 22, "寻优 λ, hidden, lr, γ0", "#b35900")
    _save(fig, "fig03_model_architecture", is_syn)


def plot_fig04_stage1(s1_data: dict, is_syn: bool):
    runs = s1_data["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"wspace": 0.28})
    colors = {"PSO": "#1b9e77", "GWO": "#d95f02", "SSA": "#7570b3"}
    best_by_opt = {}
    for name in ["PSO", "GWO", "SSA"]:
        sub = [r for r in runs if r["optimizer"] == name]
        curves = np.array([-np.array(r["history"]["best"]) for r in sub])
        iters = np.arange(1, curves.shape[1] + 1)
        axes[0].plot(iters, curves.mean(0), color=colors[name], lw=2, label=f"{name} (平均最优)")
        axes[0].fill_between(iters, curves.min(0), curves.max(0), color=colors[name], alpha=0.16)
        best_by_opt[name] = max(sub, key=lambda r: 0.5 * r["valid_NSE"] + 0.5 * r["valid_KGE"])
    axes[0].set_xlabel("迭代代数"); axes[0].set_ylabel("率定期综合适应度 J = 0.5(NSE+KGE)")
    axes[0].set_title("阶段 I：物理核全局参数率定收敛曲线", fontweight="bold")
    axes[0].legend(loc="lower right", frameon=True); axes[0].grid(alpha=0.25)

    # normalised parameters comparison across the 3 optimisers
    from src.utils import load_cfg
    bounds = load_cfg()["physics"]["bounds"]
    x = np.arange(len(PARAM_NAMES)); w = 0.25
    for i, name in enumerate(["PSO", "GWO", "SSA"]):
        prm = best_by_opt[name]["params"]
        norm = [(prm[k] - bounds[k][0]) / (bounds[k][1] - bounds[k][0]) for k in PARAM_NAMES]
        axes[1].bar(x + (i - 1) * w, norm, width=w, color=colors[name], label=f"{name} (val NSE={best_by_opt[name]['valid_NSE']:.3f})")
    axes[1].set_xticks(x); axes[1].set_xticklabels(PARAM_NAMES, rotation=35, ha="right")
    axes[1].set_ylim(0, 1.08); axes[1].set_ylabel("物理参数归一化位置 [0, 1]")
    axes[1].set_title("三算法率定最优先验物理参数对比", fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True, fontsize=8.5); axes[1].grid(axis="y", alpha=0.25)
    _save(fig, "fig04_stage1_physics_calibration", is_syn)


def plot_fig05_stage2(s2_data: dict, is_syn: bool):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"wspace": 0.28})
    colors = {"PSO": "#1b9e77", "GWO": "#d95f02", "SSA": "#7570b3"}
    for name in ["PSO", "GWO", "SSA"]:
        h = -np.array(s2_data["single"][name]["history"]["best"])
        axes[0].plot(np.arange(1, len(h) + 1), h, "-o", color=colors[name], lw=2, ms=5,
                     label=f"{name} (best val J={s2_data['single'][name]['best_J']:.4f})")
    axes[0].set_xlabel("外环 II 迭代代数"); axes[0].set_ylabel("验证期综合目标 J")
    axes[0].set_title("阶段 II：PINN 超参数与物理权重单目标寻优对比", fontweight="bold")
    axes[0].legend(loc="lower right", frameon=True); axes[0].grid(alpha=0.25)

    ns = s2_data["nsga2"]
    eval_f = np.array([[e["1-KGE"] if "1-KGE" in e else 1 - e["KGE"], abs(e["WBE"])] for e in ns["evals"]])
    pf = np.array(ns["pareto_F"])
    order = np.argsort(pf[:, 0])
    pf_s = pf[order]
    k = ns["knee_index"]; kf = np.array(ns["knee_F"])
    axes[1].scatter(eval_f[:, 0], eval_f[:, 1], s=18, color="#bdbdbd", alpha=0.65, label="搜索历史候选解")
    axes[1].plot(pf_s[:, 0], pf_s[:, 1], "-o", color="#2b8cbe", lw=1.8, ms=6, label="NSGA-II Pareto 前沿")
    axes[1].scatter([kf[0]], [kf[1]], s=150, marker="*", color="#e31a1c", zorder=5,
                    label=f"膝点推荐解 (KGE={1-kf[0]:.3f}, |WBE|={kf[1]:.2f}%)")
    axes[1].set_xlabel("f1 = 1 - KGE_val (拟合精度损失)"); axes[1].set_ylabel("f2 = |WBE_val| (%) (水量平衡偏差)")
    axes[1].set_title("NSGA-II 精度 vs 水量守恒双目标 Pareto 前沿", fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True, fontsize=8.5); axes[1].grid(alpha=0.25)
    _save(fig, "fig05_stage2_optimizer_pareto", is_syn)


def plot_fig06_hydrographs(df_feat: pd.DataFrame, base_df: pd.DataFrame, meta_df: pd.DataFrame, cfg: dict, is_syn: bool):
    m_test = (df_feat.index >= cfg["split"]["test"][0]) & (df_feat.index <= cfg["split"]["test"][1])
    m_ext = (df_feat.index >= "2021-07-01") & (df_feat.index <= "2021-11-15")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.6), gridspec_kw={"hspace": 0.30})
    series = [("obs", "目标径流", "black", 1.5, "-"),
              ("SSA-LoessHydro", "SSA-LoessHydro", "#1b9e77", 1.0, "--"),
              ("LSTM", "纯 LSTM", "#ff7f00", 1.0, ":"),
              ("Base-PINN", "Base-PINN", "#984ea3", 1.0, "-."),
              ("Meta-PINN-Hydro", "Meta-PINN-Hydro", "#e41a1c", 1.3, "-")]
    for ax, mask, title in [(axes[0], m_test, "独立测试期 (2021-2023) 逐日降雨-径流过程线对比"),
                            (axes[1], m_ext, "2021年极端秋汛暴雨洪峰期局部放大 (2021-07 ~ 2021-11)")]:
        ax_r = ax.twinx(); ax_r.spines["top"].set_visible(False)
        ax_r.bar(df_feat.index[mask], df_feat.loc[mask, "P"], width=1.0, color="#a6bddb", alpha=0.5)
        ax_r.invert_yaxis(); ax_r.set_ylabel("降雨 P (mm/d)", color="#3690c0")
        for col, lbl, c, lw, ls in series:
            s = meta_df[col] if col in meta_df else (base_df[col] if col in base_df else None)
            if s is not None:
                ax.plot(df_feat.index[mask], s.loc[df_feat.index[mask]], color=c, lw=lw, ls=ls, label=lbl)
        ax.set_ylabel("径流深 (mm/d)"); ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper left", ncol=5, frameon=True, fontsize=8.5); ax.grid(alpha=0.2)
    axes[1].set_xlabel("日期")
    _save(fig, "fig06_hydrograph_and_2021_flood", is_syn)


def plot_fig07_scatter_fdc(base_df: pd.DataFrame, meta_df: pd.DataFrame, cfg: dict, is_syn: bool):
    m = (base_df.index >= cfg["split"]["test"][0]) & (base_df.index <= cfg["split"]["test"][1])
    obs = base_df.loc[m, "obs"].values
    models = [("SSA-LoessHydro", "#1b9e77", base_df.loc[m, "SSA-LoessHydro"].values if "SSA-LoessHydro" in base_df else None),
              ("LSTM", "#ff7f00", base_df.loc[m, "LSTM"].values if "LSTM" in base_df else None),
              ("Base-PINN", "#984ea3", base_df.loc[m, "Base-PINN"].values if "Base-PINN" in base_df else None),
              ("Meta-PINN-Hydro", "#e41a1c", meta_df.loc[m, "Meta-PINN-Hydro"].values)]
    models = [(n, c, v) for n, c, v in models if v is not None]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"wspace": 0.28})

    mx = max(float(np.nanmax(obs)), *(float(np.nanmax(v)) for _, _, v in models)) * 1.05
    for name, c, sim in models:
        axes[0].scatter(obs, sim, s=10, alpha=0.45, color=c, label=name)
    axes[0].plot([0, mx], [0, mx], "k--", lw=1.2, label="1:1 线")
    axes[0].set_xlim(0, mx); axes[0].set_ylim(0, mx)
    axes[0].set_xlabel("目标径流 (mm/d)"); axes[0].set_ylabel("模拟径流 (mm/d)")
    axes[0].set_title("测试期 1:1 散点拟合对比", fontweight="bold")
    axes[0].legend(loc="upper left", frameon=True, fontsize=8); axes[0].grid(alpha=0.25)

    p_exc = (np.arange(1, len(obs) + 1) / (len(obs) + 1)) * 100
    axes[1].semilogy(p_exc, np.sort(obs)[::-1] + 1e-4, "k-", lw=1.8, label="目标径流")
    for name, c, sim in models:
        axes[1].semilogy(p_exc, np.sort(sim)[::-1] + 1e-4, color=c, lw=1.3, label=name)
    axes[1].set_xlabel("超越概率 (%)"); axes[1].set_ylabel("径流深 (mm/d, 对数)")
    axes[1].set_title("测试期流量历时曲线 (FDC)", fontweight="bold")
    axes[1].legend(loc="lower left", frameon=True, fontsize=8); axes[1].grid(alpha=0.25)

    # flow regime residuals: low (<30%), mid (30-90%), high (>=90%)
    q30, q90 = np.nanpercentile(obs, [30, 90])
    regimes = [("低水(<P30)", obs < q30), ("中水(P30-90)", (obs >= q30) & (obs < q90)), ("高水(≥P90)", obs >= q90)]
    data_box, lbls = [], []
    for rname, rm in regimes:
        for name, _, sim in models:
            data_box.append((sim[rm] - obs[rm]))
            lbls.append(f"{rname}\n{name.split('-')[0]}")
    bp = axes[2].boxplot(data_box, tick_labels=lbls, patch_artist=True, showfliers=False)
    pal = [c for _, c, _ in models] * 3
    for patch, col in zip(bp["boxes"], pal):
        patch.set_facecolor(col); patch.set_alpha(0.55)
    axes[2].axhline(0, color="black", ls="--", lw=1)
    axes[2].tick_params(axis="x", rotation=45, labelsize=7.5)
    axes[2].set_ylabel("模拟偏差 Sim - Obs (mm/d)")
    axes[2].set_title("高/中/低水阶段残差分布", fontweight="bold"); axes[2].grid(axis="y", alpha=0.25)
    _save(fig, "fig07_scatter_fdc_residuals", is_syn)


def plot_fig08_ablation_small(final_data: dict, base_data: dict, is_syn: bool):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"wspace": 0.30})
    items = [("Meta-PINN-Hydro (完整)", final_data["ensemble"])] + list(final_data["ablation"].items())
    names = [n for n, _ in items]
    nse_t = [d["test"]["NSE"] for _, d in items]
    kge_t = [d["test"]["KGE"] for _, d in items]
    nse_e = [d["extreme_2021"]["NSE"] for _, d in items]
    y = np.arange(len(names)); h = 0.26
    axes[0].barh(y + h, nse_t, height=h, color="#2b8cbe", label="测试期 NSE (2021-2023)")
    axes[0].barh(y, kge_t, height=h, color="#1b9e77", label="测试期 KGE (2021-2023)")
    axes[0].barh(y - h, nse_e, height=h, color="#e34a33", label="2021极值年 NSE")
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=9)
    axes[0].invert_yaxis(); axes[0].set_xlabel("模型效率系数")
    axes[0].set_title("核心模块消融实验对比", fontweight="bold")
    axes[0].legend(loc="lower left", frameon=True, fontsize=8); axes[0].grid(axis="x", alpha=0.25)

    # Small sample comparison (50% calib 2018-2019 vs full calib 2016-2019)
    sm = final_data["small_sample"]
    full_map = {
        "PSO-LoessHydro": base_data["metrics"]["PSO-LoessHydro"]["test"],
        "LSTM": base_data["metrics"].get("LSTM", {}).get("test", {"NSE": np.nan, "KGE": np.nan}),
        "Meta-PINN-Hydro": final_data["ensemble"]["test"],
    }
    comp = [k for k in ["PSO-LoessHydro", "LSTM", "Meta-PINN-Hydro"] if k in sm]
    x = np.arange(len(comp)); w = 0.20
    axes[1].bar(x - 1.5 * w, [full_map[k]["NSE"] for k in comp], width=w, color="#2b8cbe", label="全率定期(4年) NSE")
    axes[1].bar(x - 0.5 * w, [sm[k]["test"]["NSE"] for k in comp], width=w, color="#a6bddb", label="50%小样本(2年) NSE")
    axes[1].bar(x + 0.5 * w, [full_map[k]["KGE"] for k in comp], width=w, color="#1b9e77", label="全率定期(4年) KGE")
    axes[1].bar(x + 1.5 * w, [sm[k]["test"]["KGE"] for k in comp], width=w, color="#a6dba0", label="50%小样本(2年) KGE")
    axes[1].set_xticks(x); axes[1].set_xticklabels(comp)
    axes[1].set_ylabel("测试期评价指标"); axes[1].set_title("50%少样本率定稳健性压力测试", fontweight="bold")
    axes[1].legend(loc="lower right", frameon=True, fontsize=8); axes[1].grid(axis="y", alpha=0.25)
    _save(fig, "fig08_ablation_and_small_sample", is_syn)


def plot_fig09_physical_states(df_feat: pd.DataFrame, states_df: pd.DataFrame, diag: dict, is_syn: bool):
    m = df_feat.index >= "2020-01-01"
    idx = df_feat.index[m]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.2), gridspec_kw={"hspace": 0.32, "wspace": 0.24})

    ax = axes[0, 0]; ax_r = ax.twinx(); ax_r.spines["top"].set_visible(False)
    ax.plot(idx, states_df.loc[m, "S1"], color="#1b9e77", lw=1.2, label="PINN 土壤蓄水 S1 (mm)")
    ax_r.plot(idx, df_feat.loc[m, "API_15"], color="#d95f02", lw=0.9, alpha=0.75, label="前期降雨指数 API_15")
    sp_r = diag["test_2021_2023"]["spearman_S1_API15"]
    ax.set_ylabel("S1 (mm)", color="#1b9e77"); ax_r.set_ylabel("API_15 (mm)", color="#d95f02")
    ax.set_title(f"土壤蓄水状态 S1 与 API_15 联动 (Spearman r={sp_r:.3f})", fontweight="bold")

    ax = axes[0, 1]
    ax.plot(idx, df_feat.loc[m, "ET0"], color="#bdbdbd", lw=1.0, label="潜在蒸散 ET0")
    ax.plot(idx, states_df.loc[m, "Ea"], color="#e6550d", lw=1.1, label="模型实际蒸散 Ea")
    ax.set_ylabel("蒸散发 (mm/d)"); ax.set_title("实际蒸散发 Ea 与潜在蒸发上限 ET0 对比", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=8.5); ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.stackplot(idx, states_df.loc[m, "Qinf"], states_df.loc[m, "Qsat"], states_df.loc[m, "Qb"],
                 labels=["超渗快径流 Qinf", "蓄满地表径流 Qsat", "地下基流 Qb"],
                 colors=["#e34a33", "#2b8cbe", "#1b9e77"], alpha=0.8)
    ax.plot(idx, states_df.loc[m, "Q"], color="black", lw=0.8, alpha=0.8, label="混合输出 Q_hat")
    ax.set_ylabel("产流组分 (mm/d)"); ax.set_title("黄土高原混合产流组分拆解", fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=8); ax.grid(alpha=0.2)

    ax = axes[1, 1]
    cum_p = np.cumsum(df_feat.loc[m, "P"].values)
    cum_ea = np.cumsum(states_df.loc[m, "Ea"].values)
    cum_q = np.cumsum(states_df.loc[m, "Q"].values)
    cum_qp = np.cumsum(states_df.loc[m, "Qphys"].values)
    S_all = (states_df["S0"] + states_df["S1"] + states_df["Sf"] + states_df["Ss"]).values
    i0 = int(np.flatnonzero(m)[0])
    dS = S_all[m] - (S_all[i0 - 1] if i0 > 0 else S_all[i0])
    ax.plot(idx, cum_p, "k-", lw=1.6, label="累计降雨 ΣP")
    ax.plot(idx, cum_ea + cum_qp + dS, "--", color="#1b9e77", lw=1.8, label="物理核 Σ(Ea+Qphys)+ΔS")
    ax.plot(idx, cum_ea + cum_q + dS, ":", color="#e41a1c", lw=1.8, label="混合模型 Σ(Ea+Qhat)+ΔS")
    ax.set_ylabel("累计水量 (mm)"); ax.set_title("2020-2023年流域累计水量平衡闭合检验", fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=8); ax.grid(alpha=0.2)
    for sub_ax in axes.ravel():
        sub_ax.xaxis.set_major_locator(mdates.YearLocator())
        sub_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, "fig09_internal_physical_states", is_syn)


def plot_fig10_uncertainty(base_df: pd.DataFrame, meta_df: pd.DataFrame, final_data: dict, cfg: dict, is_syn: bool):
    m_ext = (meta_df.index >= "2021-07-01") & (meta_df.index <= "2021-11-15")
    m_test = (meta_df.index >= cfg["split"]["test"][0]) & (meta_df.index <= cfg["split"]["test"][1])
    cov = final_data["ensemble"]["test"].get("band_coverage_pct", np.nan)
    summ = final_data["per_seed_summary"]["test"]
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 6.8), gridspec_kw={"hspace": 0.30})
    for ax, m, title in [(axes[0], m_test, f"5随机种子集成预测与90%置信区间 (测试期 NSE={summ['NSE']['mean']:.3f}±{summ['NSE']['std']:.3f}, 覆盖率={cov:.1f}%)"),
                         (axes[1], m_ext, "2021年极端秋汛洪峰多种子90%不确定性包络放大")]:
        idx = meta_df.index[m]
        ax.fill_between(idx, meta_df.loc[m, "Meta-PINN-Hydro_p05"], meta_df.loc[m, "Meta-PINN-Hydro_p95"],
                        color="#fb8072", alpha=0.45, label="5-95% 多种子预测包络")
        ax.plot(idx, meta_df.loc[m, "Meta-PINN-Hydro"], color="#e41a1c", lw=1.3, label="集成均值 Meta-PINN-Hydro")
        if "obs" in base_df and base_df["obs"].notna().any():
            ax.plot(idx, base_df.loc[m, "obs"], color="black", lw=0.9, ls="--", label="目标径流")
        ax.set_ylabel("径流深 (mm/d)"); ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper left", frameon=True, fontsize=8.5); ax.grid(alpha=0.2)
    axes[1].set_xlabel("日期")
    _save(fig, "fig10_multi_seed_uncertainty", is_syn)
