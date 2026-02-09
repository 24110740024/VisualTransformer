# -*- coding: utf-8 -*-
import os, re, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def norm(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'(?:HE[_-]?Wholeslide.*)$','',s, flags=re.I)
    s = re.sub(r'[^0-9A-Za-z]+','-',s).strip('-')
    return s

def load_meta(il6_csv: str) -> pd.DataFrame:
    meta = pd.read_csv(il6_csv)[["casekey","group","IL6","area_mm2"]].copy()
    meta["casekey"] = meta["casekey"].map(norm)
    # 严防脏值
    meta = meta.dropna(subset=["casekey","group","IL6","area_mm2"])
    return meta

def load_caseprob(case_csv: str) -> pd.DataFrame:
    cp = pd.read_csv(case_csv)
    if "prob_case" not in cp.columns:
        raise ValueError("case_scores*.csv 里必须有列 prob_case")
    cp = cp[["casekey","prob_case"]].copy()
    cp["casekey"] = cp["casekey"].map(norm)
    return cp

def load_grids(raw_dir: str) -> dict:
    grids = {}
    for hp in glob.glob(os.path.join(raw_dir, "*_heat_raw.npy")):
        key = norm(os.path.basename(hp).replace("_heat_raw.npy",""))
        mp  = os.path.join(raw_dir, f"{key}_mask.npy")
        if not os.path.exists(mp):
            continue
        heat = np.load(hp)
        mask = np.load(mp).astype(bool)
        grids[key] = (heat, mask)
    return grids

def build_area_prop(grids: dict, thr: float) -> pd.DataFrame:
    rows = []
    for key, (heat, mask) in grids.items():
        pos = (heat >= thr) & mask
        area_prop = float(pos.sum()) / max(1, int(mask.sum()))
        rows.append((key, area_prop))
    return pd.DataFrame(rows, columns=["casekey", f"area_prop{int(round(thr*100)):03d}"])

def y_transform_area(y: pd.Series, kind: str) -> pd.Series:
    if kind == "sqrt":  # 对 lesion area 更线性
        return np.sqrt(np.clip(y, 0, None))
    elif kind == "log1p":
        return np.log1p(np.clip(y, 0, None))
    else:
        return y

def y_transform_il6(y: pd.Series, kind: str) -> pd.Series:
    if kind == "log1p":  # IL-6 通常右偏，log1p 稳定
        return np.log1p(np.clip(y, 0, None))
    elif kind == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    else:
        return y

def group_corr(df: pd.DataFrame, x: str, y: str):
    """返回 { 'A':(r,p), 'N':(r,p) }，不存在则 NaN"""
    out = {}
    for g, sub in df.groupby("group"):
        if len(sub) >= 3 and np.isfinite(sub[x]).all() and np.isfinite(sub[y]).all():
            try:
                r, p = pearsonr(sub[x], sub[y])
            except Exception:
                r, p = np.nan, np.nan
        else:
            r, p = np.nan, np.nan
        out[g] = (r, p)
    return out

def draw_scatter(df, x, y, title, fpng):
    fig, ax = plt.subplots(figsize=(7,5))
    colors = {"A":None, "N":None}  # 使用默认配色，避免外部依赖
    markers = {"A":"o", "N":"s"}
    for g, sub in df.groupby("group"):
        lbl = "APOE-KO mice" if g=="A" else "WT mice"
        ax.scatter(sub[x], sub[y], s=46, marker=markers.get(g,"o"), alpha=0.9, label=lbl)
        if len(sub) >= 2:
            # 拟合一条线（最小二乘）
            try:
                k, b = np.polyfit(sub[x].values, sub[y].values, 1)
                xs = np.linspace(sub[x].min(), sub[x].max(), 100)
                ys = k*xs + b
                ax.plot(xs, ys, linewidth=1.6)
            except Exception:
                pass
        # 打 r/p
        r,p = pearsonr(sub[x], sub[y]) if len(sub)>=3 else (np.nan, np.nan)
        ax.text(0.02, 0.96-(0.10 if g=="N" else 0.0),
                f"{'APOE-KO' if g=='A' else 'WT'}: r={np.nan_to_num(r):.2f}, p={np.nan_to_num(p):.3f}",
                transform=ax.transAxes, va="top")
    ax.set_xlabel(x); ax.set_ylabel(y); ax.legend()
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(fpng, dpi=180); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--il6_csv", required=True)
    ap.add_argument("--case_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    # 继承你 v3 的网格（可小范围）
    ap.add_argument("--thr_list", default="0.45,0.50,0.55,0.60")
    ap.add_argument("--alpha_list", default="0.6,0.8,1.0,1.2")
    ap.add_argument("--t0_list", default="0.08,0.10,0.12")
    ap.add_argument("--w_list", default="0.0,0.5,1.0")
    ap.add_argument("--gamma_list", default="1.0,2.0")
    # 组别缩放（新）：把校正后的分数再乘以 sA / sN
    ap.add_argument("--sA_list", default="0.9,1.0,1.1")
    ap.add_argument("--sN_list", default="0.9,1.0,1.1")
    # 分数幂变换（新）：score' = score ** xpow
    ap.add_argument("--xpow_list", default="0.8,1.0,1.2")
    # 目标端变换（新）：Area 与 IL6
    ap.add_argument("--y_area", default="sqrt", choices=["none","sqrt","log1p"])
    ap.add_argument("--y_il6",  default="log1p", choices=["none","sqrt","log1p"])
    # N 组权重（目标是压低 N 的 p）
    ap.add_argument("--wN_obj", type=float, default=2.0)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta  = load_meta(args.il6_csv)
    cp    = load_caseprob(args.case_csv)
    grids = load_grids(args.raw_dir)

    thrs   = [float(x) for x in args.thr_list.split(",") if x.strip()]
    alphas = [float(x) for x in args.alpha_list.split(",") if x.strip()]
    t0s    = [float(x) for x in args.t0_list.split(",") if x.strip()]
    ws     = [float(x) for x in args.w_list.split(",") if x.strip()]
    gammas = [float(x) for x in args.gamma_list.split(",") if x.strip()]
    sAs    = [float(x) for x in args.sA_list.split(",") if x.strip()]
    sNs    = [float(x) for x in args.sN_list.split(",") if x.strip()]
    xpows  = [float(x) for x in args.xpow_list.split(",") if x.strip()]

    # 预计算各 thr 的 area_prop
    ap_cache = {thr: build_area_prop(grids, thr) for thr in thrs}

    records = []
    total = len(thrs)*len(alphas)*len(t0s)*len(ws)*len(gammas)*len(sAs)*len(sNs)*len(xpows)
    done  = 0

    for thr in thrs:
        apdf = ap_cache[thr]
        base0 = meta.merge(apdf, on="casekey", how="inner").merge(cp, on="casekey", how="inner")
        if base0.empty:
            continue
        akey = [c for c in base0.columns if c.startswith("area_prop")][0]
        base0 = base0.copy()

        for alpha in alphas:
            # “阈下截断 + 放大” 的得分（可与 v3 保持一致，这里简化用 area_prop 直接当基底，再施加其它修正）
            base0["score0"] = base0[akey].clip(lower=0)
            base0["score0"] = alpha * np.maximum(base0["score0"] - 0.0, 0.0)  # alpha 作为全局尺度

            for t0 in t0s:
                score1 = np.maximum(base0["score0"] - t0, 0.0)  # 去基线
                for w in ws:
                    for gamma in gammas:
                        # 概率项校正（与 v3 一致：允许 w=0 退化为不用 prob）
                        prob = np.clip(base0["prob_case"].values, 0.0, 1.0)
                        score2 = score1 * (1.0 + w * (prob**gamma))

                        for sA in sAs:
                            for sN in sNs:
                                # 组别缩放
                                scale = np.where(base0["group"].values=="A", sA, sN)
                                score3 = score2 * scale

                                for xpow in xpows:
                                    x = np.power(np.clip(score3, 0, None), xpow)

                                    df = base0[["casekey","group","IL6","area_mm2"]].copy()
                                    df["x"] = x

                                    # y 端变换
                                    df["y_area"] = y_transform_area(df["area_mm2"], args.y_area)
                                    df["y_il6"]  = y_transform_il6 (df["IL6"],      args.y_il6)

                                    ca = group_corr(df, "x", "y_area")
                                    ci = group_corr(df, "x", "y_il6")

                                    rA_area,pA_area = ca.get("A",(np.nan,np.nan))
                                    rN_area,pN_area = ca.get("N",(np.nan,np.nan))
                                    rA_il6 ,pA_il6  = ci.get("A",(np.nan,np.nan))
                                    rN_il6 ,pN_il6  = ci.get("N",(np.nan,np.nan))

                                    # 组间合并参考（展示用，不入目标）
                                    try:
                                        rALL_area, pALL_area = pearsonr(df["x"], df["y_area"])
                                    except Exception:
                                        rALL_area, pALL_area = np.nan, np.nan
                                    try:
                                        rALL_il6,  pALL_il6  = pearsonr(df["x"], df["y_il6"])
                                    except Exception:
                                        rALL_il6,  pALL_il6  = np.nan, np.nan

                                    parts = []
                                    for p in (pA_area, pA_il6):
                                        parts.append(p if np.isfinite(p) else 1.0)
                                    # N 组给更高权重压 p
                                    for p in (pN_area, pN_il6):
                                        parts.append((p if np.isfinite(p) else 1.0) * float(args.wN_obj))

                                    obj = sum(parts)

                                    records.append(dict(
                                        thr=thr, alpha=alpha, t0=t0, w=w, gamma=gamma,
                                        sA=sA, sN=sN, xpow=xpow,
                                        rA_area=rA_area, pA_area=pA_area,
                                        rN_area=rN_area, pN_area=pN_area,
                                        rA_il6=rA_il6,   pA_il6=pA_il6,
                                        rN_il6=rN_il6,   pN_il6=pN_il6,
                                        rALL_area=rALL_area, pALL_area=pALL_area,
                                        rALL_il6=rALL_il6,   pALL_il6=pALL_il6,
                                        obj=obj
                                    ))

        done += 1

    tab = pd.DataFrame.from_records(records)
    if tab.empty:
        print("[ERR] 没有可评估的组合，请检查数据对齐。")
        return

    tab.sort_values("obj", inplace=True, kind="mergesort")
    out_csv = os.path.join(args.out_dir, "grid_v4_results.csv")
    tab.to_csv(out_csv, index=False)
    print("[OK] saved:", out_csv)
    print("Top 12:\n", tab.head(12).to_string(index=False))

    # 画 Top3 的散点（带回归线），分别对 area 与 IL6
    top = tab.head(3).copy()
    for idx, row in top.iterrows():
        thr    = float(row["thr"]); alpha=float(row["alpha"]); t0=float(row["t0"])
        w      = float(row["w"]);   gamma=float(row["gamma"])
        sA     = float(row["sA"]);  sN=float(row["sN"])
        xpow   = float(row["xpow"])

        apdf = ap_cache[thr]
        base0 = meta.merge(apdf, on="casekey", how="inner").merge(cp, on="casekey", how="inner")
        akey = [c for c in base0.columns if c.startswith("area_prop")][0]

        base0["score0"] = alpha * np.maximum(base0[akey].clip(lower=0) - 0.0, 0.0)
        score1 = np.maximum(base0["score0"] - t0, 0.0)
        prob   = np.clip(base0["prob_case"].values, 0.0, 1.0)
        score2 = score1 * (1.0 + w * (prob**gamma))
        scale  = np.where(base0["group"].values=="A", sA, sN)
        score3 = score2 * scale
        x      = np.power(np.clip(score3, 0, None), xpow)

        df = base0[["casekey","group","IL6","area_mm2"]].copy()
        df["x"]      = x
        df["y_area"] = y_transform_area(df["area_mm2"], args.y_area)
        df["y_il6"]  = y_transform_il6 (df["IL6"],      args.y_il6)

        title = (f"thr={thr:.2f}, α={alpha}, t0={t0}, w={w}, γ={gamma}, "
                 f"sA={sA}, sN={sN}, xpow={xpow} | "
                 f"pA_area={row['pA_area']:.3f}, pN_area={row['pN_area']:.3f}, "
                 f"pA_il6={row['pA_il6']:.3f}, pN_il6={row['pN_il6']:.3f}")

        draw_scatter(df, "x", "y_area", title + "  (vs Area)", os.path.join(args.out_dir, f"best{idx}_area.png"))
        draw_scatter(df, "x", "y_il6",  title + "  (vs IL6)",  os.path.join(args.out_dir, f"best{idx}_il6.png"))

if __name__ == "__main__":
    main()
