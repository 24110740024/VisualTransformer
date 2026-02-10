# -*- coding: utf-8 -*-
import os, argparse, warnings, json
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso, LassoCV, lasso_path
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import networkx as nx

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({
    "font.size": 12,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------- 列名适配 ----------
def pick_il6_col(df):
    for c in df.columns:
        lc = c.lower().replace('-', '').replace('_', '')
        if lc == 'il6':
            return c
    raise KeyError("在 1IL6Area_clean.csv 里找不到 IL-6 列（IL6 / IL-6 / il6）")

def pick_area_col(df):
    for c in df.columns:
        lc = c.lower().replace(' ', '').replace('-', '_')
        if lc in ('area', 'y_area', 'lesion_area', 'area_mm2') or 'area' in lc:
            return c
    raise KeyError("在 1IL6Area_clean.csv 里找不到面积列（area/lesion_area/area_mm2 等）")

# ---------- 数据读取 ----------
def load_xy(feat_csv, il6_csv, target, y_area="sqrt", y_il6="log1p", drop_dup_probcase=True):
    X = pd.read_csv(feat_csv)
    Y = pd.read_csv(il6_csv)

    def find_key(df):
        cols = list(df.columns)
        low = [c.lower() for c in cols]
        for k in ["casekey", "case", "id", "key"]:
            if k in low:
                return cols[low.index(k)]
        raise KeyError(f"找不到 ID 列（casekey/case/id/key），现有列：{list(df.columns)}")

    ckey_x = find_key(X)
    ckey_y = find_key(Y)

    X = X.set_index(ckey_x)
    Y = Y.set_index(ckey_y)

    col_il6  = pick_il6_col(Y)
    col_area = pick_area_col(Y)

    if target == "il6":
        y = Y[col_il6].astype(float)
        if y_il6 == "log1p":
            y = np.log1p(y)
        y_name = f"{col_il6} ({'log1p' if y_il6=='log1p' else 'raw'})"
    else:
        y = Y[col_area].astype(float)
        if y_area == "sqrt":
            y = np.sqrt(y)
        elif y_area == "log1p":
            y = np.log1p(y)
        y_name = f"{col_area} ({'sqrt' if y_area=='sqrt' else ('log1p' if y_area=='log1p' else 'raw')})"

    idx = X.index.intersection(y.index)
    X = X.loc[idx].copy()
    y = y.loc[idx].copy()

    bad = [c for c in X.columns if (X[c].dtype == "object") or (c.lower() in ["casekey","group","label","id","key"])]
    X = X.drop(columns=bad, errors="ignore")

    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    X = X.fillna(X.median(numeric_only=True))

    # ---- 关键修复：prob_case 与 prob_topk 完全重复 -> 删除 prob_case，避免 LASSO path 出现“塌平怪线” ----
    if drop_dup_probcase and ("prob_case" in X.columns) and ("prob_topk" in X.columns):
        diff = (X["prob_case"] - X["prob_topk"]).abs().max()
        if float(diff) == 0.0:
            X = X.drop(columns=["prob_case"])
            print("[INFO] drop duplicated column: prob_case (identical to prob_topk)")

    return X, y, y_name

# ---------- Boruta ----------
def boruta_select(X, y, seed):
    rf = RandomForestRegressor(
        n_estimators=1000, max_depth=5, n_jobs=1, random_state=seed
    )
    boruta = BorutaPy(rf, n_estimators='auto', max_iter=100, random_state=seed)
    boruta.fit(X.values, y.values)
    mask = boruta.support_
    return list(X.columns[mask])

# ---------- LASSO（CV/路径/λmin/λ1se 两套特征） ----------
def lasso_cv_and_paths(
    X, y, seed, kfolds=5, repeats=1,
    n_alphas=80, path_eps=1e-3
):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    yv = y.values.astype(float)

    # 数据驱动 alpha 网格：alpha_max -> alpha_max*eps
    n = Xs.shape[0]
    alpha_max = np.max(np.abs(Xs.T @ yv)) / n
    alpha_min_grid = alpha_max * float(path_eps)
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min_grid), int(n_alphas))  # 大->小

    lcv = LassoCV(
        alphas=alphas,
        cv=RepeatedKFold(n_splits=kfolds, n_repeats=repeats, random_state=seed),
        random_state=seed,
        max_iter=50000
    )
    lcv.fit(Xs, yv)

    # λ_min 与 λ_1se
    mse_mean = lcv.mse_path_.mean(axis=1)
    mse_std  = lcv.mse_path_.std(axis=1)
    idx_min  = int(np.argmin(mse_mean))
    mse_1se  = mse_mean[idx_min] + mse_std[idx_min]
    cand = np.where(mse_mean <= mse_1se)[0]

    # 选更大的 alpha（更强正则）作为 1se
    if lcv.alphas_[0] > lcv.alphas_[-1]:   # 降序
        idx_1se = int(cand[0])
    else:                                  # 升序兜底
        idx_1se = int(cand[-1])

    alpha_min = float(lcv.alphas_[idx_min])
    alpha_1se = float(lcv.alphas_[idx_1se])

    # 路径（用同一套 alphas）
    alphas_path, coefs_path, _ = lasso_path(Xs, yv, alphas=alphas, max_iter=50000)

    # 两套：min / 1se
    Lmin = Lasso(alpha=alpha_min, max_iter=50000, random_state=seed).fit(Xs, yv)
    coef_min = Lmin.coef_
    feats_min = list(np.array(X.columns)[np.abs(coef_min) > 1e-8])

    L1 = Lasso(alpha=alpha_1se, max_iter=50000, random_state=seed).fit(Xs, yv)
    coef_1se = L1.coef_
    feats_1se = list(np.array(X.columns)[np.abs(coef_1se) > 1e-8])

    return {
        "lcv": lcv,
        "alpha_min": alpha_min,
        "alpha_1se": alpha_1se,
        "alphas_path": alphas_path,
        "coefs_path": coefs_path,
        "scaler": scaler,
        "lasso_selected_min": feats_min,
        "lasso_selected_1se": feats_1se,
        "coef_min": coef_min,
        "coef_1se": coef_1se,
    }

# ---------- 画图 ----------
def plot_network(selected_boruta, selected_lasso, out_png, title="Feature Selection by Boruta and Lasso"):
    selected_boruta = list(selected_boruta) if selected_boruta is not None else []
    selected_lasso  = list(selected_lasso)  if selected_lasso  is not None else []

    B = set(selected_boruta)
    L = set(selected_lasso)
    feats = sorted(B | L)

    G = nx.Graph()
    G.add_node("Boruta", type="method")
    G.add_node("Lasso",  type="method")

    for f in feats:
        G.add_node(f, type="feat")
        if f in B: G.add_edge("Boruta", f)
        if f in L: G.add_edge("Lasso",  f)

    pos = {"Boruta": (0.0, 0.5), "Lasso": (1.0, 0.5)}

    if len(feats) == 0:
        plt.figure(figsize=(6,4))
        nx.draw_networkx_nodes(G, pos, nodelist=["Boruta"], node_color="#9ecae1", node_size=1200)
        nx.draw_networkx_nodes(G, pos, nodelist=["Lasso"],  node_color="#9ecae1", node_size=1200)
        nx.draw_networkx_labels(G, pos, font_size=11)
        plt.title(title + " (No features selected)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        return

    ys = np.linspace(0.85, 0.15, len(feats))
    for y, f in zip(ys, feats):
        pos[f] = (0.5, float(y))

    plt.figure(figsize=(7, 0.35*len(feats) + 2.8))
    nx.draw_networkx_edges(G, pos, alpha=0.55, width=1.2)

    nx.draw_networkx_nodes(G, pos, nodelist=["Boruta"], node_color="#9ecae1", node_size=1200)
    nx.draw_networkx_nodes(G, pos, nodelist=["Lasso"],  node_color="#9ecae1", node_size=1200)

    both = [f for f in feats if (f in B and f in L)]
    only = [f for f in feats if f not in both]
    nx.draw_networkx_nodes(G, pos, nodelist=only, node_color="#fdd0a2", node_size=380)
    nx.draw_networkx_nodes(G, pos, nodelist=both, node_color="#fdae6b", node_size=460)

    nx.draw_networkx_labels(G, pos, font_size=10)

    if len(selected_lasso) == 0:
        plt.text(0.98, 0.05, "Lasso selected 0 features",
                 transform=plt.gca().transAxes, ha="right", va="bottom", fontsize=9)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_paths(
    alphas, coefs, out_png,
    alpha_min=None, alpha_1se=None,
    auto_trim=True, max_lines=40,
    feature_names=None,
    legend=True,
    legend_max=14,
    out_map_csv=None
):
    """
    alphas: (n_alphas,)
    coefs:  (n_features, n_alphas)
    feature_names: list[str] length n_features, used for legend
    out_map_csv: save line->feature mapping (after filtering max_lines)
    """
    alphas = np.asarray(alphas).astype(float)
    coefs  = np.asarray(coefs).astype(float)

    # alpha 从大到小
    order = np.argsort(alphas)[::-1]
    alphas = alphas[order]
    coefs  = coefs[:, order]
    x = np.log10(alphas)

    names = None
    if feature_names is not None and len(feature_names) == coefs.shape[0]:
        names = list(feature_names)

    # 太多线就只画幅度最大的前 max_lines 条（并同步筛 names）
    keep_idx = np.arange(coefs.shape[0])
    if coefs.shape[0] > int(max_lines):
        strength = np.max(np.abs(coefs), axis=1)
        keep_idx = np.argsort(strength)[::-1][:int(max_lines)]
        coefs = coefs[keep_idx, :]
        if names is not None:
            names = [names[i] for i in keep_idx]

    # 裁掉“几乎全为 0”的长尾
    if auto_trim:
        l1 = np.sum(np.abs(coefs), axis=0)
        mx = float(np.max(l1)) if np.max(l1) > 0 else 0.0
        thr = 0.01 * mx
        keep = np.where(l1 > thr)[0]
        if len(keep) >= 5:
            i0, i1 = int(keep[0]), int(keep[-1])
            pad = max(2, int(0.05 * (i1 - i0 + 1)))
            i0 = max(0, i0 - pad)
            i1 = min(len(x) - 1, i1 + pad)
            x = x[i0:i1+1]
            coefs = coefs[:, i0:i1+1]

    # --- 画图 ---
    show_feat_legend = legend and (names is not None)

    if show_feat_legend:
        plt.figure(figsize=(8.8, 4.2))
    else:
        plt.figure(figsize=(7, 4))

    handles = []
    labels  = []

    for i in range(coefs.shape[0]):
        if names is None:
            h, = plt.plot(x, coefs[i], lw=2)
        else:
            h, = plt.plot(x, coefs[i], lw=2, label=names[i])
            handles.append(h)
            labels.append(names[i])

    plt.gca().invert_xaxis()  # 大 λ 在左，小 λ 在右（论文常见）

    # λmin/λ1se
    lam_h, lam_l = [], []
    if alpha_min is not None:
        h = plt.axvline(np.log10(alpha_min), color="k",  ls="--", lw=1.5)
        lam_h.append(h); lam_l.append(r"$\lambda_{min}$")
    if alpha_1se is not None:
        h = plt.axvline(np.log10(alpha_1se), color="C0", ls="--", lw=1.5)
        lam_h.append(h); lam_l.append(r"$\lambda_{1se}$")

    plt.xlabel(r"$\log_{10}(\lambda)$")
    plt.ylabel("Coefficients")
    plt.title("Lasso Paths")

    # 图例策略：
    # - 右侧只放前 legend_max 条特征（太多会挤爆）
    # - λmin/λ1se 放图内右上
    if show_feat_legend:
        if len(lam_h) > 0:
            leg1 = plt.legend(lam_h, lam_l, frameon=False, loc="upper right")
            plt.gca().add_artist(leg1)

        if len(handles) > 0:
            k = min(int(legend_max), len(handles))
            plt.legend(
                handles[:k], labels[:k],
                frameon=False, fontsize=9,
                loc="center left", bbox_to_anchor=(1.02, 0.5),
                title=f"Top {k} lines"
            )
        plt.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        if len(lam_h) > 0:
            plt.legend(lam_h, lam_l, frameon=False)
        plt.tight_layout()

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    # 输出 mapping：线条顺序 -> feature 名（经过 max_lines 筛选后的）
    if out_map_csv is not None and names is not None:
        pd.DataFrame({
            "line_rank": np.arange(1, len(names) + 1),
            "feature": names
        }).to_csv(out_map_csv, index=False)

def plot_cvcurve(lcv, alpha_min, alpha_1se, out_png):
    a = np.asarray(lcv.alphas_).astype(float)
    m = lcv.mse_path_.mean(axis=1)
    s = lcv.mse_path_.std(axis=1)

    order = np.argsort(a)  # 小->大
    a, m, s = a[order], m[order], s[order]

    plt.figure(figsize=(7,4))
    plt.errorbar(a, m, yerr=s, fmt="o", ms=3, lw=1, alpha=0.6)
    plt.axvline(alpha_min, color="k",  ls="--", label=r"$\lambda_{min}$")
    plt.axvline(alpha_1se, color="C0", ls="--", label=r"$\lambda_{1se}$")
    plt.xscale("log")
    plt.xlabel(r"$\lambda$ (alpha)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Lasso CV Curve")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def compose_panel(png_net, png_paths, png_cv, out_png):
    fig = plt.figure(figsize=(6,10))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], figure=fig)
    for i, img in enumerate([png_net, png_paths, png_cv]):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(plt.imread(img))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def scatter_oof(x, y, title, xlab, ylab, out_png):
    r, p = pearsonr(x, y)
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=36, alpha=0.9)
    k, b = np.polyfit(x, y, 1)
    xs = np.linspace(min(x), max(x), 100)
    plt.plot(xs, k*xs+b, lw=2)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.text(0.02, 0.02, f"Pearson r={r:.3f}, p={p:.3g}", transform=plt.gca().transAxes, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return r, p

# ---------- OOF 预测 + 置换检验 ----------
def oof_predict(X, y, alpha, seed, kfolds=5, repeats=1):
    rkf = RepeatedKFold(n_splits=kfolds, n_repeats=repeats, random_state=seed)
    pipe = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=50000, random_state=seed))
    oof = np.zeros_like(y.values, dtype=float)
    idx = np.arange(len(y))
    for tr, te in rkf.split(idx):
        pipe.fit(X.values[tr], y.values[tr])
        oof[te] = pipe.predict(X.values[te])
    return oof

def perm_p_value(X, y, alpha, seed, kfolds, repeats, perm=1000):
    rng = np.random.default_rng(seed)
    oof = oof_predict(X, y, alpha, seed, kfolds, repeats)
    r_obs = pearsonr(oof, y.values)[0]
    cnt = 0
    for _ in range(int(perm)):
        y_perm = rng.permutation(y.values)
        oof_p = oof_predict(X, pd.Series(y_perm, index=y.index), alpha, seed, kfolds, repeats)
        r_p = pearsonr(oof_p, y_perm)[0]
        if abs(r_p) >= abs(r_obs):
            cnt += 1
    pval = (cnt + 1) / (perm + 1)
    return r_obs, pval, oof

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_csv", required=True)
    ap.add_argument("--il6_csv",  required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--target",   choices=["area","il6"], required=True)
    ap.add_argument("--y_area",   choices=["raw","sqrt","log1p"], default="sqrt")
    ap.add_argument("--y_il6",    choices=["raw","sqrt","log1p"], default="log1p")
    ap.add_argument("--kfolds",   type=int, default=5)
    ap.add_argument("--repeats",  type=int, default=1)
    ap.add_argument("--perm",     type=int, default=0)
    ap.add_argument("--seed",     type=int, default=42)

    # 控制 lasso path 外观
    ap.add_argument("--n_alphas", type=int, default=80, help="number of alphas for LASSO path/CV grid")
    ap.add_argument("--path_eps", type=float, default=1e-3, help="alpha_min = alpha_max * path_eps")
    ap.add_argument("--max_lines", type=int, default=40, help="max lines to draw in lasso path")
    ap.add_argument("--no_trim", action="store_true", help="disable auto trim of near-zero tail on path plot")
    ap.add_argument("--legend_max", type=int, default=14, help="max feature names shown in legend (right side)")

    # 新增：是否保留重复 prob_case
    ap.add_argument("--keep_dup_probcase", action="store_true",
                    help="do NOT drop prob_case even if identical to prob_topk")

    # 新增：LASSO 规则
    ap.add_argument("--lasso_rule", choices=["auto", "1se", "min"], default="auto",
                    help="auto: use 1se unless it selects 0 features, then fallback to min")

    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    X, y, y_name = load_xy(
        args.feat_csv, args.il6_csv, args.target,
        args.y_area, args.y_il6,
        drop_dup_probcase=(not args.keep_dup_probcase)
    )
    print(f"[INFO] X shape={X.shape}, y shape={y.shape}")
    print(f"[INFO] y -> {y_name}")

    # Boruta
    print("[STEP] Boruta selecting...")
    boruta_feats = boruta_select(X, y, args.seed)

    # LASSO
    print("[STEP] LASSO CV/paths...")
    L = lasso_cv_and_paths(
        X, y, args.seed,
        args.kfolds, args.repeats,
        n_alphas=args.n_alphas,
        path_eps=args.path_eps
    )

    # 选择用于“特征输出 + OOF”的 alpha
    if args.lasso_rule == "min":
        lasso_feats = L["lasso_selected_min"]
        alpha_use = L["alpha_min"]
    elif args.lasso_rule == "1se":
        lasso_feats = L["lasso_selected_1se"]
        alpha_use = L["alpha_1se"]
    else:
        if len(L["lasso_selected_1se"]) == 0:
            lasso_feats = L["lasso_selected_min"]
            alpha_use = L["alpha_min"]
            print("[WARN] Lasso 1se selected 0 features -> fallback to alpha_min for model/OOF.")
        else:
            lasso_feats = L["lasso_selected_1se"]
            alpha_use = L["alpha_1se"]

    # --- 图件输出 ---
    f_net   = os.path.join(args.out_dir, f"{args.target}_boruta_lasso_network.png")
    f_paths = os.path.join(args.out_dir, f"{args.target}_lasso_paths.png")
    f_cv    = os.path.join(args.out_dir, f"{args.target}_lasso_cv.png")
    f_panel = os.path.join(args.out_dir, f"{args.target}_boruta_lasso_panel.png")
    f_map   = os.path.join(args.out_dir, f"{args.target}_lasso_path_line_map.csv")

    plot_network(boruta_feats, lasso_feats, f_net, "Feature Selection by Boruta and Lasso")

    # 关键：传 feature_names + 输出线条映射表
    plot_paths(
        L["alphas_path"], L["coefs_path"], f_paths,
        alpha_min=L["alpha_min"], alpha_1se=L["alpha_1se"],
        auto_trim=(not args.no_trim),
        max_lines=args.max_lines,
        feature_names=list(X.columns),
        legend=True,
        legend_max=args.legend_max,
        out_map_csv=f_map
    )

    plot_cvcurve(L["lcv"], L["alpha_min"], L["alpha_1se"], f_cv)
    compose_panel(f_net, f_paths, f_cv, f_panel)

    # OOF / permutation
    print("[STEP] OOF & permutation test...")
    r_obs, p_perm, oof = perm_p_value(X, y, alpha_use, args.seed, args.kfolds, args.repeats, args.perm)
    f_scatter = os.path.join(args.out_dir, f"{args.target}_oof_scatter.png")
    r_s, p_s = scatter_oof(
        y.values, oof,
        f"LASSO OOF — {args.target.upper()}",
        "Ground truth", "OOF prediction",
        f_scatter
    )

    # 保存摘要
    summary = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "boruta_selected": boruta_feats,
        "lasso_rule_used": args.lasso_rule,
        "alpha_used_for_oof": float(alpha_use),
        "lasso_selected_used": lasso_feats,
        "lasso_selected_min": L["lasso_selected_min"],
        "lasso_selected_1se": L["lasso_selected_1se"],
        "alpha_min": float(L["alpha_min"]),
        "alpha_1se": float(L["alpha_1se"]),
        "pearson_r_scatter": float(r_s),
        "pearson_p_scatter": float(p_s),
        "pearson_r_perm": float(r_obs),
        "perm_p_value": float(p_perm),
        "kfolds": args.kfolds,
        "repeats": args.repeats,
        "perm": args.perm,
        "y_name": y_name,
        "n_alphas": int(args.n_alphas),
        "path_eps": float(args.path_eps),
        "max_lines": int(args.max_lines),
        "auto_trim": bool(not args.no_trim),
        "legend_max": int(args.legend_max),
        "dropped_dup_probcase": bool(not args.keep_dup_probcase),
        "feature_names_for_path": list(X.columns),
        "path_line_map_csv": os.path.basename(f_map),
    }
    with open(os.path.join(args.out_dir, f"{args.target}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.Series(boruta_feats, name="boruta_selected").to_csv(
        os.path.join(args.out_dir, f"{args.target}_boruta_selected.csv"), index=False
    )
    pd.Series(lasso_feats, name="lasso_selected_used").to_csv(
        os.path.join(args.out_dir, f"{args.target}_lasso_selected.csv"), index=False
    )

    print(f"[OK] Boruta: {len(boruta_feats)} feats")
    print(f"[OK] LASSO used (alpha={alpha_use:.4g}): {len(lasso_feats)} feats")
    print(f"[OK] r={r_obs:.3f}, perm_p={p_perm:.3g}")
    print(f"[OK] saved -> {args.out_dir}")

if __name__ == "__main__":
    main()
