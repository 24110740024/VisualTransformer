# -*- coding: utf-8 -*-
import os, argparse, warnings, math, json
os.environ["MPLBACKEND"] = "Agg"   # 必须在导入 pyplot 前
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
    # 兼容 IL6 / il6 / IL-6 / il_6 等
    for c in df.columns:
        lc = c.lower().replace('-', '').replace('_', '')
        if lc == 'il6':
            return c
    raise KeyError("在 1IL6Area_clean.csv 里找不到 IL-6 列（IL6 / IL-6 / il6）")

def pick_area_col(df):
    # 兼容 area / y_area / lesion_area / area_mm2 / *area*
    for c in df.columns:
        lc = c.lower().replace(' ', '').replace('-', '_')
        if lc in ('area', 'y_area', 'lesion_area', 'area_mm2') or 'area' in lc:
            return c
    raise KeyError("在 1IL6Area_clean.csv 里找不到面积列（area/lesion_area/area_mm2 等）")

# ---------- 数据读取 ----------
def load_xy(feat_csv, il6_csv, target, y_area="sqrt", y_il6="log1p"):
    X = pd.read_csv(feat_csv)
    Y = pd.read_csv(il6_csv)

    # casekey / id 对齐
    def find_key(df):
        for k in ["casekey","case","id","key"]:
            if k in map(str.lower, df.columns.str.lower()):
                # 找到原始大小写列名
                for c in df.columns:
                    if c.lower()==k: return c
        raise KeyError(f"找不到 ID 列（casekey/case/id/key），现有列：{list(df.columns)}")
    ckey_x = find_key(X)
    ckey_y = find_key(Y)

    X = X.set_index(ckey_x)
    Y = Y.set_index(ckey_y)

    col_il6  = pick_il6_col(Y)
    col_area = pick_area_col(Y)

    if target == "il6":
        y = Y[col_il6].astype(float)
        if y_il6 == "log1p": y = np.log1p(y)
        y_name = f"{col_il6} ({'log1p' if y_il6=='log1p' else 'raw'})"
    else:
        y = Y[col_area].astype(float)
        if y_area == "sqrt": y = np.sqrt(y)
        elif y_area == "log1p": y = np.log1p(y)
        y_name = f"{col_area} ({'sqrt' if y_area=='sqrt' else ('log1p' if y_area=='log1p' else 'raw')})"

    # 只取两边都有的病例
    idx = X.index.intersection(y.index)
    X = X.loc[idx].copy()
    y = y.loc[idx].copy()

    # 去掉非数值列、明显的 ID/分组列
    bad = [c for c in X.columns if (X[c].dtype=="object") or (c.lower() in ["casekey","group","label","id","key"])]
    X = X.drop(columns=bad, errors="ignore")

    # 缺失处理
    X = X.replace([np.inf,-np.inf], np.nan).astype(float)
    X = X.fillna(X.median(numeric_only=True))

    return X, y, y_name

# ---------- Boruta ----------
def boruta_select(X, y, seed):
    rf = RandomForestRegressor(
        n_estimators=1000, max_depth=5, n_jobs=1, random_state=seed  # n_jobs=1 更稳
    )
    boruta = BorutaPy(rf, n_estimators='auto', max_iter=100, random_state=seed)
    boruta.fit(X.values, y.values)
    mask = boruta.support_
    return list(X.columns[mask])

# ---------- LASSO（CV/路径/α1se） ----------
def lasso_cv_and_paths(X, y, seed, kfolds=5, repeats=1):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    alphas = np.logspace(-4, 0, 80)
    lcv = LassoCV(alphas=alphas, cv=RepeatedKFold(n_splits=kfolds, n_repeats=repeats, random_state=seed),
                  random_state=seed, max_iter=50000)
    lcv.fit(Xs, y.values)

    # 计算 λ_min 与 λ_1se
    mse_mean = lcv.mse_path_.mean(axis=1)
    mse_std  = lcv.mse_path_.std(axis=1)
    idx_min  = np.argmin(mse_mean)
    mse_1se  = mse_mean[idx_min] + mse_std[idx_min]
    idx_1se  = np.where(mse_mean <= mse_1se)[0][-1]  # 右侧较大的 alpha
    alpha_min = lcv.alphas_[idx_min]
    alpha_1se = lcv.alphas_[idx_1se]

    # LASSO 路径
    alphas_path, coefs_path, _ = lasso_path(Xs, y.values, alphas=alphas, max_iter=50000)

    # 1se 下非零系数
    L1 = Lasso(alpha=alpha_1se, max_iter=50000).fit(Xs, y.values)
    coef_1se = L1.coef_
    lasso_feats = list(np.array(X.columns)[np.abs(coef_1se) > 1e-8])

    return {
        "lcv": lcv,
        "alpha_min": alpha_min,
        "alpha_1se": alpha_1se,
        "alphas_path": alphas_path,
        "coefs_path": coefs_path,
        "scaler": scaler,
        "lasso_selected": lasso_feats,
        "coef_1se": coef_1se,
    }

# ---------- 画图 ----------
def plot_network(selected_boruta, selected_lasso, out_png, title="Feature Selection by Boruta and Lasso"):
    G = nx.Graph()
    G.add_node("Boruta", type="method")
    G.add_node("Lasso", type="method")
    for f in selected_boruta:
        G.add_node(f, type="feat")
        G.add_edge("Boruta", f, w=1.0)
    for f in selected_lasso:
        G.add_node(f, type="feat")
        G.add_edge("Lasso", f, w=1.0)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6,6))
    methods = [n for n,d in G.nodes(data=True) if d["type"]=="method"]
    feats   = [n for n,d in G.nodes(data=True) if d["type"]=="feat"]
    nx.draw_networkx_nodes(G, pos, nodelist=methods, node_color="#9ecae1", node_size=900)
    nx.draw_networkx_nodes(G, pos, nodelist=feats,    node_color="#fdd0a2", node_size=300)
    nx.draw_networkx_edges(G, pos, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_paths(alphas, coefs, out_png):
    plt.figure(figsize=(7,4))
    for i in range(coefs.shape[0]):
        plt.plot(np.log10(alphas), coefs[i], lw=2)
    plt.xlabel("Log Lambda")
    plt.ylabel("Coefficients")
    plt.title("Lasso Paths")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_cvcurve(lcv, alpha_min, alpha_1se, out_png):
    a = lcv.alphas_
    m = lcv.mse_path_.mean(axis=1)
    s = lcv.mse_path_.std(axis=1)
    plt.figure(figsize=(7,4))
    plt.errorbar(a, m, yerr=s, fmt="o", ms=3, lw=1, alpha=0.6)
    plt.axvline(alpha_min, color="k",  ls="--", label=r"$\lambda_{min}$")
    plt.axvline(alpha_1se, color="C0", ls="--", label=r"$\lambda_{1se}$")
    plt.xscale("log")
    plt.xlabel("Alpha (λ) value")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Lasso CV Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def compose_panel(png_net, png_paths, png_cv, out_png):
    fig = plt.figure(figsize=(6,10))
    gs = GridSpec(3,1, height_ratios=[1,1,1], figure=fig)
    for i, img in enumerate([png_net, png_paths, png_cv]):
        ax = fig.add_subplot(gs[i,0])
        ax.imshow(plt.imread(img))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def scatter_oof(x, y, title, xlab, ylab, out_png):
    r, p = pearsonr(x, y)
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=36, alpha=0.9)
    # 拟合线
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
    pipe = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=50000))
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
    for _ in range(perm):
        y_perm = rng.permutation(y.values)
        oof_p = oof_predict(X, pd.Series(y_perm, index=y.index), alpha, seed, kfolds, repeats)
        r_p = pearsonr(oof_p, y_perm)[0]
        if abs(r_p) >= abs(r_obs): cnt += 1
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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X, y, y_name = load_xy(args.feat_csv, args.il6_csv, args.target, args.y_area, args.y_il6)
    print(f"[INFO] X shape={X.shape}, y shape={y.shape}")
    print(f"[INFO] y -> {y_name}")

    # Boruta
    print("[STEP] Boruta selecting...")
    boruta_feats = boruta_select(X, y, args.seed)

    # LASSO
    print("[STEP] LASSO CV/paths...")
    L = lasso_cv_and_paths(X, y, args.seed, args.kfolds, args.repeats)
    lasso_feats = L["lasso_selected"]

    # --- 图件输出 ---
    f_net   = os.path.join(args.out_dir, f"{args.target}_boruta_lasso_network.png")
    f_paths = os.path.join(args.out_dir, f"{args.target}_lasso_paths.png")
    f_cv    = os.path.join(args.out_dir, f"{args.target}_lasso_cv.png")
    f_panel = os.path.join(args.out_dir, f"{args.target}_boruta_lasso_panel.png")
    plot_network(boruta_feats, lasso_feats, f_net, "Feature Selection by Boruta and Lasso")
    plot_paths(L["alphas_path"], L["coefs_path"], f_paths)
    plot_cvcurve(L["lcv"], L["alpha_min"], L["alpha_1se"], f_cv)
    compose_panel(f_net, f_paths, f_cv, f_panel)

    # OOF / 置换
    print("[STEP] OOF & permutation test...")
    r_obs, p_perm, oof = perm_p_value(X, y, L["alpha_1se"], args.seed, args.kfolds, args.repeats, args.perm)
    f_scatter = os.path.join(args.out_dir, f"{args.target}_oof_scatter.png")
    r_s, p_s = scatter_oof(y.values, oof, f"LASSO OOF — {args.target.upper()}",
                           "Ground truth", "OOF prediction", f_scatter)

    # 保存摘要
    summary = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "boruta_selected": boruta_feats,
        "lasso_selected_alpha1se": lasso_feats,
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
    }
    with open(os.path.join(args.out_dir, f"{args.target}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 保存选特征列表
    pd.Series(boruta_feats, name="boruta_selected").to_csv(
        os.path.join(args.out_dir, f"{args.target}_boruta_selected.csv"),
        index=False
    )
    pd.Series(lasso_feats, name="lasso_selected_alpha1se").to_csv(
        os.path.join(args.out_dir, f"{args.target}_lasso_selected.csv"),
        index=False
    )

    print(f"[OK] Boruta: {len(boruta_feats)} feats")
    print(f"[OK] LASSO (alpha_1se={L['alpha_1se']:.4g}): {len(lasso_feats)} feats")
    print(f"[OK] r={r_obs:.3f}, perm_p={p_perm:.3g}")
    print(f"[OK] saved -> {args.out_dir}")

if __name__ == "__main__":
    main()
