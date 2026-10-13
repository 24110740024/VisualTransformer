"""Execute the recovered separate Figure 1F/1H analysis; no model selection."""
from pathlib import Path
import ast
import hashlib
import itertools
import json
import math
import os
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(version, audit_dir):
    from input_gate import checked
    checked(audit_dir)
    assert version == "v6"
    root = HERE / "balance_candidate_v6"
    old = HERE / "recovered_protocol/historical_figure1F1H_group_statistics.py.txt"
    tree = ast.parse(old.read_text(encoding="utf-8-sig"))
    names = {"exact_two_sided_permutation_p", "hedges_g", "balanced_2x2_anova"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    namespace = {"np": np, "pd": pd, "stats": stats, "math": math, "itertools": itertools}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(old), "exec"), namespace)
    perm, hedges, anova_func = [namespace[n] for n in
                              ("exact_two_sided_permutation_p", "hedges_g", "balanced_2x2_anova")]
    # Meaningful checks independent of new observed scores: all assignments and OLS contrasts.
    assert perm([0, 1, 2], [3, 4, 5]) == 0.1
    assert perm([0, 1, 2], [0, 1, 2]) == 1.0
    assert math.isclose(hedges([0, 1, 2], [3, 4, 5]), 2.4)
    synthetic = pd.DataFrame([
        {"temperature": t, "ozone": o, "genotype": "N",
         "pathology_score": 0.3 + 0.1*a + 0.05*b + 0.02*a*b + e}
        for t, a in [(30, -1), (36, 1)] for o, b in [(80, -1), (120, 1)]
        for e in [-0.03, 0.01, 0.02]])
    a = (synthetic.temperature.to_numpy() - 33) / 3
    b = (synthetic.ozone.to_numpy() - 100) / 20
    X = np.column_stack([np.ones(12), a, b, a*b])
    y = synthetic.pathology_score.to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    mse = np.sum((y - X @ beta)**2) / 8
    expected_F = beta[1:]**2 / (mse * np.diag(np.linalg.inv(X.T @ X))[1:])
    assert np.allclose(anova_func(synthetic).F, expected_F)

    state = json.loads((root / "jobs/pipeline_status.json").read_text())
    assert state["phase"] == "core_complete"
    source = Path(state["statistics"]).resolve()
    assert source.is_relative_to(root.resolve() / "statistics")
    stage = json.loads((root / "jobs/a5_statistics.done.json").read_text())
    assert Path(stage["output"]).resolve() == source
    complete = json.loads((source / "COMPLETE.json").read_text())
    expected_complete = stage["complete_sha256"] if version == "v2" else stage["outputs_sha256"]["COMPLETE.json"]
    assert sha(source / "COMPLETE.json") == expected_complete
    assert sha(source / "joined21.csv") == complete["outputs_sha256"]["joined21.csv"]
    out = HERE / "reports" / ("group_" + version + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    if out.exists():
        raise FileExistsError("Preserve prior analysis; create a new version explicitly")
    out.mkdir(parents=True)
    groups = {"30-40-80-N": "G3", "36-40-80-N": "G4", "36-40-80-A": "G5",
              "30-30-120-N": "G6", "36-30-120-N": "G7", "36-30-120-A": "G8",
              "36-20-160-A": "G9"}
    planned = [("Temperature: WT, 80 ppb / 40% RH", "G3", "G4"),
               ("Temperature: WT, 120 ppb / 30% RH", "G6", "G7"),
               ("Genotype: 36 C, 80 ppb / 40% RH", "G4", "G5"),
               ("Genotype: 36 C, 120 ppb / 30% RH", "G7", "G8")]
    spec = {"purpose": "Separate historical Figure1F1H method rerun; not A5 outcome adjustment", "version": version,
            "specification_origin": "Recovered historical script and user terminal; not a new preregistration",
            "historical_script_sha256": sha(old), "input_sha256": sha(source / "joined21.csv"),
            "input": str(source / "joined21.csv"), "groups": groups, "planned_contrasts": planned,
            "anova": "balanced WT 2x2 temperature by O3/RH scenario, 12 animals, error df8",
            "contrast_methods": "two-sided Welch t; exact all20-label-assignment permutation; Hedges g B-A",
            "multiplicity": "unadjusted; all historical comparisons reported",
            "G9": "descriptive only, no matched WT, no inference of a pure ozone effect",
            "created_utc": datetime.now(timezone.utc).isoformat()}
    (out / "ANALYSIS_SPEC.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    df = pd.read_csv(source / "joined21.csv", float_precision="round_trip")
    assert len(df) == 21 and df.case_id.is_unique and df.source_sha256.is_unique
    parts = df.case_id.str.split("-", expand=True)
    assert parts.shape[1] == 5
    df["temperature"], df["rh"], df["ozone"] = [parts[i].astype(int) for i in range(3)]
    df["genotype"], df["replicate"] = parts[3], parts[4].astype(int)
    df["group"] = df.case_id.str.rsplit("-", n=1).str[0].map(groups)
    assert df.group.notna().all() and (df.groupby("group").size() == 3).all()
    order = list(groups.values())
    summary = df.groupby("group").pathology_score.agg(["count", "mean", "std", "min", "max"]).reindex(order)
    anova = anova_func(df)
    rows = []
    for label, ga, gb in planned:
        a, b = [df.loc[df.group == g, "pathology_score"].to_numpy() for g in (ga, gb)]
        welch = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"contrast": label, "group_A": ga, "group_B": gb, "n_A": 3, "n_B": 3,
                     "difference_B_minus_A": b.mean()-a.mean(), "Welch_t": float(welch.statistic),
                     "Welch_p_nominal": float(welch.pvalue), "exact_permutation_p_nominal": perm(a,b),
                     "Hedges_g_B_minus_A": hedges(a,b)})
    contrasts = pd.DataFrame(rows)
    df.to_csv(out / "animals_with_groups.csv", index=False)
    summary.to_csv(out / "group_summary.csv")
    anova.to_csv(out / "WT_factorial_ANOVA.csv", index=False)
    contrasts.to_csv(out / "planned_contrasts.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    labels = []
    for i, (base, group) in enumerate(groups.items()):
        t, rh, o3, genotype = base.split("-")
        labels.append(f"{group}\n{'WT' if genotype == 'N' else 'ApoE-/-'} {t} C\n{o3} ppb / {rh}% RH")
        g = df[df.group == group].sort_values("replicate")
        ax.scatter(i + np.array([-0.08, 0, 0.08]), g.pathology_score, s=45,
                   color="#256D8D" if genotype == "N" else "#B75A3B", zorder=3)
        ax.errorbar(i, summary.loc[group, "mean"], yerr=summary.loc[group, "std"],
                    fmt="_", color="#202020", markersize=17, capsize=5, lw=1.3, zorder=4)
    # Mean +/- SD is descriptive and may extend beyond the support of individual scores.
    lower = min(0.0, float((summary["mean"]-summary["std"]).min())-0.025)
    upper = max(1.0, float((summary["mean"]+summary["std"]).max())+0.025)
    ax.set(xticks=range(7), xticklabels=labels, ylim=(lower,upper), ylabel="Cross-fitted pathology score",
           title=f"{version} animal scores by exposure group")
    ax.tick_params(axis="x", labelsize=8)
    ax.text(0.01, 0.98, "Dots: individual animals (n=3/group); bars: mean +/- SD\nG9 is descriptive only; axis includes full SD bars", va="top", transform=ax.transAxes, fontsize=9)
    fig.savefig(out / "group_scores.png", dpi=300)
    fig.savefig(out / "group_scores.svg")
    plt.close(fig)
    lines = [f"# {version} 分组分析：恢复的 Figure 1F/1H 方法", "",
             "使用本次固定21动物评分，恢复旧方案的分组和比较。它独立于Area/IL6关联，不用于选择模型或调整主评分。", "",
             "WT四组的温度×O3/RH场景双因素ANOVA（12动物，误差自由度8）：", "",
             "| 效应 | F | 上尾名义p（F检验） | partial eta² |", "|---|---:|---:|---:|"]
    for r in anova.to_dict("records"):
        lines.append(f"| {r['effect']} | {r['F']:.6f} | {r['p']:.6g} | {r['partial_eta2']:.6f} |")
    lines += ["", "历史指定的全部四个组间对比（每组3动物）：", "",
              "| A→B | 均值差B−A | Welch名义p | 精确置换名义p | Hedges g |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['group_A']}→{r['group_B']} | {r['difference_B_minus_A']:.6f} | {r['Welch_p_nominal']:.6g} | {r['exact_permutation_p_nominal']:.2f} | {r['Hedges_g_B_minus_A']:.6f} |")
    lines += ["", "所有p均未作多重比较校正。O3与RH共同变化，场景效应不是纯臭氧效应；G9没有匹配WT，仅描述。n=3的两组双侧全枚举置换只有20种分配，最小p为0.1，不能以Welch较小p替代或隐去这一结果。模型产生的评分和同组暴露设计均限制推断，当前统计以已拟合评分为条件，不包含重新训练的不确定性。", "",
              "图示全部21动物和均值±SD，不使用显著性星号。纵轴包含完整误差线；误差线可超出单个概率的0–1范围，不是新概率值或置信区间。分析函数直接提取自已读过的恢复脚本，执行前在独立合成数据上通过OLS方差分析对照、置换极值和Hedges g检查。", "",
              "![全部动物分组图](group_scores.png)"]
    (out / "分组结果与局限.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    audit = {"status": "numerically_checked_visual_review_pending", "script_sha256": sha(Path(__file__)),
             "checks": ["source completion/input hashes", "21 unique sources and 3 per group", "synthetic OLS ANOVA equivalence", "permutation boundary cases", "Hedges g known case"],
             "outputs_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / "report_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--version",choices=["v6"],required=True)
    parser.add_argument("--audit-dir",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps({"output":str(main(args.version,args.audit_dir))}))
