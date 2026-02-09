import os, re, argparse, numpy as np, pandas as pd

def norm_key(s: str) -> str:
    s = str(s).strip().split()[0]
    s = re.sub(r'[-_]+$','',s); s = re.sub(r'[^0-9A-Za-z]+','-',s)
    return re.sub(r'-{2,}','-',s).strip('-')

def get_case_col(df):
    for c in ["slide_id","case","casekey","wsi","id"]:
        if c in df.columns: return c
    if "path" in df.columns:
        df["_case_from_path"] = df["path"].apply(
            lambda p: os.path.basename(os.path.dirname(p)) if isinstance(p,str) else None)
        return "_case_from_path"
    raise ValueError("找不到 case/slide 标识列（期待 slide_id/case/casekey/wsi/id 或 path）")

def get_prob_col(df, prefer=None):
    if prefer and prefer in df.columns:
        return prefer
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["prob_lesion","prob","p_pos","p1","score","y_prob","yhat","pred"]):
            candidates.append(c)
    if candidates:
        for key in ["prob_lesion","p_pos","prob","y_prob","score","p1","yhat","pred"]:
            for c in candidates:
                if c.lower().startswith(key):
                    return c
        return candidates[0]
    raise ValueError("找不到概率列（可用 --prob_col 指定）")

def topk_mean(v: np.ndarray, frac=0.05) -> float:
    v = np.asarray(v, float)
    if v.size == 0: return 0.0
    k = max(1, int(np.ceil(frac * v.size)))
    return float(np.mean(np.partition(v, -k)[-k:]))

def logistic_pool(v: np.ndarray) -> float:
    v = np.clip(np.asarray(v, float), 1e-6, 1-1e-6)
    z = np.log(v/(1-v)).mean()
    return float(1 / (1 + np.exp(-z)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_csv", required=True, help="all_patches.csv")
    ap.add_argument("--out_csv",   required=True, help="输出 case_scores.csv")
    ap.add_argument("--topk",      type=float, default=0.05, help="top-k 比例，默认 0.05")
    ap.add_argument("--prob_col",  type=str, default="", help="手动指定概率列名（如 prob_lesion）")
    args = ap.parse_args()

    df = pd.read_csv(args.patch_csv)
    case_col = get_case_col(df)
    prob_col = get_prob_col(df, args.prob_col or None)

    df["_casekey"] = df[case_col].astype(str).map(norm_key)
    gp = df.groupby("_casekey")[prob_col]

    rows = []
    for ck, g in gp:
        v = g.values
        rows.append(dict(
            casekey   = ck,
            prob_topk = topk_mean(v, args.topk),
            prob_mean = float(np.mean(v)) if v.size else 0.0,
            prob_max  = float(np.max(v))  if v.size else 0.0,
            prob_logm = logistic_pool(v),
        ))
    out = pd.DataFrame(rows)
    out["prob_case"] = out["prob_topk"] 
    out.to_csv(args.out_csv, index=False)
    print("[OK] saved", args.out_csv, "rows=", len(out))
    print("Columns:", list(out.columns))

if __name__ == "__main__":
    main()
