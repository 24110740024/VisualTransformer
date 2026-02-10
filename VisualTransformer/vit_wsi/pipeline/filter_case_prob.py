import argparse, os, pandas as pd, glob, re

def norm(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'[ _]+HE_Wholeslide.*$', '', s)
    s = re.sub(r'[^0-9A-Za-z\-]+', '-', s)
    s = re.sub(r'-{2,}', '-', s).strip('-')
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--case_csv', default='case_scores.csv')
    ap.add_argument('--il6_csv',  default='1IL6Area_clean.csv')
    ap.add_argument('--raw_dir',  default='_rawgrid')
    ap.add_argument('--out_csv',  default='case_scores_filtered.csv')
    args = ap.parse_args()

    run = args.run
    case_csv = os.path.join(run, args.case_csv)
    il6_csv  = os.path.join(run, args.il6_csv)
    raw_dir  = os.path.join(run, args.raw_dir)
    out_csv  = os.path.join(run, args.out_csv)

    cs  = pd.read_csv(case_csv)
    il6 = pd.read_csv(il6_csv)[['casekey']].copy()

    cs['casekey']  = cs['casekey'].map(norm)
    il6['casekey'] = il6['casekey'].map(norm)

    keep = set(il6['casekey'])

    keys_raw = set()
    for p in glob.glob(os.path.join(raw_dir, '*_heat_raw.npy')):
        key = os.path.basename(p).replace('_heat_raw.npy', '')
        keys_raw.add(norm(key))
    if keys_raw:
        keep = keep & keys_raw

    cs_f = cs[cs['casekey'].isin(keep)].reset_index(drop=True)
    cs_f.to_csv(out_csv, index=False)
    dropped = sorted(set(cs['casekey']) - set(cs_f['casekey']))
    print(f'[OK] 写入 {out_csv}  共 {len(cs_f)} 行；丢弃 {len(dropped)} 行')
    if dropped:
        print('[DROP 示例] ', dropped[:6])

if __name__ == "__main__":
    main()
