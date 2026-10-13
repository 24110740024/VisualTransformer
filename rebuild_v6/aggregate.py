"""Verify all nine coordinate-keyed outputs before equal-seed animal means."""
import argparse,json,math,itertools,gzip
from collections import defaultdict,Counter
from pathlib import Path
import csv
from data import read_rows,write_csv,sha

SEEDS=(42,43,44)
def key(r):return (r['case_id'],r['source_sha256'],int(r['x']),int(r['y']),int(r['crop_size']),int(r['level']))

def aggregate(coordinates,predictions):
    expected={key(r):int(r['outer_fold']) for r in coordinates}
    if len(expected)!=len(coordinates) or len(expected)!=132032:raise ValueError('Require all132032 unique coordinates')
    cases={r['case_id']:r['source_sha256'] for r in coordinates};counts=Counter(r['case_id'] for r in coordinates)
    if len(cases)!=21 or len(set(cases.values()))!=21:raise ValueError('Require21 sources and animals')
    seen_runs=set();per_seed={}
    for seed,fold,rows in predictions:
        if (seed,fold) in seen_runs or (seed,fold) not in itertools.product(SEEDS,(1,2,3)):raise ValueError('Repeated/unplanned run')
        seen_runs.add((seed,fold));seen=set();values=defaultdict(list);checkpoints=set()
        for r in rows:
            k=key(r)
            if k not in expected or expected[k]!=fold or k in seen:raise ValueError('Unexpected/repeated tile identity')
            if int(r['seed'])!=seed or int(r['outer_fold'])!=fold:raise ValueError('Wrong seed/fold')
            v=float(r['prob_lesion'])
            if not math.isfinite(v) or not 0<=v<=1:raise ValueError('Invalid probability')
            checkpoints.add(r['checkpoint_sha256']);seen.add(k);values[r['case_id']].append(v)
        if len(checkpoints)!=1 or seen!={k for k,v in expected.items() if v==fold}:raise ValueError('Missing tiles or mixed checkpoints')
        for case,v in values.items():per_seed[seed,case]=math.fsum(v)/len(v)
    if seen_runs!=set(itertools.product(SEEDS,(1,2,3))):raise ValueError('Require all nine runs')
    out=[]
    for case in sorted(cases):
        values=[per_seed[s,case] for s in SEEDS]
        out.append(dict(case_id=case,source_sha256=cases[case],outer_fold=int(case[-1]),n_tiles=counts[case],seed_count=3,
                        **{f'seed{s}_mean_probability':v for s,v in zip(SEEDS,values)},oof_prob_mean_raw=math.fsum(values)/3))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--coordinates',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs=9,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    predictions=[]
    for path in a.runs:
        marker=json.loads((path/'COMPLETE.json').read_text());file=path/'predictions.csv'
        if marker['status']!='complete' or marker['predictions_sha256']!=sha(file) or marker['coordinate_sha256']!=sha(a.coordinates):raise ValueError('Unfinished or changed inference')
        predictions.append((marker['seed'],marker['fold'],read_rows(file)))
    result=aggregate(read_rows(a.coordinates),predictions)
    a.out.mkdir(parents=True,exist_ok=False);write_csv(a.out/'animal_scores.csv',result)
    print(json.dumps(dict(animals=len(result),output=str(a.out))))

if __name__=='__main__':main()
