"""Raw-outcome associations and fixed diagnostics; no score fitting or selection."""
import argparse,json
from pathlib import Path
from data import read_rows,write_csv
from association_core import association_results
from robustness import compute

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scores',type=Path,required=True);p.add_argument('--outcomes',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    scores=read_rows(a.scores);outcomes=read_rows(a.outcomes)
    si={r['case_id']:r for r in scores};oi={r['case_id']:r for r in outcomes}
    if len(si)!=len(scores) or len(oi)!=len(outcomes) or set(si)!=set(oi) or len(si)!=21:raise ValueError('Require exact21 unique identities on both sides')
    joined=[]
    for case in sorted(si):
        s,o=si[case],oi[case]
        if s['source_sha256']!=o['source_sha256'] or int(s['outer_fold'])!=int(o['outer_fold']):raise ValueError('Outcome identity mismatch')
        joined.append(dict(case_id=case,pathology_score=float(s['oof_prob_mean_raw']),Area=float(o['Area']),IL6=float(o['IL6'])))
    primary=association_results(joined,'primary21',True)
    secondary=association_results([r for r in joined if not r['case_id'].startswith('30-30-120-N-')],'secondary18_fixed_alias_history_exclusions',False)
    main,influence,within=compute(joined)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'associations.json').write_text(json.dumps(primary+secondary,indent=2))
    for name,rr in [('joined21',joined),('holm_primary4',main),('influence_all112',influence),('within_group_descriptive',within)]:write_csv(a.out/(name+'.csv'),rr)
    print('Completed fixed21 analysis; secondary18 does not replace the primary cohort.')

if __name__=='__main__':main()
