"""CPU-only checks of published probabilities, animal means and all40 correlations."""
import argparse,csv,gzip,io,json,math,sys,zipfile
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'rebuild_v6'))
from data import read_rows
from manifest_archive import open_manifest_archive
from aggregate import aggregate
from metrics import select_validation_threshold,metrics,bootstrap_auc
from association_core import paired_bootstrap_ci
from robustness import compute,holm

def near(a,b):
    if not math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12):raise ValueError(f'Mismatch: {a} vs {b}')

def run(full_bootstrap=False):
    r=ROOT/'results';t=r/'tables';v=read_rows(r/'predictions/a6_v6_val.csv');test=read_rows(r/'predictions/a6_v6_test.csv')
    if len(v)!=7909 or len(test)!=8704:raise ValueError('A6 row counts changed')
    for rr in (v,test):
        if len({row['pixel_sha256'] for row in rr})!=len(rr):raise ValueError('Repeated A6 pixels')
        if [int(x['manifest_row_index']) for x in rr]!=list(range(len(rr))):raise ValueError('A6 order changed')
        if any(int(x['y_lesion'])!=int(x['label_index']=='0') for x in rr):raise ValueError('Wrong positive class')
    if {x['source_id'] for x in v}&{x['source_id'] for x in test}:raise ValueError('Validation/test source leakage')
    vy=np.array([int(x['y_lesion']) for x in v]);vp=np.array([float(x['prob_lesion']) for x in v]);y=np.array([int(x['y_lesion']) for x in test]);p=np.array([float(x['prob_lesion']) for x in test])
    lock=select_validation_threshold(vy,vp);saved=json.loads((t/'validation_threshold_v6.json').read_text());near(lock['threshold'],saved['threshold'])
    measured=metrics(y,p,lock['threshold']);summary=next(x for x in read_rows(t/'a6_all_six_models.csv') if x['version']=='Balanced-sampling ViT v6')
    for k in ('accuracy','precision','recall','specificity','f1','TN','FP','FN','TP'):near(measured[k],summary[k])
    near(measured['auc_lesion'],summary['auc'])
    with open_manifest_archive() as z:coords=list(csv.DictReader(io.StringIO(z.read('coordinates_all21.csv').decode())))
    runs=[]
    for seed in (42,43,44):
        for fold in (1,2,3):
            with gzip.open(r/'predictions'/f'a5_seed{seed}_fold{fold}.csv.gz','rt',encoding='utf-8',newline='') as f:rr=list(csv.DictReader(f))
            runs.append((seed,fold,rr))
    scores=aggregate(coords,runs);saved_scores={x['case_id']:x for x in read_rows(t/'case_scores_equal_seed_mean.csv')}
    for x in scores:
        old=saved_scores[x['case_id']]
        for k in ('oof_prob_mean_raw','seed42_mean_probability','seed43_mean_probability','seed44_mean_probability'):near(x[k],old[k])
    wide=read_rows(t/'animal_scores_all_versions.csv');associations=read_rows(t/'all_associations_v2_v3_v4_v5_v6.csv')
    if len(wide)!=21 or len({r['case_id'] for r in wide})!=21 or len(associations)!=40:raise ValueError('Wrong animal or result count')
    for version in ('v2','v3','v4','v5','v6'):
        primary=[x for x in associations if x['version']==version and x['analysis']=='primary21']
        for result in [x for x in associations if x['version']==version]:
            cohort=sorted([x for x in wide if result['analysis']=='primary21' or not x['case_id'].startswith('30-30-120-N-')],key=lambda x:x['case_id'])
            x=np.array([float(r[version+'_oof_prob_mean_raw']) for r in cohort]);yy=np.array([float(r[result['outcome']]) for r in cohort]);fn=stats.pearsonr if result['method']=='Pearson' else stats.spearmanr
            coefficient,pvalue=fn(x,yy);near(coefficient,result['coefficient']);near(pvalue,result['p_nominal']);near(len(cohort),result['n'])
            if full_bootstrap and result['analysis']=='primary21' and version=='v6':
                ci=paired_bootstrap_ci(x,yy,lambda a,b:fn(a,b)[0]);near(ci['low'],result['ci_low']);near(ci['high'],result['ci_high'])
        for result,adj in zip(primary,holm([float(r['p_nominal']) for r in primary])):near(adj,result['p_holm_family4'])
    joined=read_rows(t/'v6_all21_animal_scores.csv');main,influence,within=compute(joined)
    for file,actual in [('holm_primary4.csv',main),('influence_all112.csv',influence),('within_group_descriptive.csv',within)]:
        old=read_rows(t/file)
        if len(old)!=len(actual):raise ValueError('Diagnostic row count changed')
        for a,b in zip(actual,old,strict=True):
            for k,value in a.items():
                if isinstance(value,(int,float)):near(value,b[k])
                elif value is None:
                    if b[k]!='':raise ValueError('Undefined diagnostic changed')
                elif str(value)!=b[k]:raise ValueError('Diagnostic identity mismatch')
    if full_bootstrap:
        ids=np.array([r['source_id'] for r in test])
        order=json.loads((t/'a6_source_order.json').read_text())
        boot=bootstrap_auc(y,p,ids,order)
        near(boot['source_wsi_cluster']['percentile_95_ci'][0],summary['ci_low']);near(boot['source_wsi_cluster']['percentile_95_ci'][1],summary['ci_high'])
    return dict(status='pass',a6_validation_rows=7909,a6_test_rows=8704,a5_predictions_checked=sum(len(x[2]) for x in runs),animal_seed_means=63,animals=21,associations_checked=40,influence_checks=112,full_bootstrap=full_bootstrap,model_retrained=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--full-bootstrap',action='store_true');ap.add_argument('--receipt',type=Path);a=ap.parse_args()
    result=run(a.full_bootstrap);print(json.dumps(result,indent=2))
    if a.receipt:a.receipt.write_text(json.dumps(result,indent=2)+'\n')
