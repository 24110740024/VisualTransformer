"""One-time CPU QA and report protocol binding, using synthetic study values only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,copy,json,sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score,confusion_matrix
from report import HERE,NEW,KEYS,write,table,all_roc,append_associations
from figures import figure_a5,figure_a6,sha


def rejection(fn):
    try:fn()
    except (ValueError,AssertionError):return
    raise AssertionError('Invalid case accepted')


def freeze():
    assert not (HERE/'PROTOCOL.json').exists(),'Do not overwrite frozen protocol'
    out=HERE/datetime.now(timezone.utc).strftime('synthetic_QA_%Y%m%dT%H%M%S_%fZ');out.mkdir()
    rng=np.random.default_rng(763)
    cases=[f'{g}-{i}' for g in ('30-40-80-N','36-40-80-N','36-40-80-A','30-30-120-N','36-30-120-N','36-30-120-A','36-20-160-A') for i in (1,2,3)]
    xx=rng.uniform(.1,.8,21);yy=rng.uniform(0,20,21);zz=rng.uniform(50,150,21)
    joined=[dict(case_id=c,pathology_score=x,Area=y,IL6=z) for c,x,y,z in zip(cases,xx,yy,zz,strict=True)]
    results=[];holm=[]
    for analysis,n in [('primary21',21),('secondary18_fixed_alias_history_exclusions',18)]:
        for outcome,y in [('Area',yy),('IL6',zz)]:
            for method,f in [('Pearson',stats.pearsonr),('Spearman',stats.spearmanr)]:
                r,p=f(xx[:n],y[:n]);row=dict(analysis=analysis,n_animals=n,outcome=outcome,statistic=method,coefficient=float(r),p_two_sided_nominal=float(p))
                if n==21:
                    row['paired_animal_percentile95_CI']={'low':-1.,'high':1.}
                    holm.append(dict(outcome=outcome,statistic=method,coefficient=float(r),p_nominal=float(p),p_holm_family4=1.))
                results.append(row)
    old=[]
    for v in ('v2','v3','v4','v5'):
        for r in results:old.append(dict(version=v,analysis=r['analysis'],outcome=r['outcome'],method=r['statistic'],n=r['n_animals'],coefficient=r['coefficient'],p_nominal=r['p_two_sided_nominal'],p_holm_family4='',ci_low='',ci_high=''))
    combined=append_associations(old,results,holm);assert len(combined)==40 and combined[:32]==old
    assert len([r for r in combined if r['version']=='v6' and r['n']==21])==4
    bad=copy.deepcopy(results);bad[0]['n_animals']=63;rejection(lambda:append_associations(old,bad,holm))
    bad=copy.deepcopy(results);bad[1]=bad[0];rejection(lambda:append_associations(old,bad,holm))
    bad=copy.deepcopy(holm);bad[0]['p_nominal']+=.01;rejection(lambda:append_associations(old,results,bad))
    rejection(lambda:append_associations(old[:-1],results,holm))
    figure_a5(joined,results,holm,out)
    # 8,704 synthetic rows, four source identities; no real model probabilities.
    y=np.arange(8704)%2;p=np.clip(.2+.55*y+rng.normal(0,.25,len(y)),0,1)
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=.5).ravel());auc=float(roc_auc_score(y,p))
    m=dict(n=len(y),lesion_n=int(y.sum()),normal_n=int((y==0).sum()),auc_lesion=auc,TN=tn,FP=fp,FN=fn,TP=tp,
           accuracy=(tn+tp)/len(y),precision=tp/(tp+fp),recall=tp/(tp+fn),specificity=tn/(tn+fp),f1=2*tp/(2*tp+fp+fn))
    rr=[]
    for i,(target,prob) in enumerate(zip(y,p,strict=True)):
        row={k:f'{k}-{i}' for k in KEYS};row.update(source_id=f'source{i%4}',y_lesion=int(target),prob_lesion=float(prob));rr.append(row)
    ev=out/'synthetic_evaluation';ev.mkdir();table(ev/'test_predictions.csv',rr)
    summary=dict(test=m,bootstrap={'source_wsi_cluster':{'percentile_95_ci':[.5,1.]}})
    write(ev/'evaluation_summary.json',summary)
    audit=dict(prediction_files_sha256={'test':sha(ev/'test_predictions.csv')},recomputed_test_metrics=m,validation_threshold=.5)
    figure_a6(ev,audit,out)
    metrics=[dict(version=v,auc=auc) for v in ('v2','v3','ResNet','v4','v5','v6')]
    prior={'prediction_files_sha256':{}}
    for i in range(5):
        folder=out/f'synthetic_comparison{i}';folder.mkdir();table(folder/'test_predictions.csv',rr)
        prior['prediction_files_sha256'][str(folder/'test_predictions.csv')]=sha(folder/'test_predictions.csv')
    assert len(all_roc(prior,metrics,ev,audit,out))==6
    bad=copy.deepcopy(metrics);bad[-1]['auc']-=.01
    rejection(lambda:all_roc(prior,bad,ev,audit,out/'should_never_write'))
    rr[0]['pixel_sha256']='wrong-case';table(ev/'shifted.csv',rr)
    changed=copy.deepcopy(prior);first=next(iter(changed['prediction_files_sha256']));del changed['prediction_files_sha256'][first]
    changed['prediction_files_sha256'][str(ev/'shifted.csv')]=sha(ev/'shifted.csv')
    rejection(lambda:all_roc(changed,metrics,ev,audit,out/'should_never_write'))
    # Undefined threshold must be reported explicitly, without invented confusion counts.
    undef=out/'synthetic_undefined';undef.mkdir()
    table(undef/'test_predictions.csv',rr)
    mu=dict(m)
    for k in ('TN','FP','FN','TP','accuracy','precision','recall','specificity','f1'):mu[k]=None
    write(undef/'evaluation_summary.json',dict(test=mu,bootstrap=summary['bootstrap']))
    au=dict(prediction_files_sha256={'test':sha(undef/'test_predictions.csv')},recomputed_test_metrics=mu,validation_threshold=None)
    figure_a6(undef,au,undef)
    assert 'seeds 42' in (HERE/'figures.py').read_text(encoding='utf-8')
    oldtree=ast.parse((NEW/'optimization_review_20260926/render_v4_verified_report.py').read_text(encoding='utf-8'))
    newtree=ast.parse((HERE/'verify_groups.py').read_text(encoding='utf-8'))
    a=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_group')
    b=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_group')
    assert [ast.dump(n) for n in a.body]==[ast.dump(n) for n in b.body]
    # Full synthetic group path, including producer stage identity and independent equations.
    from verify_groups import verify_group
    import groups,input_gate
    synthetic_root=out/'synthetic_group_root';(synthetic_root/'recovered_protocol').mkdir(parents=True)
    original=NEW/'recovered_protocol/historical_figure1F1H_group_statistics.py.txt'
    (synthetic_root/'recovered_protocol'/original.name).write_bytes(original.read_bytes())
    root=synthetic_root/'balance_candidate_v6';study=root/'statistics/attempt_synthetic';study.mkdir(parents=True)
    (root/'jobs').mkdir()
    table(study/'joined21.csv',[dict(r,source_sha256=f'synthetic-source-{i}') for i,r in enumerate(joined)])
    write(study/'COMPLETE.json',dict(outputs_sha256={'joined21.csv':sha(study/'joined21.csv')}))
    write(root/'jobs/a5_statistics.done.json',dict(output=str(study),outputs_sha256={'COMPLETE.json':sha(study/'COMPLETE.json')}))
    write(root/'jobs/pipeline_status.json',dict(phase='core_complete',statistics=str(study)))
    saved_here,saved_gate=groups.HERE,input_gate.checked
    try:
        groups.HERE=synthetic_root;input_gate.checked=lambda directory:None
        group=groups.main('v6',out/'SYNTHETIC_AUDIT_ONLY');group_audit=verify_group(group)
    finally:groups.HERE=saved_here;input_gate.checked=saved_gate
    write(out/'synthetic_group_independent_audit.json',group_audit)
    (out/'SYNTHETIC_ONLY_DO_NOT_USE.md').write_text('Synthetic QA data and placeholder CI/Holm only. No observed study outcomes or v6 A5 predictions. Do not use these figures in the manuscript.\n',encoding='utf-8')
    files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.md')]
    files += [NEW/'v6_visualization_v1/prepare_visuals.py',NEW/'v6_visualization_v1/VISUAL_PROTOCOL.json',
        NEW/'v6_postprocessing_v1/PROTOCOL.json',NEW/'v6_postprocessing_v1/run_audit.py',
        NEW/'balance_candidate_v6/EXECUTION_PROTOCOL.json',NEW/'v6_independent_audits/a6_20260926T090443Z.json',
        NEW/'recovered_protocol/historical_figure1F1H_group_statistics.py.txt',
        NEW/'reports/verified_v5_standard_20260926T085652_505320Z/REPORT_COMPLETE.json',
        NEW/'reports/verified_v5_standard_20260926T085652_505320Z/REVIEW_COMPLETE.json']
    write(HERE/'CPU_QA.json',dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),synthetic_directory=str(out),
        study_outcomes_or_candidate_A5_probabilities_read=False,tests=['32+8 association rows with n/duplicate/Holm/missing-prior refusal',
        'Three-seed labels; no synthetic animal replication','8704-row synthetic A6 and six-model identity/AUC checks',
        'Wrong AUC and changed pixel identity rejected','Undefined threshold explicitly shown',
        'Full synthetic 21-animal group stage and independent OLS/all20 permutations/Welch/Hedges',
        'Group independent verification function AST unchanged'],synthetic_CI_and_adjusted_p_are_placeholders=True,
        actual_visual_review_pending=True,gpu_imported='torch' in sys.modules))
    assert 'torch' not in sys.modules
    write(HERE/'PROTOCOL.json',dict(protocol='V6_STANDARD_REPORTS_V2',files_sha256={str(p):sha(p) for p in files},
        cpu_qa_sha256=sha(HERE/'CPU_QA.json'),seed_replicates=3,new_predictions=396096,new_animal_scores=21,
        report_rule='All six classifiers and five association versions retained; no model selection',prior_results_known=True))
    from input_gate import checked
    try:checked(NEW/'v6_postprocessing_v1/nonexistent')
    except ValueError as exc:
        assert 'Wait for all candidate outputs' in str(exc),str(exc)
        write(HERE/'POSTFREEZE_CHECK.json',dict(status='passed',real_incomplete_state_refused=str(exc),
            candidate_A5_probabilities_or_outcomes_read=False,protocol_sha256=sha(HERE/'PROTOCOL.json')))
    else:raise AssertionError('Incomplete candidate accepted')
    print(json.dumps({'status':'prepared_and_frozen','protocol_sha256':sha(HERE/'PROTOCOL.json'),'synthetic_QA':str(out)}))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
    if a.freeze:freeze()
    else:ap.error('One-time --freeze required')
