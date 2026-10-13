"""Independent saved-prediction checks for the candidate A6, CPU only."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
from original_exports_v2.audit_completed_a6 import read, rows, sha, close, auc, metrics, independent_bootstraps

HERE=Path(__file__).resolve().parent
V6=HERE/'balance_candidate_v6'
V2=HERE/'original_exports_v2'
EXPECTED='8d563e695cc9280b0ec3c5734022d2705f0352dffacd98788d97b41c19a1be9b'


def main():
    assert sha(V6/'EXECUTION_PROTOCOL.json')==EXPECTED
    spec=read(V6/'EXECUTION_PROTOCOL.json')
    for path,digest in spec['files_sha256'].items():assert sha(path)==digest,path
    marker=read(V6/'jobs/a6_evaluation.done.json');evaluation=Path(marker['output'])
    assert evaluation.is_relative_to(V6/'evaluation')
    for name,digest in marker['outputs_sha256'].items():assert sha(evaluation/name)==digest,name
    training=read(V6/'jobs/a6_seed42.trained.json');folder=V6/'runs/a6_seed42'
    for name,digest in training['files_sha256'].items():assert sha(folder/name)==digest,name
    checkpoint_sha=sha(folder/'best.pth')
    summary=read(evaluation/'evaluation_summary.json');protocol=summary['protocol']
    assert protocol['checkpoint_sha256']==checkpoint_sha==training['files_sha256']['best.pth']
    assert protocol['checkpoint_config_sha256']==training['status']['config_sha256']
    assert protocol['epoch']==training['status']['best_epoch']
    assert protocol['candidate']=='TRAIN_BALANCE_CANDIDATE_V6' and protocol['initialization']=='torchvision_ViT_B_16_IMAGENET1K_V1'
    assert protocol['evaluator_sha256']==sha(V6/'evaluate_candidate.py')
    frozen=read(V2/'FROZEN.json')
    for role in ('train','val','test'):
        path=V2/'manifests'/f'a6_{role}.csv'
        assert sha(path)==protocol['manifests'][role]['sha256']==frozen['manifests'][f'a6_{role}']['sha256']
    threshold=read(evaluation/'validation_threshold.json')
    assert threshold['checkpoint_sha256']==checkpoint_sha and threshold['status']=='locked'
    predictions={}
    for split,count in [('val',7909),('test',8704)]:
        manifest=rows(V2/'manifests'/f'a6_{split}.csv');prediction=rows(evaluation/f'{split}_predictions.csv')
        assert len(manifest)==len(prediction)==count
        for i,(m,r) in enumerate(zip(manifest,prediction)):
            assert all(m[k]==r[k] for k in m) and int(r['manifest_row_index'])==i
            assert int(r['y_lesion'])==int(r['label']=='0')==int(r['class']=='Lesion')
        y=np.array([int(r['y_lesion']) for r in prediction]);p=np.array([float(r['prob_lesion']) for r in prediction])
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        predictions[split]=(prediction,y,p)
    _,vy,vp=predictions['val']
    candidates=np.r_[np.inf,np.unique(vp)[::-1]]
    positive=np.sort(vp[vy==1]);negative=np.sort(vp[vy==0])
    tpr=1-np.searchsorted(positive,candidates,'left')/len(positive)
    fpr=1-np.searchsorted(negative,candidates,'left')/len(negative)
    selected=float(candidates[int(np.argmax(tpr-fpr))])
    close(selected,threshold['threshold']);close(max(tpr-fpr),threshold['youden_j'])
    close(auc(vy,vp),threshold['validation_auc_lesion'])
    test,y,p=predictions['test'];calculated=metrics(y,p,selected)
    for k,v in calculated.items():close(v,summary['test'][k])
    assert all(int(r['pred_lesion_at_validation_threshold'])==int(float(r['prob_lesion'])>=selected) for r in test)
    sources=np.array([r['source_id'] for r in test]);order=list(dict.fromkeys(sources));assert len(order)==4
    per_source=rows(evaluation/'test_per_source_metrics.csv');assert [r['source_id'] for r in per_source]==order
    source_aucs=[]
    for row in per_source:
        mask=sources==row['source_id'];m=metrics(y[mask],p[mask],selected)
        for k,v in m.items():close(v,row[k])
        source_aucs.append(m['auc_lesion'])
    close(np.mean(source_aucs),summary['per_source_patch_auc_macro_mean'])
    boots=independent_bootstraps(y,p,sources,order)
    for key,value in boots.items():
        assert value['valid_draws']==summary['bootstrap'][key]['valid_draws']
        for a,b in zip(value['percentile_95_ci'],summary['bootstrap'][key]['percentile_95_ci']):close(a,b)
    # All baseline identities are matched, without selecting by their metrics.
    oldmarker=read(V2/'jobs/a6_evaluation.done.json');oldpath=Path(oldmarker['output'])/'test_predictions.csv'
    oldrows=rows(oldpath);assert len(oldrows)==len(test)
    fields=list(rows(V2/'manifests/a6_test.csv')[0])
    assert all(all(a[k]==b[k] for k in fields) for a,b in zip(oldrows,test))
    out=HERE/'v6_independent_audits'/datetime.now(timezone.utc).strftime('a6_%Y%m%dT%H%M%SZ.json')
    result=dict(utc=datetime.now(timezone.utc).isoformat(),all_checks_passed=True,script_sha256=sha(__file__),
        independent_numeric_library_sha256=sha(V2/'audit_completed_a6.py'),execution_protocol_sha256=EXPECTED,
        evaluation_path=str(evaluation),checkpoint_sha256=checkpoint_sha,selected_epoch=training['status']['best_epoch'],
        recomputed_test_metrics=calculated,recomputed_bootstrap=boots,validation_threshold=selected,
        validation_auc=auc(vy,vp),prediction_files_sha256={s:sha(evaluation/f'{s}_predictions.csv') for s in ('val','test')},
        v2_test_identity_matches=8704,v2_predictions_sha256=sha(oldpath),
        scope='Saved A6 predictions, identities, threshold, confusion metrics and bootstrap independently verified; no inference/selection',
        limitations=['Four independent WSI; not 8704 independent animals.','Candidate A5 models and statistics not complete.','Does not establish model superiority or pretrained causal benefit.'])
    out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'audit':str(out),'passed':True,'test':calculated,'source_ci':boots['source_wsi_cluster']}))


if __name__=='__main__':main()
