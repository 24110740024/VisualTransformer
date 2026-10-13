"""Read-only independent checks of completed v6 training records, no torch/GPU."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,math

HERE=Path(__file__).resolve().parent
V6=HERE/'balance_candidate_v6'
EXPECTED='8d563e695cc9280b0ec3c5734022d2705f0352dffacd98788d97b41c19a1be9b'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),(a,b)


def main(allow_partial):
    spec=read(V6/'EXECUTION_PROTOCOL.json');assert sha(V6/'EXECUTION_PROTOCOL.json')==EXPECTED
    for path,digest in spec['files_sha256'].items():assert sha(path)==digest,path
    verified=[];pending=[]
    for task,seed,prefix in spec['tasks']:
        path=V6/'jobs'/f'{task}.trained.json'
        if not path.exists():pending.append(task);continue
        marker=read(path);folder=V6/'runs'/task
        assert marker['task']==task and marker['manifest_prefix']==prefix
        for name,digest in marker['files_sha256'].items():assert sha(folder/name)==digest,(task,name)
        status=read(folder/'status.json');assert status==marker['status']
        config=read(folder/'config.json');c=config['configuration']
        digest=hashlib.sha256(json.dumps(c,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        assert digest==config['config_sha256']==status['config_sha256']
        assert c['protocol_sha256']==EXPECTED and c['seed']==seed
        for key,value in spec['parameters'].items():
            assert key in c and c[key]==value,(task,key)
        assert c['epochs']==20 and c['patience']==6 and c['bs']==32 and c['weighted_sampler'] and not c['class_weight']
        assert c['initialization']=='torchvision_ViT_B_16_IMAGENET1K_V1'
        assert c['pretrained_weights']['sha256']=='c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9'
        for role in ('train','val'):
            expected=HERE/'original_exports_v2/manifests'/f'{prefix}_{role}.csv'
            m=c['manifests']['manifests'][role]
            assert Path(c[f'{role}_manifest'])==expected and sha(expected)==m['sha256']
            assert len(rows(expected))==m['rows']
        from audit_v6_sampling import audit_sampling
        sampling_audit=audit_sampling(c,prefix)
        history=rows(folder/'history.csv');assert [int(h['epoch']) for h in history]==list(range(1,status['completed_epoch']+1))
        best=float('-inf');best_epoch=0;no_improve=0
        for h in history:
            e=int(h['epoch']);value=float(h['val_auc_lesion']);assert math.isfinite(value)
            improved=value>best
            assert (h['improved']=='True')==improved
            if improved:best=value;best_epoch=e;no_improve=0
            else:no_improve+=1
            close(h['lr_used'],1e-5*(1+math.cos(math.pi*(e-1)/20))/2)
            close(h['head_lr_used'],1e-4*(1+math.cos(math.pi*(e-1)/20))/2)
            if e<status['completed_epoch']:assert no_improve<6
        assert best_epoch==status['best_epoch'] and no_improve==status['no_improve']
        close(best,status['best_val_auc_lesion'])
        assert status['completed_epoch']==20 or no_improve==6
        verified.append(dict(sampling=sampling_audit,task=task,completed_epoch=status['completed_epoch'],best_epoch=best_epoch,best_validation_auc=best,
                             config_sha256=digest,best_checkpoint_sha256=marker['files_sha256']['best.pth'],marker_sha256=sha(path)))
    assert verified
    if not allow_partial:assert not pending, pending
    report=dict(utc=datetime.now(timezone.utc).isoformat(),status='partial_training_record_audit_pass' if pending else 'all_training_records_pass',
                script_sha256=sha(__file__),execution_protocol_sha256=EXPECTED,verified=verified,pending=pending,
                evaluation_or_test_performance_verified=False,scope='Completed records and actual file digests; no tensor forward or model inference')
    target=HERE/'v6_independent_audits';target.mkdir(exist_ok=True)
    out=target/datetime.now(timezone.utc).strftime('training_%Y%m%dT%H%M%SZ.json')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'verified':len(verified),'pending':len(pending)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');a=p.parse_args();main(a.allow_partial)
