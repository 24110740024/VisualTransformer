"""Freeze only v6 train-derived sampling; does not prepare or start training."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,csv,hashlib,json,math,sys
import numpy as np
from sampling import POLICY,probabilities,summary

HERE=Path(__file__).resolve().parent;NEW=HERE.parent


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Invalid input accepted')


def self_test():
    synth=[]
    for c,s,n in [('0','A',4),('0','B',16),('1','A',9),('1','C',1)]:
        synth.extend(dict(split='train',label=c,source_id=s,pixel_sha256=f'{c}-{s}-{i}',
                          **{'class':'Lesion' if c=='0' else 'Normal'}) for i in range(n))
    p=probabilities(synth)
    # Class0 source masses 1/6 and1/3; class1 masses3/8 and1/8.
    assert np.allclose([p[:4].sum(),p[4:20].sum(),p[20:29].sum(),p[29:].sum()],[1/6,1/3,3/8,1/8],rtol=0,atol=1e-14)
    perm=np.random.default_rng(44).permutation(len(synth))
    assert np.array_equal(probabilities([synth[i] for i in perm]),p[perm])
    for kind in ('val','heldout','missing','direction','duplicate'):
        bad=copy.deepcopy(synth)
        if kind in ('val','heldout'):bad[0]['split']=kind
        elif kind=='missing':bad[0]['source_id']=''
        elif kind=='direction':bad[0]['class']='Normal'
        else:bad[0]['pixel_sha256']=bad[1]['pixel_sha256']
        reject(lambda:probabilities(bad))
    reject(lambda:probabilities([]));reject(lambda:probabilities(synth[:20]))
    import torch
    assert not torch.cuda.is_initialized()
    weights=torch.tensor(p,dtype=torch.double);generator=torch.Generator().manual_seed(42)
    state=generator.get_state();one=list(torch.utils.data.WeightedRandomSampler(weights,10000,True,generator))
    generator.set_state(state);two=list(torch.utils.data.WeightedRandomSampler(weights,10000,True,generator))
    assert one==two
    # Check draw counts against known probabilities with a conservative deterministic tolerance.
    draw=torch.multinomial(weights,200000,replacement=True,generator=torch.Generator().manual_seed(43)).numpy()
    observed=np.bincount(draw,minlength=len(p))/len(draw)
    assert np.all(abs(observed-p)<7*np.sqrt(p*(1-p)/len(draw))+2/len(draw))
    assert not torch.cuda.is_initialized()
    return dict(status='passed',synthetic_closed_form_masses=True,row_permutation_equivariance=True,
                invalid_input_rejections=7,CPU_weighted_sampler_draws=200000,sampler_rng_roundtrip_exact=True,
                cuda_initialized=False,study_predictions_or_outcomes_read=False)


def main():
    assert not (HERE/'SAMPLING_FROZEN.json').exists(),'Preserve frozen sampling'
    qa=self_test();frozen=read(NEW/'original_exports_v2/FROZEN.json');out=HERE/'sampling';out.mkdir(exist_ok=False)
    audit={};manifest={}
    for name in ('a6','a5_fold1','a5_fold2','a5_fold3'):
        p=NEW/'original_exports_v2/manifests'/f'{name}_train.csv'
        assert sha(p)==frozen['manifests'][name+'_train']['sha256']
        rr=rows(p);pp=probabilities(rr);report=summary(rr,pp)
        assert len(rr)==frozen['manifests'][name+'_train']['rows']
        # Independently reconstruct probabilities by per-class cell square-root masses.
        reconstructed={}
        for c in ('0','1'):
            cells=[r for r in report['cells'] if r['label']==c];z=sum(math.sqrt(r['rows']) for r in cells)
            for cell in cells:reconstructed[c,cell['source_id']]=(.5*math.sqrt(cell['rows'])/z)/cell['rows']
        assert np.allclose(pp,[reconstructed[str(r['label']),r['source_id']] for r in rr],rtol=0,atol=1e-18)
        target=out/f'{name}_train_probabilities.csv'
        with target.open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['row_index','path','label','source_id','pixel_sha256','probability','expected_repeats_per_epoch']);w.writeheader()
            for i,(r,v) in enumerate(zip(rr,pp,strict=True)):
                w.writerow(dict(row_index=i,**{k:r[k] for k in ('path','label','source_id','pixel_sha256')},probability=repr(float(v)),expected_repeats_per_epoch=repr(len(rr)*float(v))))
        saved=rows(target);assert np.array_equal(pp,[float(r['probability']) for r in saved])
        report['source_manifest_sha256']=sha(p);audit[name]=report
        manifest[name]=dict(path=str(target),sha256=sha(target),rows=len(rr),source_manifest=str(p),source_manifest_sha256=sha(p))
    save(HERE/'CPU_QA.json',qa);save(HERE/'TRAIN_INPUT_AUDIT.json',audit)
    save(HERE/'SAMPLING_FROZEN.json',dict(protocol='V6_SAMPLING_PREPARATION_ONLY',policy=POLICY,
        created_utc=datetime.now(timezone.utc).isoformat(),files_sha256={str(HERE/n):sha(HERE/n) for n in ('sampling.py','prepare_sampling.py','PLAN.md','CPU_QA.json','TRAIN_INPUT_AUDIT.json')},
        manifests=manifest,production_execution_protocol_frozen=False,GPU_started=False,
        loss_class_weights=None,training_samples_per_epoch='Own training row count, replacement',
        basis='Read-only training count diagnostic, not test/outcome/hotspot selection'))
    print(json.dumps({'status':'sampling_prepared_only','sha256':sha(HERE/'SAMPLING_FROZEN.json'),'summary':{k:{n:v[n] for n in ('rows','sources','max_expected_repeats_per_patch_per_epoch','maximum_source_sampling_mass')} for k,v in audit.items()}},indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:main()
    else:p.error('One-time --freeze required')
