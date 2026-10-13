"""A5 four-flip FP16 inference on supplied frozen calibrated coordinates."""
import argparse,json,csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from candidate_model import create_model,validate_checkpoint_identity
from data import read_rows,validate_roles,sha,check_published_manifest
from dense_core import DenseCropDataset,worker_init,flips4_probabilities

def check_coordinates(rows,fold,roles):
    seen=set();cases={};held_sources=set()
    for r in rows:
        for k in ('outer_fold','x','y','crop_size','level','source_width','source_height'):r[k]=int(r[k])
        if r['outer_fold']!=fold:raise ValueError('Wrong outer fold')
        if r['crop_size'] not in (408,409) or r['level']!=0:raise ValueError('Wrong calibrated footprint')
        if not (0<=r['x']<=r['source_width']-r['crop_size'] and 0<=r['y']<=r['source_height']-r['crop_size']):raise ValueError('Window outside source')
        key=(r['case_id'],r['source_sha256'],r['x'],r['y'],r['crop_size'],r['level'])
        if key in seen:raise ValueError('Duplicate dense key')
        seen.add(key);held_sources.add(r['source_sha256'])
        if r['case_id'] in cases and cases[r['case_id']]!=r['source_sha256']:raise ValueError('Multiple sources per animal')
        cases[r['case_id']]=r['source_sha256']
    if len(cases)!=7 or len(held_sources)!=7:raise ValueError('Each original outer fold contains seven animals/sources')
    for role,records in roles.items():
        if held_sources & {r['source_id'] for r in records}:raise ValueError('Held-out source in '+role)
        if set(cases) & {r['canonical_animal_id'] for r in records}:raise ValueError('Held-out animal or source alias in '+role)
        if any(int(r.get('outer_fold',-1))!=fold for r in records):raise ValueError('Train/val role belongs to different fold')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('coordinates','train','val','checkpoint','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--fold',type=int,choices=(1,2,3),required=True);p.add_argument('--seed',type=int,choices=(42,43,44),required=True)
    a=p.parse_args()
    for path in (a.train,a.val,a.coordinates):check_published_manifest(path)
    roles=validate_roles({'train':a.train,'val':a.val},True)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=validate_checkpoint_identity(ck)
    if cfg['seed']!=a.seed:raise ValueError('Seed differs from checkpoint')
    for role in ('train','val'):
        if cfg['manifests']['manifests'][role]['sha256']!=sha(getattr(a,role)):raise ValueError('Checkpoint role mismatch')
    all_rows=read_rows(a.coordinates);rows=[r for r in all_rows if int(r['outer_fold'])==a.fold]
    rows=check_coordinates(rows,a.fold,roles)
    sources={}
    for r in rows:
        path=Path(r['source_path']);r['source_path']=str((path if path.is_absolute() else a.coordinates.resolve().parent/path).resolve(strict=True))
        if r['source_path'] in sources and sources[r['source_path']]!=r['source_sha256']:raise ValueError('Source identity conflict')
        sources[r['source_path']]=r['source_sha256']
    for path,digest in sources.items():
        if sha(path)!=digest:raise ValueError('WSI content hash mismatch')
    if not torch.cuda.is_available():raise ValueError('CUDA FP16 required')
    a.out.mkdir(parents=True,exist_ok=False);model=create_model(False);model.load_state_dict(ck['model'],strict=True);model.to('cuda').eval()
    ck_sha=sha(a.checkpoint);dataset=DenseCropDataset(rows)
    fields=['case_id','source_sha256','outer_fold','x','y','crop_size','level','seed','checkpoint_sha256','prob_lesion']
    output=a.out/'predictions.csv'
    with output.open('w',encoding='utf-8',newline='') as f,torch.no_grad():
        wr=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');wr.writeheader()
        for images,indices in DataLoader(dataset,batch_size=64,num_workers=0,shuffle=False,worker_init_fn=worker_init):
            probs=flips4_probabilities(model,images.to('cuda'),True).cpu().tolist()
            for i,prob in zip(indices.tolist(),probs,strict=True):
                wr.writerow({**{k:rows[i][k] for k in fields[:7]},'seed':a.seed,'checkpoint_sha256':ck_sha,'prob_lesion':prob})
    dataset.close()
    marker=dict(status='complete',seed=a.seed,fold=a.fold,rows=len(rows),coordinate_sha256=sha(a.coordinates),checkpoint_sha256=ck_sha,predictions_sha256=sha(output),tta='flips4_mean_softmax',temperature=1.0,post_manifest_filter=False)
    (a.out/'COMPLETE.json').write_text(json.dumps(marker,indent=2));print(json.dumps(marker))

if __name__=='__main__':main()
