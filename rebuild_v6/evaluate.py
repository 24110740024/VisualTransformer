"""A6 inference for portable-v6 checkpoints; validation-only threshold."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from candidate_model import create_model,validate_checkpoint_identity
from data import validate_roles,sha,write_csv,check_published_manifest
from scale_transforms import build_transforms
from metrics import select_validation_threshold,metrics,bootstrap_auc

class Images(Dataset):
    def __init__(self,rows):self.rows=rows;self.transform=build_transforms(False)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['path']) as im:return self.transform(im.convert('RGB'))

def predict(model,rows):
    out=[]
    with torch.no_grad():
        for batch in DataLoader(Images(rows),batch_size=64,shuffle=False,num_workers=0):
            with torch.autocast('cuda',dtype=torch.float16):logits=model(batch.to('cuda'))
            out.extend(logits.softmax(1)[:,0].cpu().tolist())
    return np.asarray(out,dtype=np.float64)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train','val','test','checkpoint','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();paths={r:getattr(a,r) for r in ('train','val','test')}
    if any(path.name!='a6_'+role+'.csv' for role,path in paths.items()):raise ValueError('Use the three published A6 roles')
    for path in paths.values():check_published_manifest(path)
    roles=validate_roles(paths,True)
    # Only load locally trusted checkpoints: they include Python RNG state.
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=validate_checkpoint_identity(ck)
    if cfg['seed']!=42:raise ValueError('A6 uses seed42')
    for role in ('train','val'):
        if cfg['manifests']['manifests'][role]['sha256']!=sha(paths[role]):raise ValueError('Training role changed')
    if not torch.cuda.is_available():raise ValueError('Recorded inference requires CUDA FP16')
    a.out.mkdir(parents=True,exist_ok=False)
    model=create_model(False);model.load_state_dict(ck['model'],strict=True);model.to('cuda').eval()
    vp=predict(model,roles['val']);vy=np.array([int(r['label']=='0') for r in roles['val']]);lock=select_validation_threshold(vy,vp)
    (a.out/'validation_threshold.json').write_text(json.dumps(lock,indent=2))
    tp=predict(model,roles['test']);ty=np.array([int(r['label']=='0') for r in roles['test']])
    for role,probs in (('val',vp),('test',tp)):
        write_csv(a.out/(role+'_predictions.csv'),[dict(manifest_row_index=i,source_id=r['source_id'],pixel_sha256=r['pixel_sha256'],label_index=int(r['label']),y_lesion=int(r['label']=='0'),prob_lesion=float(v)) for i,(r,v) in enumerate(zip(roles[role],probs,strict=True))])
    ids=np.array([r['source_id'] for r in roles['test']]);names={r['source_id']:r.get('legacy_wsi',r['source_id']) for r in roles['test']}
    report=dict(test=metrics(ty,tp,lock['threshold']),bootstrap=bootstrap_auc(ty,tp,ids,sorted(names,key=names.get)),checkpoint_sha256=sha(a.checkpoint),manifest_sha256={k:sha(v) for k,v in paths.items()})
    (a.out/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
