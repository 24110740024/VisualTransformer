"""Full frozen geometry with synthetic probabilities only, before candidate A5 output."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,math,sys
import numpy as np
import pandas as pd
from rectangle_field import *
HERE=Path(__file__).resolve().parent;NEW=HERE.parent;DENSE=NEW/'scale_candidate_v4/dense_scaled'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    frozen=read(DENSE/'FROZEN.json');require(sha(DENSE/'coordinates_all21.csv')==frozen['manifest_sha256'],'Changed geometry')
    require(sha(DENSE/'COMPLETE.json')==frozen['complete_sha256'],'Changed geometry marker')
    coords=pd.read_csv(DENSE/'coordinates_all21.csv');meta=read(DENSE/'COMPLETE.json')['cases'];checks=[]
    rng=np.random.default_rng(20260926)
    for case in meta:
        a=coords[coords.case_id==case['case_id']][['x','y','crop_size']].to_numpy();values=rng.random(len(a))
        width,height=case['source_width'],case['source_height']
        f=field_from_rectangles(a,values,width,height);d=independent_difference(a,values,width,height);check_fields(f,d)
        union=footprint_from_windows(a);footprint=footprint_from_cells(f)
        require(union.symmetric_difference(footprint).area==0,'Actual windows differ from field footprint')
        heat,fraction=display64(f);require(heat.shape==(case['thumbnail_height'],case['thumbnail_width']),'Display geometry differs')
        expected_integral=float(np.nansum(f['probability']*areas(f)));display_integral=float(np.nansum(heat*fraction)*64**2)
        require(math.isclose(expected_integral,display_integral,rel_tol=1e-12,abs_tol=1e-6),'Display changes integrated probability')
        # Independent membership and values at deterministic real window centers/edges.
        points=[]
        for x,y,s in a[np.linspace(0,len(a)-1,7,dtype=int)]:
            points.extend([(x,y),(x+s/2,y+s/2),(x+s,y+s/2),(x+s-.1,y+s-.1)])
        point_array=np.asarray(points);sample,scored,_=sample_centers(point_array[:,0],point_array[:,1],f)
        for (x,y),p,valid in zip(points,sample,scored,strict=True):
            mask=(a[:,0]<=x)&(x<a[:,0]+a[:,2])&(a[:,1]<=y)&(y<a[:,1]+a[:,2])
            require(bool(mask.any())==bool(valid),'Actual coordinate center/edge coverage differs')
            if valid:require(math.isclose(float(values[mask].mean()),p,abs_tol=1e-14,rel_tol=0),'Direct rectangle point mean differs')
        checks.append(dict(case_id=case['case_id'],n_windows=len(a),partition_shape=list(f['probability'].shape),
                           max_independent_difference=float(np.nanmax(np.abs(f['probability']-d['probability']))),
                           covered_area_px2=covered_area(f),raw_union_equal=True,display_integral_preserved=True,points_verified=len(points)))
    require(sum(r['n_windows'] for r in checks)==132032 and len(checks)==21,'Incomplete full geometry test')
    result=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),synthetic_probability_seed=20260926,
       actual_model_probabilities_read=False,biological_outcomes_read=False,gpu_imported='torch' in sys.modules,
       synthetic_tests=self_test(),cases=checks,source_sha256={str(p):sha(p) for p in [Path(__file__),HERE/'rectangle_field.py',DENSE/'FROZEN.json',DENSE/'coordinates_all21.csv']})
    require(not result['gpu_imported'],'CPU-only requirement')
    with (HERE/'GEOMETRY_QA.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status='passed',cases=len(checks),windows=132032,max_error=max(r['max_independent_difference'] for r in checks))))
if __name__=='__main__':main()
