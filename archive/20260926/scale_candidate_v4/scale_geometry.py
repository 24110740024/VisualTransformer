"""Deterministic calibrated windows bound to the original outcome-blind anchors."""
from pathlib import Path
from collections import Counter
import csv,json,hashlib,math
HERE=Path(__file__).resolve().parent
NEW=HERE.parent
GEOMETRY=NEW/'roi_reference_v1/coordinate_audit_20260925T173044_281065Z/coordinate_audit.json'
DIRECTORY=HERE/'dense_scaled'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(test,message):
    if not test:raise ValueError(message)
def sizes_by_source():
    data={}
    for item in read(GEOMETRY)['images']:
        mpp=float(item['mpp']); size=round(224*.5/mpp)
        require(size in (408,409) and size==item['tile_level0_size'],'Unexpected source calibration')
        value={'mpp':mpp,'crop_size':size,'fov_um':size*mpp,'center_delta_level0':(size-512)/2+(512-size)//2}
        if item['source_sha256'] in data:require(data[item['source_sha256']]==value,'Alias scale mismatch')
        data[item['source_sha256']]=value
    return data
def config():
    return {'protocol':'SOURCE_CALIBRATED_SCALE_V4','target_mpp':.5,'output_pixels':224,
            'anchor_policy':'original512 center, integer floor crop origin;409 has -0.5px center offset',
            'source_geometry_sha256':sha(GEOMETRY),'sizes':sizes_by_source(),
            'coordinate_filtering':False,'outcomes_used':False,'predictions_used':False}
def digest_config(c):return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def transform(row,cfg):
    new=dict(row); size=cfg['sizes'][row['source_sha256']]['crop_size']; offset=(512-size)//2
    require(row['crop_size']==512 and not row['pad_right'] and not row['pad_bottom'],'Original geometry changed')
    new.update(x=row['x']+offset,y=row['y']+offset,crop_size=size,
               pad_right=max(0,row['x']+offset+size-row['source_width']),
               pad_bottom=max(0,row['y']+offset+size-row['source_height']),config_sha256=digest_config(cfg))
    require(new['pad_right']==new['pad_bottom']==0,'Calibrated footprint must remain inside original valid window')
    return new
def load_scaled(directory,original,cases,original_identity,fold=None):
    require(Path(directory).resolve()==DIRECTORY.resolve(),'Unplanned scaled coordinate directory')
    binding=read(DIRECTORY/'FROZEN.json');complete=read(DIRECTORY/'COMPLETE.json'); cfg=config()
    require(read(DIRECTORY/'config.json')=={'configuration':cfg,'config_sha256':digest_config(cfg)},'Scaled config changed')
    require(binding['status']=='accepted_for_inference' and binding['outcomes_used'] is False and binding['model_outputs_used'] is False,'Scaled freeze not accepted')
    require(binding['original_identity']==original_identity,'Original anchor binding changed')
    require(binding['manifest_sha256']==sha(DIRECTORY/'coordinates_all21.csv'),'Scaled coordinate SHA mismatch')
    require(binding['complete_sha256']==sha(DIRECTORY/'COMPLETE.json'),'Scaled completion changed')
    require(binding['geometry_code_sha256']==sha(__file__),'Geometry implementation changed')
    require(complete['cases']==list(cases.values()),'Source cases changed')
    ints=('outer_fold','x','y','crop_size','level','source_width','source_height','pad_right','pad_bottom')
    with (DIRECTORY/'coordinates_all21.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    require(len(rows)==len(original)==132032,'Must preserve all132032 anchor identities')
    for old,row in zip(original,rows,strict=True):
        for key in ints:row[key]=int(row[key])
        require(row==transform(old,cfg),'Scaled row differs from independent deterministic derivation')
    require(len({(r['case_id'],r['x'],r['y']) for r in rows})==len(rows),'Scaled duplicates')
    identity={'directory':str(DIRECTORY),'manifest_sha256':sha(DIRECTORY/'coordinates_all21.csv'),
              'complete_sha256':sha(DIRECTORY/'COMPLETE.json'),'frozen_sha256':sha(DIRECTORY/'FROZEN.json'),
              'config_sha256':digest_config(cfg),'total_coordinates':len(rows),
              'case_counts':dict(sorted(Counter(r['case_id'] for r in rows).items())),
              'anchor_origin_manifest_sha256':original_identity['manifest_sha256']}
    return [r for r in rows if fold is None or r['outer_fold']==fold],cases,identity
