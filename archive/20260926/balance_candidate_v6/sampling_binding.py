"""Fail-closed binding between each frozen train role and sampling probabilities."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from sampling import POLICY, probabilities

HERE=Path(__file__).resolve().parent
FROZEN_SHA='9f5e75921b6c9bacb4959e03bd9e02de73a7f67c6612b23c1dcb04c82054876d'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def require(ok,message):
    if not ok:raise ValueError(message)

def binding_identity(manifest):
    path=HERE/'SAMPLING_FROZEN.json'
    require(sha(path)==FROZEN_SHA,'Sampling freeze changed')
    frozen=json.loads(path.read_text(encoding='utf-8'))
    require(frozen['policy']==POLICY,'Sampling policy mismatch')
    for name,digest in frozen['files_sha256'].items():
        require(sha(name)==digest,'Sampling preparation dependency changed')
    matches=[(k,v) for k,v in frozen['manifests'].items() if Path(v['source_manifest']).resolve()==Path(manifest).resolve()]
    require(len(matches)==1,'Only a frozen own-train manifest is accepted')
    prefix,record=matches[0]
    require(sha(manifest)==record['source_manifest_sha256'],'Training manifest changed')
    require(sha(record['path'])==record['sha256'],'Sampling CSV changed')
    return dict(policy=POLICY,prefix=prefix,sampling_freeze_sha256=FROZEN_SHA,
                probability_csv=record['path'],probability_csv_sha256=record['sha256'],
                train_manifest_sha256=record['source_manifest_sha256'],rows=record['rows'],
                samples_per_epoch=record['rows'],replacement=True,loss_class_weights=None)

def validate_probability_rows(train_rows,saved):
    expected=probabilities(train_rows)
    require(len(saved)==len(train_rows),'Sampling row count mismatch')
    values=[]
    for index,(row,record) in enumerate(zip(train_rows,saved,strict=True)):
        require(int(record['row_index'])==index,'Sampling row order changed')
        require(Path(record['path']).resolve()==Path(row['path']).resolve(),'Sampling image path mismatch')
        for key in ('label','source_id','pixel_sha256'):
            require(str(record[key])==str(row[key]),f'Sampling {key} mismatch')
        value=float(record['probability']);values.append(value)
        require(float(record['expected_repeats_per_epoch'])==len(train_rows)*value,'Sampling repetition metadata mismatch')
    values=np.asarray(values,dtype=np.float64)
    require(np.array_equal(values,expected),'Sampling probabilities differ from frozen mathematical rule')
    return values

def load_training_probabilities(manifest,train_rows):
    identity=binding_identity(manifest)
    require(len(train_rows)==identity['rows'],'Train role count mismatch')
    with Path(identity['probability_csv']).open(encoding='utf-8-sig',newline='') as f:
        saved=list(csv.DictReader(f))
    return validate_probability_rows(train_rows,saved),identity
