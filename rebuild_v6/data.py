"""Strict, portable manifest validation; no data-dependent resplitting."""
import csv, hashlib, itertools, json
from pathlib import Path
import numpy as np
from PIL import Image

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def check_published_manifest(path):
    index=json.loads((Path(__file__).resolve().parents[1]/'data/MANIFEST_INDEX.json').read_text())
    expected=index['members_sha256'].get(Path(path).name)
    if expected is None or sha(path)!=expected:raise ValueError('Use unchanged published manifest bytes; asset paths are relative to their directory')

def read_role(path,role,verify_pixels=False):
    rows=read_rows(path);keys=set()
    if not rows:raise ValueError('Empty manifest')
    for r in rows:
        for key in ('path','label','class','source_id','split','pixel_sha256','canonical_animal_id'):
            if not r.get(key):raise ValueError('Missing manifest field: '+key)
        if r['split']!=role or r['label'] not in ('0','1'):raise ValueError('Wrong role or label')
        if r['class']!=('Lesion' if r['label']=='0' else 'Normal'):raise ValueError('Class direction mismatch')
        if r['pixel_sha256'] in keys:raise ValueError('Duplicate pixel identity in role')
        keys.add(r['pixel_sha256'])
        p=Path(r['path']);p=p if p.is_absolute() else Path(path).resolve().parent/p
        r['path']=str(p.resolve(strict=True))
        if verify_pixels:
            with Image.open(p) as im:
                if im.size!=(224,224):raise ValueError('Expected complete 224x224 PNG')
                rgb=im.convert('RGB')
                # Preserve the original dimension/mode prefix in the pixel hash.
                if hashlib.sha256(b'224,224,RGB:'+rgb.tobytes()).hexdigest()!=r['pixel_sha256']:raise ValueError('Decoded pixel hash mismatch')
    if {r['label'] for r in rows}!={'0','1'}:raise ValueError('Both classes required')
    return rows

def validate_roles(paths,verify_pixels=False):
    roles={role:read_role(path,role,verify_pixels) for role,path in paths.items()}
    for a,b in itertools.combinations(roles,2):
        for key in ('canonical_animal_id','source_id','pixel_sha256','path'):
            if {r[key] for r in roles[a]} & {r[key] for r in roles[b]}:raise ValueError(f'{a}/{b} leakage: {key}')
    return roles

def write_csv(path,rows):
    if not rows:raise ValueError('No rows')
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');wr.writeheader();wr.writerows(rows)
