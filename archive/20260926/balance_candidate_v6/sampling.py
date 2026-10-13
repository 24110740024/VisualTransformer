"""Train-only, class-balanced square-root source sampling. No GPU dependencies."""
from collections import Counter,defaultdict
import math
import numpy as np

POLICY='class_half_source_sqrt_count_v1'


def probabilities(rows):
    if not rows:raise ValueError('Empty training manifest')
    cells=Counter();labels=set();seen=set()
    for r in rows:
        if r.get('split')!='train':raise ValueError('Only this role train is accepted')
        c=str(r.get('label'));s=r.get('source_id');pix=r.get('pixel_sha256')
        if c not in ('0','1') or not s or not pix:raise ValueError('Missing or invalid identity/class')
        if r.get('class')!=('Lesion' if c=='0' else 'Normal'):raise ValueError('Wrong class direction')
        if pix in seen:raise ValueError('Duplicate pixel identity; no hidden multiplicity')
        seen.add(pix);labels.add(c);cells[c,s]+=1
    if labels!={'0','1'}:raise ValueError('Both classes required')
    z={c:math.fsum(math.sqrt(n) for (k,s),n in cells.items() if k==c) for c in labels}
    p=np.array([.5/(math.sqrt(cells[str(r['label']),r['source_id']])*z[str(r['label'])]) for r in rows],dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p<=0) or not math.isclose(math.fsum(p),1.,abs_tol=1e-14):
        raise ValueError('Invalid sampling probabilities')
    for c in labels:
        if not math.isclose(math.fsum(p[i] for i,r in enumerate(rows) if str(r['label'])==c),.5,abs_tol=1e-14):
            raise ValueError('Class mass not half')
    return p


def summary(rows,p):
    cells=defaultdict(list);source=defaultdict(list);n=len(rows)
    for r,v in zip(rows,p,strict=True):
        cells[str(r['label']),r['source_id']].append(float(v));source[r['source_id']].append(float(v))
    return dict(policy=POLICY,rows=n,sources=len(source),loss_class_weights=None,
        max_expected_repeats_per_patch_per_epoch=n*float(np.max(p)),
        maximum_source_sampling_mass=max(map(math.fsum,source.values())),
        minimum_source_sampling_mass=min(map(math.fsum,source.values())),
        cells=[dict(label=c,source_id=s,rows=len(v),sampling_mass=math.fsum(v),
                    expected_draws_per_epoch=n*math.fsum(v),expected_repeats_per_patch_per_epoch=n*v[0])
               for (c,s),v in sorted(cells.items())])
