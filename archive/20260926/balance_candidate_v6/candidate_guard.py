"""Fail closed before any v6 GPU task; absence of a frozen protocol blocks it."""
from pathlib import Path
import sys
from candidate_model import HERE, NEW, V2, sha, read, require
sys.path.insert(0,str(NEW))
from run_after_training import alive


def validate_execution_protocol():
    path=HERE/'EXECUTION_PROTOCOL.json'
    require(path.is_file(), 'Candidate implementation is not frozen; GPU execution is not authorized by a ready protocol')
    spec=read(path)
    require(spec.get('protocol') == 'TRAIN_BALANCE_CANDIDATE_V6', 'Wrong candidate protocol')
    for name,digest in spec['files_sha256'].items():
        require(sha(name) == digest, f'Candidate frozen dependency changed: {name}')
    return spec


def upstream_guard():
    validate_execution_protocol()
    for version in ('scale_candidate_v4','phikon_candidate_v5'):
        prior=read(NEW/version/'jobs/pipeline_status.json')
        require(prior.get('phase')=='core_complete',f'{version} incomplete')
        require(not any(alive(prior.get(k)) for k in ('pid','child_pid')),f'{version} still alive')
    previous=read(NEW/'pretrained_candidate_v3/jobs/pipeline_status.json')
    require(previous.get('phase')=='core_complete', 'Prior v3 incomplete')
    require(not any(alive(previous.get(k)) for k in ('pid','child_pid')), 'Prior v3 still alive')
    core=read(V2/'jobs/pipeline_status.json')
    baseline=read(V2/'jobs/resnet50_original_exports_v2_seed42.status.json')
    for status,phase in ((core,'core_complete'),(baseline,'complete')):
        require(status.get('phase') == phase, 'Previous v2 GPU workflow is incomplete')
        require(not any(alive(status.get(k)) for k in ('pid','child_pid')), 'Previous controller or child still alive')
    directory=Path(baseline['evaluation']).resolve(strict=True)
    require(directory.is_relative_to(V2/'runs/resnet50_original_exports_v2_seed42'), 'Not the v2 baseline result')
    require(sha(directory/'COMPLETE.json') == baseline['complete_sha256'], 'Baseline completion marker changed')
    complete=read(directory/'COMPLETE.json')
    require(complete['protocol_sha256'] == sha(V2/'RESNET_V2_PROTOCOL.json'), 'Baseline protocol changed')
    for name,digest in complete['outputs_sha256'].items():
        require(sha(directory/name) == digest, 'Baseline completed output changed')


def gpu_guard():
    upstream_guard()
    import audit_a5_precision as guard
    previous=guard.KNOWN_GUI_PROCESSES
    try:
        guard.KNOWN_GUI_PROCESSES=previous|{'chrome'}
        result=guard.require_idle_gpu()
    finally:
        guard.KNOWN_GUI_PROCESSES=previous
    upstream_guard()
    return result
