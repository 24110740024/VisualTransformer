"""Read only independently accepted, completed v6 results for reporting."""
from pathlib import Path
import math,sys
HERE=Path(__file__).resolve().parent;NEW=HERE.parent
sys.path.insert(0,str(NEW/'v6_visualization_v1'))
from prepare_visuals import validate_numeric_audit,validate_visual_protocol,read,sha,require
need=require
V6=NEW/'balance_candidate_v6'
A6AUDIT=NEW/'v6_independent_audits/a6_20260926T090443Z.json'


def close(a,b):need(math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12),'Numeric mismatch')


def protocol_check():
    p=read(HERE/'PROTOCOL.json');need(p['protocol']=='V6_STANDARD_REPORTS_V2','Wrong report protocol')
    for path,digest in p['files_sha256'].items():need(sha(path)==digest,'Changed report dependency: '+path)


def checked(directory):
    protocol_check();validate_visual_protocol()
    bound=validate_numeric_audit(directory)
    state=read(V6/'jobs/pipeline_status.json')
    sys.path.insert(0,str(NEW))
    from run_after_training import alive
    need(not any(alive(state.get(k)) for k in ('pid','child_pid')),'Wait for producer exit')
    ev=bound['a6_evaluation'];audit=read(A6AUDIT)
    need(audit['all_checks_passed'] and audit['execution_protocol_sha256']==sha(V6/'EXECUTION_PROTOCOL.json'),'A6 audit not accepted')
    need(Path(audit['evaluation_path']).resolve()==ev,'Wrong audited A6 evaluation')
    marker=read(V6/'jobs/a6_evaluation.done.json')
    need(Path(marker['output']).resolve()==ev,'Wrong A6 marker')
    for name,digest in marker['outputs_sha256'].items():
        need(Path(name).name==name and sha(ev/name)==digest,'A6 output changed')
    for name,digest in audit['prediction_files_sha256'].items():need(sha(ev/f'{name}_predictions.csv')==digest,'Audited predictions changed')
    need(sha(V6/'runs/a6_seed42/best.pth')==audit['checkpoint_sha256'],'A6 checkpoint changed')
    summary=read(ev/'evaluation_summary.json')
    for k,v in audit['recomputed_test_metrics'].items():close(v,summary['test'][k])
    close(read(ev/'validation_threshold.json')['threshold'],audit['validation_threshold'])
    return dict(bound,evaluation=ev,a6=audit)
