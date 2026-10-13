"""Freeze/validate CPU display rules and bind completed candidate inputs only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, importlib.util, json, sys
import numpy as np
HERE=Path(__file__).resolve().parent;NEW=HERE.parent;V6=NEW/'balance_candidate_v6'
sys.path.insert(0,str(NEW/'v6_postprocessing_v1'))
from run_audit import read, sha, require, save, within, paths


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def functions(path):
    return {n.name:ast.dump(n) for n in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef)}


def dependencies():
    files=[HERE/n for n in ['prepare_visuals.py','render_heatmaps.py','analyze_spatial.py','render_spatial_report.py','rectangle_field.py','test_geometry.py','GEOMETRY_QA.json','qa.py','README.md']]
    files += [NEW/'render_a5_heatmaps.py',NEW/'original_exports_v2/render_v2_heatmaps.py',
        NEW/'spatial_diagnostic_v1/analyze_spatial_canonical.py',NEW/'spatial_diagnostic_v1/render_report.py',
        NEW/'spatial_diagnostic_v1/PROTOCOL_CANONICAL_FIELD.json',NEW/'roi_reference_v1/FROZEN_REFERENCE.json',
        NEW/'v6_postprocessing_v1/PROTOCOL.json',NEW/'v6_postprocessing_v1/run_audit.py',
        V6/'EXECUTION_PROTOCOL.json']
    require(sha(V6/'EXECUTION_PROTOCOL.json')=='8d563e695cc9280b0ec3c5734022d2705f0352dffacd98788d97b41c19a1be9b','Candidate changed')
    require(sha(NEW/'v6_postprocessing_v1/PROTOCOL.json')=='05f195a6cb53c4d5bf185a92f3c2c72df55e80d96bdb23fcc06671adb4c1d601','A5 audit protocol changed')
    return {str(p):sha(p) for p in files}


def self_test():
    expected=functions(NEW/'spatial_diagnostic_v1/analyze_spatial_canonical.py')
    adapted=functions(HERE/'analyze_spatial.py')
    require(all(expected[n]==adapted[n] for n in ['pair_auc','centers_for_image']),'Fixed reference categories/AUC changed')
    require(functions(NEW/'original_exports_v2/render_v2_heatmaps.py')['reference_panel']==functions(HERE/'render_heatmaps.py')['reference_panel'],'Reference drawing changed')
    rect=load(HERE/'rectangle_field.py','v6_exact_geometry_tests');tests=rect.self_test()
    require(tests['status']=='pass','Exact rectangle field QA failed')
    actual=read(HERE/'GEOMETRY_QA.json')
    require(actual['status']=='passed' and len(actual['cases'])==21 and sum(r['n_windows'] for r in actual['cases'])==132032,'Real geometry QA incomplete')
    for path,digest in actual['source_sha256'].items():require(sha(path)==digest,'Geometry changed after full QA')
    require('torch' not in sys.modules,'GPU dependency imported')
    return dict(status='passed',tests=tests,reference_drawing_identical=True,gpu_imported=False,
        study_predictions_or_outcomes_read=False,files_sha256=dependencies())


def validate_visual_protocol():
    protocol=read(HERE/'VISUAL_PROTOCOL.json')
    require(dependencies()==protocol['files_sha256'],'Visualization code/dependency changed after freeze')


def validate_numeric_audit(directory):
    current=paths(read(V6/'jobs/pipeline_status.json'))
    folder=within(directory,NEW/'v6_postprocessing_v1')
    complete=read(folder/'COMPLETE.json');audit=read(folder/'report_audit.json')
    require(complete['status']=='complete' and audit['status']=='numeric_pass','A5 audit incomplete')
    require(complete['protocol']=='V6_FIXED_A5_AUDIT_AND_ROBUSTNESS_V1','Wrong A5 audit')
    require(audit['protocol_sha256']==sha(NEW/'v6_postprocessing_v1/PROTOCOL.json'),'Wrong audit method')
    require(Path(complete['main_statistics_directory']).resolve()==current['statistics'],'Wrong audited statistics')
    require(complete['main_statistics_complete_sha256']==sha(current['statistics']/'COMPLETE.json'),'Statistics changed')
    require(Path(audit['aggregate_directory']).resolve()==current['aggregate'],'Wrong audited aggregate')
    require(audit['all_predictions_checked']==396096 and audit['animals']==21,'Incomplete prediction audit')
    for name,digest in complete['outputs_sha256'].items():
        require(Path(name).name==name and sha(folder/name)==digest,'Audit output changed')
    # Bind every numerical input, including completed predictions, not just its summary.
    for path,digest in audit['input_sha256'].items():require(sha(path)==digest,'Audited input changed')
    return current


def bind_spatial(heatmap_dir):
    validate_visual_protocol()
    folder=within(heatmap_dir,V6/'reports')
    heat=read(folder/'report_audit.json')
    state=validate_numeric_audit(heat['numeric_audit_directory'])
    require(heat['cases']==21 and heat['frozen_tiles']==132032 and heat['predictions']==396096,'Heatmap set incomplete')
    require(heat['visual_protocol_sha256']==sha(HERE/'VISUAL_PROTOCOL.json'),'Wrong heatmap rules')
    require(heat['source_aggregate_complete_sha256']==sha(state['aggregate']/'COMPLETE.json'),'Heatmap aggregate mismatch')
    require(heat['numeric_audit_sha256']==sha(Path(heat['numeric_audit_directory'])/'report_audit.json'),'Heatmap audit changed')
    for name,digest in heat['outputs_sha256'].items():require(sha(within(folder/name,folder))==digest,'Heatmap output changed')
    reference=NEW/'roi_reference_v1/FROZEN_REFERENCE.json'
    require(heat['reference_freeze_sha256']==sha(reference),'Reference mismatch')
    previous=read(NEW/'spatial_diagnostic_v1/PROTOCOL_CANONICAL_FIELD.json')
    bound={str(p):sha(p) for p in [HERE/'VISUAL_PROTOCOL.json',HERE/'analyze_spatial.py',HERE/'render_spatial_report.py',HERE/'rectangle_field.py',
        NEW/'scale_candidate_v4/dense_scaled/FROZEN.json',NEW/'scale_candidate_v4/dense_scaled/COMPLETE.json',NEW/'scale_candidate_v4/dense_scaled/coordinates_all21.csv',reference,
        NEW/previous['center_audit_relative'],folder/'report_audit.json']}
    protocol=dict(status='fixed_v6_input_binding_exact_rectangle_field',utc=datetime.now(timezone.utc).isoformat(),
        parent_protocol_sha256=sha(NEW/'spatial_diagnostic_v1/PROTOCOL_CANONICAL_FIELD.json'),
        visualization_protocol_sha256=sha(HERE/'VISUAL_PROTOCOL.json'),
        versions={'v6':{'root':'balance_candidate_v6','heatmap_directory':str(folder.relative_to(V6))}},
        center_audit_relative=previous['center_audit_relative'],grid=64,measurement_field='exact408/409 rectangle-edge partition; half-open boundaries',display='covered-area weighted64; not used for quantitative measurements',threshold=None,reads_outcomes=False,
        pools_reference_versions=False,files_sha256=bound,float_csv_reading='round_trip',
        authoritative_probability='SHA-verified saved exact float64 NPZ and edges; independent prefix-sum and raw-window-union checks',
        results_selection=False,prior_v2_results_known=True)
    save(HERE/'SPATIAL_INPUT_BINDING.json',protocol)
    print(json.dumps({'binding_sha256':sha(HERE/'SPATIAL_INPUT_BINDING.json')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-test',action='store_true');g.add_argument('--bind-heatmap-dir');a=ap.parse_args()
    if a.self_test:print(json.dumps(self_test(),ensure_ascii=True,indent=2))
    else:bind_spatial(a.bind_heatmap_dir)
