"""CPU-only candidate A5 audit and fixed robustness, after core_complete."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, hashlib, json, math, sys
import numpy as np

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
V6 = NEW / 'balance_candidate_v6'
V2 = NEW / 'original_exports_v2'
sys.path.insert(0, str(NEW))
import verify_a5 as verifier
from original_exports_v2 import analyze_robustness as robust

EXPECTED = '8d563e695cc9280b0ec3c5734022d2705f0352dffacd98788d97b41c19a1be9b'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)
def save(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def within(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    require(path != parent and path.is_relative_to(parent), f'Wrong result source: {path}')
    return path


def paths(state):
    require(state.get('phase') == 'core_complete', 'Wait for all candidate outputs; no partial/v2 fallback')
    return {key: within(state[key], V6 / directory) for key, directory in
            [('aggregate', 'aggregate'), ('statistics', 'statistics'), ('a6_evaluation', 'evaluation')]}


def dependencies():
    require(sha(V6/'EXECUTION_PROTOCOL.json') == EXPECTED, 'Candidate execution protocol changed')
    for path, digest in read(V6/'EXECUTION_PROTOCOL.json')['files_sha256'].items():
        require(sha(path) == digest, f'Frozen candidate dependency changed: {path}')
    return {str(p): sha(p) for p in (HERE/'run_audit.py', HERE/'verify_a5.py',
            V2/'analyze_robustness.py', NEW/'render_a5_report.py', NEW/'prepare_outcomes.py',
            NEW/'analyze_a5_associations.py', V2/'QUALITY_ADDENDUM.md', V6/'EXECUTION_PROTOCOL.json')}


def checked_stage(name, expected, parent):
    marker = read(V6/'jobs'/f'{name}.done.json')
    folder = within(marker['output'], parent)
    require(folder == expected, 'Stage marker and final controller disagree')
    for name, digest in marker['outputs_sha256'].items():
        require(Path(name).name == name and sha(folder/name) == digest, 'Stage output changed')


def preflight():
    bound = dependencies()
    state = read(V6/'jobs/pipeline_status.json')
    return dict(status='cpu_dependency_preflight_passed', candidate_phase=state['phase'],
                results_ready=state['phase']=='core_complete', files_sha256=bound,
                gpu_started=False, study_scores_or_outcomes_read=False)


def self_test():
    # The independent verifier is unchanged except for producer SHA identities.
    old = (NEW/'render_a5_report.py').read_text(encoding='utf-8-sig')
    body = old[old.index('def read_json(path):'):old.index('def render_scatter(')]
    body = body.replace('HERE / "analyze_a5_associations.py"', 'HERE / "analyze_candidate.py"')
    body = body.replace('HERE / "aggregate_a5.py"', 'HERE / "aggregate_candidate.py"')
    new = (HERE/'verify_a5.py').read_text(encoding='utf-8')
    require(ast.dump(ast.parse(body)) == ast.dump(ast.parse(new[new.index('def read_json(path):'):])),
            'Independent arithmetic changed during adaptation')
    robust.self_test()
    valid = dict(phase='core_complete', aggregate=str(V6/'aggregate/attempt_test'),
                 statistics=str(V6/'statistics/attempt_test'), a6_evaluation=str(V6/'evaluation/attempt_test'))
    require(len(paths(valid)) == 3, 'Valid candidate path rejected')
    invalid = [dict(valid, phase='training'), dict(valid, aggregate=str(V2/'aggregate/attempt_test')),
               dict(valid, statistics=str(V6/'statistics/../evaluation/attempt_test')),
               dict(valid, aggregate=str(V6/'aggregate'))]
    for case in invalid:
        try: paths(case)
        except ValueError: pass
        else: raise AssertionError('Incomplete, cross-version or misbound path accepted')
    print('PASS: arithmetic equivalence; Holm/Simpson/112-deletion checks; complete-state and four path rejection cases. No study scores read.')


def run():
    protocol = read(HERE/'PROTOCOL.json')
    current = dependencies()
    require(current == protocol['files_sha256'], 'Postprocessing code/dependency changed after freeze')
    state = read(V6/'jobs/pipeline_status.json'); bound = paths(state)
    from run_after_training import alive
    require(not any(alive(state.get(k)) for k in ('pid','child_pid')), 'Wait for producer to exit')
    for name, key, parent in [('a5_aggregate', 'aggregate', V6/'aggregate'),
                              ('a5_statistics', 'statistics', V6/'statistics'),
                              ('a6_evaluation', 'a6_evaluation', V6/'evaluation')]:
        checked_stage(name, bound[key], parent)
    aggregate, stat = bound['aggregate'], bound['statistics']
    complete = read(stat/'COMPLETE.json'); agg = read(aggregate/'COMPLETE.json')
    require(Path(complete['aggregate_directory']).resolve() == aggregate, 'Statistics use another aggregate')
    require(Path(complete['outcomes_directory']).resolve() == (NEW/'outcomes_v1').resolve(), 'Wrong outcomes')
    require(Path(agg['frozen']['directory']).resolve() == (NEW/'scale_candidate_v4/dense_scaled').resolve(), 'Wrong dense inputs')
    require(agg['protocol'] == 'A5_DENSE_EQUAL_SEED_AGGREGATION_BALANCE_V6', 'Not candidate aggregation')
    for r in agg['input_runs']:
        task = f"a5_seed{r['seed']}_fold{r['fold']}"
        folder = within(r['directory'], V6/'predictions'/task)
        checked_stage(task+'.inference', folder, V6/'predictions'/task)
        done = read(folder/'COMPLETE.json')
        require(done['protocol'] == 'A5_DENSE_INFERENCE_BALANCE_V6', 'Wrong prediction version')
        trained = read(V6/'jobs'/f'{task}.trained.json')
        require(trained['task'] == task and trained['manifest_prefix'] == f"a5_fold{r['fold']}", 'Wrong model identity')
        checkpoint = V6/'runs'/task/'best.pth'
        require(sha(checkpoint) == r['checkpoint_sha256'] == trained['files_sha256']['best.pth'], 'Wrong actual checkpoint')
    verifier.STAT_DIR = stat
    marker, joined, tile_means, inputs = verifier.verify_inputs()
    results, recomputed = verifier.verify_statistics(marker, joined)
    main, influence, centered = robust.compute(joined)
    primary = [r for r in results if r['analysis']=='primary21']
    for a, b in zip(main, primary, strict=True):
        require((a['outcome'], a['statistic']) == (b['outcome'], b['statistic']), 'Primary order changed')
        for ka, kb in [('coefficient','coefficient'), ('p_nominal','p_two_sided_nominal')]:
            require(math.isclose(a[ka], b[kb], rel_tol=0, abs_tol=1e-12), 'Robustness differs from main analysis')
    p = np.array([r['p_nominal'] for r in main]); order = np.argsort(p)
    independent_holm = np.empty(4)
    independent_holm[order] = np.minimum(1, np.maximum.accumulate(p[order]*[4,3,2,1]))
    require(np.allclose(independent_holm, [r['p_holm_family4'] for r in main], rtol=0, atol=1e-12), 'Holm mismatch')
    out = HERE/datetime.now(timezone.utc).strftime('verified_%Y%m%dT%H%M%S_%fZ')
    out.mkdir()
    for name, table in [('holm_primary4.csv', main), ('influence_all112.csv', influence),
                        ('within_group_descriptive.csv', centered)]:
        with (out/name).open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    save(out/'report_audit.json', dict(status='numeric_pass', all_predictions_checked=396096,
        seed_case_scores_checked=len(tile_means), animals=len(joined), all_eight_statistics_recomputed=recomputed,
        input_sha256=inputs, main_statistics_directory=str(stat), aggregate_directory=str(aggregate),
        protocol_sha256=sha(HERE/'PROTOCOL.json'), holm_independently_checked=True))
    save(out/'COMPLETE.json', dict(status='complete', protocol='V6_FIXED_A5_AUDIT_AND_ROBUSTNESS_V1',
        created_utc=datetime.now(timezone.utc).isoformat(), main_statistics_directory=str(stat),
        main_statistics_complete_sha256=sha(stat/'COMPLETE.json'), primary_n_unchanged=21,
        primary_scores_unchanged=True, model_or_animal_selection=False, influence_rows=len(influence),
        outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        limitations=['Post hoc; prior study results known.', 'Four main tests adjusted jointly.',
                     'All 112 deletion diagnostics retained; no animal selection.',
                     'Group-centered coefficients descriptive only, 3 animals/group.',
                     'Bootstrap conditional on fitted scores; not retraining uncertainty.']))
    print(json.dumps(dict(output=str(out),status='numeric_pass'),ensure_ascii=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test',action='store_true');group.add_argument('--check',action='store_true')
    group.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    elif args.check:print(json.dumps(preflight(),ensure_ascii=True,indent=2))
    else:run()
