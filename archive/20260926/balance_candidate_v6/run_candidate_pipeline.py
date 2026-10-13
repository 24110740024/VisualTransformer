"""Single serial v6 queue. CPU --check and explicit --freeze precede --run."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import msvcrt
import os
import subprocess
import sys
import time
from candidate_model import HERE,NEW,ROOT,V2,PARAMETERS,WEIGHT_PATH,sha,read,require,verify_weights
from candidate_guard import validate_execution_protocol,upstream_guard,alive
sys.path.insert(0,str(NEW))

JOBS=HERE/'jobs'
STATUS=JOBS/'pipeline_status.json'
TASKS=[('a6_seed42',42,'a6')]+[(f'a5_seed{s}_fold{f}',s,f'a5_fold{f}') for s in (42,43,44) for f in (1,2,3)]


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def state(phase,**values):
    write(STATUS,dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),phase=phase,**values))


def dependencies():
    paths=[p for p in HERE.glob('*.py')]+[HERE/'PLAN.md',HERE/'RUN_PROTOCOL.md',NEW/'pretrained_candidate_v3/weights/WEIGHTS.json',WEIGHT_PATH]
    paths += list((NEW/'scale_candidate_v4/dense_scaled').glob('*.json'))+[NEW/'scale_candidate_v4/dense_scaled/coordinates_all21.csv']
    paths += [NEW/'scale_candidate_v4/scale_geometry.py',NEW/'scale_candidate_v4/EXECUTION_PROTOCOL.json',NEW/'phikon_candidate_v5/EXECUTION_PROTOCOL.json',HERE/'SAMPLING_FROZEN.json']
    paths += [Path(p) for p in read(HERE/'SAMPLING_FROZEN.json')['files_sha256']]
    paths += [Path(v['path']) for v in read(HERE/'SAMPLING_FROZEN.json')['manifests'].values()]
    paths += [NEW/'roi_reference_v1/coordinate_audit_20260925T173044_281065Z/coordinate_audit.json']
    paths += [V2/'FROZEN.json',V2/'USER_SCOPE.md',V2/'jobs/execution_protocol.json',V2/'RESNET_V2_PROTOCOL.json']
    paths += [Path(item['path']) for item in read(V2/'FROZEN.json')['manifests'].values()]
    paths += [Path(p) for p in read(V2/'jobs/execution_protocol.json')['dependency_sha256']]
    paths += [NEW/'train_manifest.py',NEW/'audit_a5_precision.py',NEW/'run_after_training.py',NEW/'prepare_outcomes.py']
    return {str(p.resolve()):sha(p) for p in sorted(set(paths))}


def cpu_check():
    # No model construction, CUDA initialization, or outcomes parsing.
    from train_candidate import preflight
    from sampling_binding import load_training_probabilities
    from evaluate_candidate import audit_manifests
    from predict_candidate import load_frozen_coordinates
    from candidate_model import enforce_settings
    from train_candidate import parse_args
    inputs={}
    for prefix in ('a6','a5_fold1','a5_fold2','a5_fold3'):
        train,val,audit=preflight(V2/'manifests'/f'{prefix}_train.csv',V2/'manifests'/f'{prefix}_val.csv')
        for rows in (train,val):
            require(len({r['pixel_sha256'] for r in rows})==len(rows),'Duplicate input pixels')
        _,identity=load_training_probabilities(V2/'manifests'/f'{prefix}_train.csv',train)
        inputs[prefix]={'sampling':identity,'train':len(train),'val':len(val),'source_overlap':audit['source_overlap']}
    for name,seed,prefix in TASKS:
        enforce_settings(parse_args(training_command(name,seed,prefix)[3:]))
    _,audit=audit_manifests({role:V2/'manifests'/f'a6_{role}.csv' for role in ('train','val','test')})
    coords,cases,frozen=load_frozen_coordinates(NEW/'scale_candidate_v4/dense_scaled')
    require(len(coords)==132032 and len(cases)==21,'Dense input counts changed')
    return dict(status='cpu_input_preflight_passed',inputs=inputs,a6_test=audit['test'],
                dense_coordinates=len(coords),dense_cases=len(cases),weights=verify_weights(),gpu_started=False)


def freeze():
    path=HERE/'EXECUTION_PROTOCOL.json'
    require(not path.exists(),'Preserve existing execution protocol')
    checks=read(JOBS/'FINAL_CPU_QA.json')
    require(checks['passed'] is True,'Final CPU QA required before freeze')
    for name,digest in checks['files_sha256'].items():
        require(sha(name)==digest,'Code changed after CPU QA')
    audit=cpu_check()
    spec=dict(protocol='TRAIN_BALANCE_CANDIDATE_V6',created_utc=datetime.now(timezone.utc).isoformat(),
              tasks=TASKS,parameters=PARAMETERS,initialization='torchvision ViT-B/16 ImageNet1K_V1',
              prior_study_results_known=True,report_all_results=True,selection='strict validation lesion AUC only',
              outcome_or_test_tuning=False,dense='132032 original anchors; native408/409 calibrated0.5mpp windows; FP16 flips4 temperature1; all-tile mean then equal seed mean',
              execution_order='after completed v2/v3/v4/v5/ResNet; ten models then nine calibrated-window inferences',
              files_sha256=dependencies(),cpu_qa_sha256=sha(JOBS/'FINAL_CPU_QA.json'),input_preflight=audit)
    with path.open('x',encoding='utf-8') as f:json.dump(spec,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'frozen','protocol_sha256':sha(path),'gpu_started':False}))


def training_command(name,seed,prefix):
    return [sys.executable,'-u',str(HERE/'train_candidate.py'),'--train_manifest',str(V2/'manifests'/f'{prefix}_train.csv'),
            '--val_manifest',str(V2/'manifests'/f'{prefix}_val.csv'),'--out_dir',str(HERE/'runs'/name),
            '--seed',str(seed),'--epochs','20','--bs','32','--lr','1e-5','--head_lr','1e-4',
            '--weight_decay','.05','--patience','6','--num_workers','4','--weighted_sampler','--device','cuda']


def run(command,name,phase):
    validate_execution_protocol()
    with (JOBS/(name+'.log')).open('a',encoding='utf-8') as log:
        log.write('\n'+datetime.now(timezone.utc).isoformat()+' '+repr(command)+'\n');log.flush()
        child=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
        state(phase,task=name,child_pid=child.pid,command=command)
        code=child.wait()
    require(code==0,f'{name} exited {code}; preserve failed attempt and inspect log')
    state('between_stages',last_completed=name)


def train(name,seed,prefix):
    output=HERE/'runs'/name;marker=JOBS/(name+'.trained.json')
    if marker.exists():
        record=read(marker)
        for filename,digest in record['files_sha256'].items():
            require(sha(output/filename)==digest,'Completed candidate training output changed')
        return output
    command=training_command(name,seed,prefix)
    if (output/'last.pth').exists():command+=['--resume',str(output/'last.pth')]
    run(command,name+'.train','training')
    status=read(output/'status.json')
    require(status['completed_epoch']>=20 or status['no_improve']>=6,'Training stopped before planned end')
    import torch
    from candidate_model import validate_checkpoint_identity
    checkpoint=torch.load(output/'best.pth',map_location='cpu',weights_only=False)
    config=validate_checkpoint_identity(checkpoint)
    require(config['seed']==seed and checkpoint['best_epoch']==status['best_epoch'],'Candidate identity/selected epoch mismatch')
    del checkpoint
    write(marker,dict(task=name,manifest_prefix=prefix,status=status,files_sha256={
        n:sha(output/n) for n in ('best.pth','last.pth','config.json','history.csv','status.json')}))
    return output


def stage(name,command,parent,marker_name,phase):
    marker=JOBS/(name+'.done.json')
    if marker.exists():
        saved=read(marker);output=Path(saved['output'])
        require(output.resolve().is_relative_to(HERE),'Output outside this candidate')
        for n,digest in saved['outputs_sha256'].items():require(sha(output/n)==digest,'Completed stage file changed')
        return output
    parent.mkdir(parents=True,exist_ok=True)
    output=parent/('attempt_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    run(command+['--out-dir',str(output)],name,phase)
    require((output/marker_name).is_file(),'Stage completion marker missing')
    write(marker,dict(output=str(output),outputs_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    return output


def wait_for_previous():
    while True:
        previous=[read(NEW/v/'jobs/pipeline_status.json') for v in ('scale_candidate_v4','phikon_candidate_v5')]
        for item in previous:
            require(item.get('phase')!='failed','Previous workflow failed; investigate first')
        complete=all(s.get('phase')=='core_complete' for s in previous)
        live=any(alive(s.get(k)) for s in previous for k in ('pid','child_pid'))
        if complete and not live:
            upstream_guard();return
        require(complete or live,'Previous pipeline stopped incomplete')
        state('waiting_for_v5',gpu_started=False,previous_phases=[s.get('phase') for s in previous])
        time.sleep(60)


def controller():
    JOBS.mkdir(exist_ok=True)
    with (JOBS/'controller.lock').open('a+b') as lock:
        if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        validate_execution_protocol()
        if STATUS.exists():
            previous=read(STATUS)
            require(not any(alive(previous.get(k)) for k in ('pid','child_pid')),'Candidate controller or child still alive')
            if previous.get('phase')=='core_complete':
                print('Candidate complete; no duplicate training');return
        try:
            wait_for_previous()
            models={}
            for name,seed,prefix in TASKS:
                models[name]=train(name,seed,prefix)
                if prefix=='a6':
                    command=[sys.executable,'-u',str(HERE/'evaluate_candidate.py'),'--checkpoint',str(models[name]/'best.pth')]
                    for role in ('train','val','test'):command += [f'--{role}-manifest',str(V2/'manifests'/f'a6_{role}.csv')]
                    a6=stage('a6_evaluation',command,HERE/'evaluation','evaluation_summary.json','evaluating_a6')
            predictions=[]
            for seed in (42,43,44):
                for fold in (1,2,3):
                    name=f'a5_seed{seed}_fold{fold}'
                    command=[sys.executable,'-u',str(HERE/'predict_candidate.py'),'--dense-dir',str(NEW/'scale_candidate_v4/dense_scaled'),
                             '--checkpoint',str(models[name]/'best.pth'),'--train-manifest',str(V2/'manifests'/f'a5_fold{fold}_train.csv'),
                             '--val-manifest',str(V2/'manifests'/f'a5_fold{fold}_val.csv'),'--seed',str(seed),'--fold',str(fold),
                             '--batch-size','64','--num-workers','0','--device','cuda']
                    predictions.append(stage(name+'.inference',command,HERE/'predictions'/name,'COMPLETE.json','predicting_a5'))
            aggregate=stage('a5_aggregate',[sys.executable,'-u',str(HERE/'aggregate_candidate.py'),
                            '--dense-dir',str(NEW/'scale_candidate_v4/dense_scaled'),'--runs',*map(str,predictions)],HERE/'aggregate','COMPLETE.json','aggregating_a5')
            statistics=stage('a5_statistics',[sys.executable,'-u',str(HERE/'analyze_candidate.py'),
                             '--aggregate-dir',str(aggregate),'--outcomes-dir',str(NEW/'outcomes_v1'),'--include-n18'],
                             HERE/'statistics','COMPLETE.json','analyzing_a5')
            require(read(statistics/'COMPLETE.json')['primary_n']==21,'Primary animal count differs')
            validate_execution_protocol()
            state('core_complete',a6_evaluation=str(a6),aggregate=str(aggregate),statistics=str(statistics),
                  next_steps='Independent validation, report all candidate/v2 results, heatmap ROI audit and manuscript integration')
        except Exception as exc:
            state('failed',error=repr(exc));raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--freeze',action='store_true');group.add_argument('--run',action='store_true')
    args=ap.parse_args()
    if args.check:print(json.dumps(cpu_check(),indent=2))
    elif args.freeze:freeze()
    else:controller()


if __name__=='__main__':main()
