"""Synthetic CPU tests, including interruption between last and best commits."""
from contextlib import ExitStack, redirect_stdout
import csv
import gc
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import torch
from torch import nn

import candidate_model as cm
import candidate_guard as guard
import train_candidate as trainer


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Linear(3,5)
        self.heads=nn.Sequential(nn.Linear(5,2))
    def forward(self,x):
        return self.heads(self.backbone(x.mean((-2,-1))).tanh())


class ComponentChecks(unittest.TestCase):
    def test_exact_official_backbone_and_new_head(self):
        model=cm.create_model(pretrained=True).eval()
        original=torch.load(cm.WEIGHT_PATH,map_location='cpu',weights_only=True)
        state=model.state_dict()
        for name,value in original.items():
            if not name.startswith('heads.'):
                # Official torchvision MLPBlock._load_from_state_dict migrates
                # pre-v2 module names, without changing any tensor values.
                current=name.replace('.mlp.linear_1.','.mlp.0.').replace('.mlp.linear_2.','.mlp.3.')
                self.assertTrue(torch.equal(state[current],value),name)
        self.assertEqual(tuple(state['heads.head.weight'].shape),(2,768))
        self.assertEqual(tuple(original['heads.head.weight'].shape),(1000,768))
        with torch.inference_mode(): output=model(torch.zeros(1,3,224,224))
        self.assertEqual(tuple(output.shape),(1,2))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.cuda.is_initialized())
        del model,original,state
        gc.collect()

    def test_optimizer_partition_and_cosine_ratio(self):
        model=Toy()
        optimizer=cm.build_optimizer(model)
        groups=optimizer.param_groups
        self.assertEqual([g['group_name'] for g in groups],['backbone','head'])
        self.assertEqual([g['lr'] for g in groups],[1e-5,1e-4])
        ids=[id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)
        for _ in range(8):
            optimizer.zero_grad()
            model(torch.ones(2,3,4,4)).sum().backward()
            optimizer.step();scheduler.step()
            self.assertAlmostEqual(groups[1]['lr']/groups[0]['lr'],10.)

    def test_v1_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError,'v6 checkpoint'):
            cm.validate_checkpoint_identity({'format_version':1,'initialization':'random'})

    def test_not_ready_for_gpu_without_protocol(self):
        with tempfile.TemporaryDirectory(dir=cm.HERE/'jobs') as directory:
            with patch.object(guard,'HERE',Path(directory)):
                with self.assertRaisesRegex(ValueError,'not frozen'):
                    guard.validate_execution_protocol()

    def test_frozen_settings_reject_learning_rate_changes(self):
        frozen=cm.read(cm.V2/'FROZEN.json')['manifests']
        args=trainer.parse_args(['--train_manifest',frozen['a6_train']['path'],
            '--val_manifest',frozen['a6_val']['path'],'--out_dir',str(cm.HERE/'runs/a6_seed42'),
            '--device','cuda','--weighted_sampler'])
        self.assertEqual(cm.enforce_settings(args),'a6')
        args.head_lr=3e-4
        with self.assertRaisesRegex(ValueError,'head_lr'): cm.enforce_settings(args)

    def test_full_synthetic_resume_matches_uninterrupted(self):
        # The real training loop, data augmentation, weighted sampler, AdamW,
        # two LR groups and cosine scheduler run on synthetic CPU images.
        # Only production gate/official large model are replaced for this test.
        (cm.HERE/'jobs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cm.HERE/'jobs') as name:
            root=Path(name); rng=np.random.default_rng(314)
            for role in ('train','val'):
                rows=[]
                for i in range(8 if role=='train' else 4):
                    image=root/f'{role}{i}.png'
                    Image.fromarray(rng.integers(0,256,(224,224,3),dtype=np.uint8)).save(image)
                    rows.append(dict(path=str(image),label=i%2,source_id=(f'{role}a' if i<5 else f'{role}b'),split=role,
                                     pixel_sha256=cm.sha(image),historical_file=f'{role}{i}',**{'class':'Lesion' if i%2==0 else 'Normal'}))
                with (root/f'{role}.csv').open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            def args(output,resume=False):
                result=['--train_manifest',str(root/'train.csv'),'--val_manifest',str(root/'val.csv'),
                        '--out_dir',str(output),'--epochs','2','--bs','2','--num_workers','0',
                        '--device','cpu','--weighted_sampler']
                return result+(['--resume',str(output/'last.pth')] if resume else [])
            real_sha=trainer.sha256
            def test_sha(path):
                return 'synthetic-protocol' if Path(path).name=='EXECUTION_PROTOCOL.json' else real_sha(path)
            with ExitStack() as stack:
                from sampling import probabilities
                stack.enter_context(patch.object(trainer,'load_training_probabilities',side_effect=lambda path,rows:(probabilities(rows),{'policy':'synthetic-same-formula'})))
                stack.enter_context(patch.object(trainer,'enforce_settings',return_value='synthetic'))
                stack.enter_context(patch.object(trainer,'validate_execution_protocol',return_value={}))
                stack.enter_context(patch.object(trainer,'gpu_guard',return_value={}))
                stack.enter_context(patch.object(trainer,'create_model',side_effect=lambda **_:Toy()))
                stack.enter_context(patch.object(trainer,'verify_weights',return_value={'sha256':'synthetic-only'}))
                stack.enter_context(patch.object(trainer,'sha256',side_effect=test_sha))
                stack.enter_context(patch('torch.cuda.is_available',return_value=False))
                stack.enter_context(redirect_stdout(io.StringIO()))
                trainer.main(args(root/'uninterrupted'))
                save=trainer.atomic_torch_save
                def fail_between_commits(value,path):
                    if Path(path).name=='best.pth' and value['epoch']==1:
                        raise RuntimeError('simulated interruption after last before best')
                    return save(value,path)
                with patch.object(trainer,'atomic_torch_save',side_effect=fail_between_commits):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                        trainer.main(args(root/'resumed'))
                self.assertTrue((root/'resumed/last.pth').exists())
                self.assertFalse((root/'resumed/best.pth').exists())
                trainer.main(args(root/'resumed',resume=True))
                a=torch.load(root/'uninterrupted/last.pth',weights_only=False,map_location='cpu')
                b=torch.load(root/'resumed/last.pth',weights_only=False,map_location='cpu')
                self.assertEqual(a['epoch'],2);self.assertEqual(b['epoch'],2)
                self.assertEqual(a['best_epoch'],b['best_epoch'])
                self.assertEqual(a['no_improve'],b['no_improve'])
                for key in a['model']: self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)
                for key in ('lr_used','head_lr_used','val_auc_lesion','train_loss','val_loss'):
                    self.assertEqual([r[key] for r in a['history']],[r[key] for r in b['history']],key)
                for key in ('sampling','train_workers','val_workers'):
                    self.assertTrue(torch.equal(a['rng']['generators'][key],b['rng']['generators'][key]))
                self.assertTrue((root/'resumed/best.pth').exists())
                self.assertEqual(json.loads((root/'resumed/status.json').read_text())['completed_epoch'],2)


if __name__=='__main__':
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ComponentChecks))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    audit={'utc':datetime.now(timezone.utc).isoformat(),'passed':result.wasSuccessful(),
           'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
           'gpu_work_started':False,'complete_pipeline_ready':False,
           'scope':'Training component only. Includes official backbone identity and synthetic CPU full-loop interruption/resume.',
           'files_sha256':{str(p):cm.sha(p) for p in [Path(__file__),cm.HERE/'candidate_model.py',cm.HERE/'candidate_guard.py',cm.HERE/'train_candidate.py']}}
    (cm.HERE/'jobs').mkdir(exist_ok=True)
    (cm.HERE/'jobs'/f'training_component_tests_{stamp}.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
