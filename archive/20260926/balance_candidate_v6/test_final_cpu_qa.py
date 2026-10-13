"""Independent change checks plus inherited full-loop interruption/resume test."""
from pathlib import Path
from datetime import datetime,timezone
import unittest,ast,json,copy
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F
import candidate_model as cm
import scale_geometry as geo
import scale_transforms as st
import predict_candidate as pc
import test_training_component as inherited
import test_sampling_binding as sampling_tests

class ScaleChecks(unittest.TestCase):
    def test_eval_preserves_every_pixel(self):
        a=np.random.default_rng(10).integers(0,256,(224,224,3),dtype=np.uint8)
        expected=F.normalize(F.to_tensor(Image.fromarray(a)),[.485,.456,.406],[.229,.224,.225])
        self.assertTrue(torch.equal(st.build_transforms(False)(Image.fromarray(a)),expected))
        with self.assertRaises(ValueError):st.build_transforms(False)(Image.new('RGB',(512,512)))
    def test_training_has_no_zoom_or_crop(self):
        names=[type(x).__name__ for x in st.build_transforms(True).transforms]
        self.assertEqual(names,['Require224','RandomHorizontalFlip','ColorJitter','ToTensor','Normalize'])
    def test_native_center_and_padding_algebra(self):
        cfg=geo.config()
        for source,value in cfg['sizes'].items():
            old={'source_sha256':source,'x':256,'y':512,'crop_size':512,'pad_right':0,'pad_bottom':0,'source_width':2048,'source_height':2048}
            row=geo.transform(old,cfg);size=value['crop_size']
            self.assertEqual(row['x']+size/2-(old['x']+256),-.5 if size==409 else 0)
            self.assertLessEqual(row['x']+size,old['x']+512)
            self.assertAlmostEqual(size*value['mpp'],112,delta=.1)
    def test_all_frozen_coordinates_independent_derivation(self):
        rows,cases,frozen=pc.load_frozen_coordinates(geo.DIRECTORY)
        self.assertEqual(len(rows),132032);self.assertEqual(len(cases),21)
        old,_,_=pc._load_original_coordinates(cm.NEW/'dense_v1')
        for a,b in zip(old,rows,strict=True):
            self.assertEqual((a['case_id'],a['source_sha256'],a['outer_fold']),(b['case_id'],b['source_sha256'],b['outer_fold']))
            self.assertEqual(b['x']-a['x'],51 if b['crop_size']==409 else 52)
        with self.assertRaises(ValueError):pc.load_frozen_coordinates(cm.NEW/'dense_v1')
    def test_both_dense_sizes_and_invalid_input(self):
        for size in (408,409):
            tensor=pc.preprocess_crop(np.full((size,size,3),255,dtype=np.uint8))
            self.assertEqual(tuple(tensor.shape),(3,224,224));self.assertTrue(torch.isfinite(tensor).all())
        for size in (224,512):
            with self.assertRaises(ValueError):pc.preprocess_crop(np.zeros((size,size,3),dtype=np.uint8))
    def test_numeric_algorithms_unchanged_from_v3(self):
        def funcs(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        files={'predict_candidate.py':['TiledJpegReader','flips4_probabilities','validate_checkpoint','verify_sources'],
               'evaluate_candidate.py':['predict','select_validation_threshold','metrics','bootstrap_auc'],
               'aggregate_candidate.py':['validate_prediction_rows','equal_weight_seed_scores'],
               'analyze_candidate.py':['validate_score_tables','probability']}
        for name,names in files.items():
            a=funcs((cm.NEW/'pretrained_candidate_v3'/name).read_text(encoding='utf-8-sig'))
            b=funcs((cm.HERE/name).read_text(encoding='utf-8-sig'))
            for f in names:self.assertEqual(a[f],b[f],(name,f))
    def test_old_v3_checkpoint_rejected(self):
        with self.assertRaises(ValueError):cm.validate_checkpoint_identity({'format_version':3,'initialization':cm.INITIALIZATION})

if __name__=='__main__':
    torch.set_num_threads(2)
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(inherited.ComponentChecks),
                             unittest.defaultTestLoader.loadTestsFromTestCase(ScaleChecks),
                             unittest.defaultTestLoader.loadTestsFromTestCase(sampling_tests.BindingChecks)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    value={'utc':datetime.now(timezone.utc).isoformat(),'passed':result.wasSuccessful(),'tests_run':result.testsRun,
           'failures':len(result.failures),'errors':len(result.errors),'cuda_context_initialized':torch.cuda.is_initialized(),
           'scope':'Shared v4 geometry; full224 transform; real v6 sampler binding and refusal checks; CPU full-loop crash/resume with same sampling math; no real crop re-read in this QA',
           'files_sha256':{str(p):cm.sha(p) for p in sorted(cm.HERE.glob('*.py'))}}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path=cm.HERE/'jobs'/f'CPU_QA_{stamp}.json';path.write_text(json.dumps(value,indent=2),encoding='utf-8')
    if result.wasSuccessful():
        target=cm.HERE/'jobs/FINAL_CPU_QA.json'
        with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
