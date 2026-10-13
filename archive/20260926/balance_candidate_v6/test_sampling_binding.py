"""Production sampler identities, tampering rejection, and fixed loss checks."""
import csv
import copy
import unittest
import numpy as np
import candidate_model as cm
import train_candidate as trainer
import sampling_binding as sb
from sampling import probabilities

class BindingChecks(unittest.TestCase):
    def test_all_real_training_roles(self):
        for prefix in ('a6','a5_fold1','a5_fold2','a5_fold3'):
            path=cm.V2/'manifests'/f'{prefix}_train.csv'
            rows=trainer.read_manifest(path,'train')
            p,identity=sb.load_training_probabilities(path,rows)
            self.assertTrue(np.array_equal(p,probabilities(rows)))
            self.assertEqual(identity['prefix'],prefix)
            self.assertIsNone(identity['loss_class_weights'])
            self.assertEqual(identity['samples_per_epoch'],len(rows))
    def test_refuse_held_out_roles_and_old_loss(self):
        with self.assertRaises(ValueError):sb.binding_identity(cm.V2/'manifests/a6_val.csv')
        with self.assertRaises(ValueError):sb.binding_identity(cm.V2/'manifests/a6_test.csv')
        from run_candidate_pipeline import training_command
        args=trainer.parse_args(training_command('a6_seed42',42,'a6')[3:])
        self.assertEqual(cm.enforce_settings(args),'a6')
        args.class_weight=True
        with self.assertRaisesRegex(ValueError,'class_weight'):cm.enforce_settings(args)
    def test_probability_row_tampering(self):
        rows=[dict(path=f'{cm.HERE}/synthetic_{i}.png',label=i%2,source_id='a' if i<3 else 'b',
                   pixel_sha256=str(i),split='train',**{'class':'Lesion' if i%2==0 else 'Normal'}) for i in range(6)]
        p=probabilities(rows)
        saved=[dict(row_index=i,path=r['path'],label=r['label'],source_id=r['source_id'],pixel_sha256=r['pixel_sha256'],
                    probability=float(v),expected_repeats_per_epoch=len(rows)*float(v)) for i,(r,v) in enumerate(zip(rows,p))]
        self.assertTrue(np.array_equal(sb.validate_probability_rows(rows,saved),p))
        for key,value in [('row_index',1),('path',str(cm.HERE/'other.png')),('label',1),('source_id','other'),
                          ('pixel_sha256','changed'),('probability',.1),('expected_repeats_per_epoch',0)]:
            changed=copy.deepcopy(saved);changed[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):sb.validate_probability_rows(rows,changed)
        with self.assertRaises(ValueError):sb.validate_probability_rows(rows,saved[:-1])
