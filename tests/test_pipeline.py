"""CPU contract and mathematical equivalence tests; no study model is retrained."""
import ast,csv,hashlib,io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'rebuild_v6'))
from manifest_archive import open_manifest_archive
from sampling import probabilities
from metrics import select_validation_threshold,metrics,bootstrap_auc
from robustness import holm,self_test
from data import validate_roles,write_csv
from jpeg_decoder import decode_jpeg_tile
from dense_core import preprocess_crop,flips4_probabilities

def functions(p):
    return {n.name:n for n in ast.parse(p.read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

class PipelineTests(unittest.TestCase):
    def test_positive_class_and_validation_threshold(self):
        y=np.array([1,1,0,0]);p=np.array([.9,.8,.2,.1]);lock=select_validation_threshold(y,p)
        self.assertEqual(lock['threshold'],.8);self.assertEqual(metrics(y,p,.8)['TP'],2)
        self.assertIsNone(select_validation_threshold(y,1-p)['threshold'])
    def test_holm_and_simpson_diagnostics(self):
        self.assertTrue(np.allclose(holm([.01,.04,.03,.002]),[.03,.06,.06,.008]));self_test()
    def test_cluster_bootstrap_is_clustered(self):
        b=bootstrap_auc(np.array([1,0,1,0]),np.array([.9,.1,.8,.2]),np.array(['a','a','b','b']),['a','b'],25,25)
        self.assertEqual(b['source_wsi_cluster']['percentile_95_ci'],[1.,1.])
    def test_source_sampling_mass(self):
        rows=[]
        for c,s,n in [('0','a',4),('0','b',1),('1','c',2)]:
            for i in range(n):rows.append(dict(label=c,source_id=s,pixel_sha256=f'{c}-{s}-{i}',split='train',**{'class':'Lesion' if c=='0' else 'Normal'}))
        p=probabilities(rows);self.assertAlmostEqual(p[:4].sum(),1/3);self.assertAlmostEqual(p[4],1/6);self.assertAlmostEqual(p[5:].sum(),.5)
        rows[0]['split']='val'
        with self.assertRaises(ValueError):probabilities(rows)
    def test_duplicate_sampling_rejected(self):
        rows=[dict(label=str(c),source_id=str(c),pixel_sha256='same',split='train',**{'class':'Lesion' if c==0 else 'Normal'}) for c in (0,1)]
        with self.assertRaises(ValueError):probabilities(rows)
    def test_manifest_source_leakage_and_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);paths={}
            for j,role in enumerate(('train','val')):
                rows=[]
                for c in (0,1):
                    im=Image.new('RGB',(224,224),(j*80+c*20,10,20));im.save(d/f'{role}{c}.png')
                    rows.append(dict(path=f'{role}{c}.png',label=c,source_id=role+str(c),canonical_animal_id=role+str(c),split=role,pixel_sha256=hashlib.sha256(b'224,224,RGB:'+im.tobytes()).hexdigest(),**{'class':'Lesion' if c==0 else 'Normal'}))
                paths[role]=d/(role+'.csv');write_csv(paths[role],rows)
            validate_roles(paths,True)
            rows[0]['source_id']='train0';write_csv(paths['val'],rows)
            with self.assertRaises(ValueError):validate_roles(paths,True)
    def test_jpeg_decoding(self):
        b=io.BytesIO();Image.new('RGB',(16,16),(100,50,20)).save(b,format='JPEG')
        im=decode_jpeg_tile(b.getvalue(),(16,16));self.assertEqual(im.size,(16,16))
        with self.assertRaises(ValueError):decode_jpeg_tile(b.getvalue(),(32,32))
    def test_dense_resize_and_shape(self):
        p=preprocess_crop(np.full((409,409,3),128,dtype=np.uint8));self.assertEqual(tuple(p.shape),(3,224,224))
        self.assertTrue(torch.isfinite(p).all())
        with self.assertRaises(ValueError):preprocess_crop(np.zeros((512,512,3),dtype=np.uint8))
    def test_tta_averages_probabilities(self):
        class Dummy(torch.nn.Module):
            def forward(self,x):
                v=x[:,0,0,0];return torch.stack((v,-v),1)
        x=torch.tensor([[[[0.,1.],[2.,3.]]]])
        expected=torch.stack([Dummy()(torch.flip(x,dims=d) if d else x).softmax(1)[:,0] for d in (None,(-1,),(-2,),(-2,-1))]).mean(0)
        self.assertTrue(torch.equal(flips4_probabilities(Dummy(),x),expected))
    def test_numerical_functions_match_frozen_sources(self):
        a=ROOT/'archive/20260926/balance_candidate_v6';b=ROOT/'rebuild_v6'
        pairs=[('sampling.py','sampling.py',['probabilities','summary']),('scale_transforms.py','scale_transforms.py',['Require224','build_transforms']),('candidate_model.py','candidate_model.py',['create_model','build_optimizer']),('evaluate_candidate.py','metrics.py',['select_validation_threshold','metrics','bootstrap_auc']),('predict_candidate.py','dense_core.py',['TiledJpegReader','preprocess_crop','DenseCropDataset','flips4_probabilities']),('analyze_candidate.py','association_core.py',['paired_bootstrap_ci','association_results'])]
        for original,portable,names in pairs:
            x,y=functions(a/original),functions(b/portable)
            for name in names:self.assertEqual(ast.dump(x[name]),ast.dump(y[name]),name)
        orig=functions(a/'train_candidate.py')['main'];new=functions(b/'train.py')['main']
        # Everything from RNG seeding through initialization, training, validation,
        # checkpoint selection and early stopping is the original AST.
        def body(node):
            idx=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='random.seed')
            return [ast.dump(n) for n in node.body[idx:]]
        self.assertEqual(body(orig),body(new))
    def test_all_published_roles_remain_disjoint(self):
        with open_manifest_archive() as z:
            tables={n:list(csv.DictReader(io.StringIO(z.read(n).decode()))) for n in z.namelist() if n not in ('coordinates_all21.csv','image_assets.csv')}
            for prefix in ('a6','a5_fold1','a5_fold2','a5_fold3'):
                names=[n for n in tables if n.startswith(prefix+'_')]
                for i,a in enumerate(names):
                    for b in names[i+1:]:
                        for key in ('canonical_animal_id','source_id','pixel_sha256'):
                            self.assertFalse({r[key] for r in tables[a]}&{r[key] for r in tables[b]})
            coords=list(csv.DictReader(io.StringIO(z.read('coordinates_all21.csv').decode())))
            for fold in (1,2,3):
                held={r['source_sha256'] for r in coords if int(r['outer_fold'])==fold}
                held_animals={r['case_id'] for r in coords if int(r['outer_fold'])==fold}
                for role in ('train','val'):
                    self.assertFalse(held & {r['source_id'] for r in tables[f'a5_fold{fold}_{role}.csv']})
                    self.assertFalse(held_animals & {r['canonical_animal_id'] for r in tables[f'a5_fold{fold}_{role}.csv']})
    def test_manifest_archive_integrity(self):
        spec=json.loads((ROOT/'data/MANIFEST_INDEX.json').read_text())
        with open_manifest_archive() as z:
            self.assertIsNone(z.testzip())
            for name,digest in spec['members_sha256'].items():self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),digest)

    def test_manifest_split_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);b=io.BytesIO()
            with zipfile.ZipFile(b,'w') as z:z.writestr('sample.csv','a,b\n1,2\n')
            full=b.getvalue();chunks=[full[:30],full[30:]];parts=[]
            for i,chunk in enumerate(chunks):
                name=f'part{i}';(d/name).write_bytes(chunk)
                parts.append(dict(path=name,bytes=len(chunk),sha256=hashlib.sha256(chunk).hexdigest()))
            (d/'MANIFEST_ARCHIVE.json').write_text(json.dumps(dict(parts=parts,bytes=len(full),sha256=hashlib.sha256(full).hexdigest())))
            with open_manifest_archive(d) as z:self.assertEqual(z.read('sample.csv'),b'a,b\n1,2\n')
            (d/'part0').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):open_manifest_archive(d)

if __name__=='__main__':unittest.main()
