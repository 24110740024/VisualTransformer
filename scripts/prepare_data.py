"""Materialize released manifests and optionally map authorized local source files."""
import argparse,csv,gzip,hashlib,io,json,shutil,zipfile,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rebuild_v6'))
from manifest_archive import open_manifest_archive
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--png-root',type=Path);p.add_argument('--wsi-root',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);man=a.out/'manifests';man.mkdir()
    index=json.loads((ROOT/'data/MANIFEST_INDEX.json').read_text())
    with open_manifest_archive() as z:
        if z.testzip() is not None:raise ValueError('Bad archive CRC')
        for name,h in index['members_sha256'].items():
            if Path(name).name!=name:raise ValueError('Unsafe path')
            b=z.read(name)
            if hashlib.sha256(b).hexdigest()!=h:raise ValueError('Manifest hash mismatch')
            (man/name).write_bytes(b)
    with (man/'image_assets.csv').open(newline='') as f:assets={r['source_png_sha256']:r['pixel_sha256'] for r in csv.DictReader(f)}
    if a.png_root:
        images=a.out/'images';images.mkdir();found=set()
        for path in a.png_root.rglob('*.png'):
            file_hash=digest(path)
            if file_hash in assets:
                pix=assets[file_hash]
                with Image.open(path) as im:
                    if im.size!=(224,224) or hashlib.sha256(b'224,224,RGB:'+im.convert('RGB').tobytes()).hexdigest()!=pix:raise ValueError('Decoded pixel mismatch')
                shutil.copyfile(path,images/(pix+'.png'));found.add(file_hash)
        if found!=set(assets):raise ValueError(f'Missing {len(set(assets)-found)} required PNGs; supplied files retained for inspection')
    if a.wsi_root:
        with (man/'coordinates_all21.csv').open(newline='') as f:needed={r['source_sha256'] for r in csv.DictReader(f)}
        wsi=a.out/'wsi';wsi.mkdir();found=set()
        for path in a.wsi_root.rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.tif','.tiff','.svs'):
                h=digest(path)
                if h in needed:shutil.copyfile(path,wsi/(h+'.tif'));found.add(h)
        if found!=needed:raise ValueError(f'Missing {len(needed-found)} original WSI hashes')
    print(json.dumps(dict(manifest_directory=str(man),images_mapped=bool(a.png_root),wsi_mapped=bool(a.wsi_root))))
if __name__=='__main__':main()
