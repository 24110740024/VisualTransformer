# tools/split_from_qupath.py
# -*- coding: utf-8 -*-
import os, argparse, random, shutil
from pathlib import Path
from glob import glob

NORMAL_BUCKETS = ["Normal", "Normal-Far", "Normal-Red"]  # 这三类都归为负样本

def iter_images(d: Path):
    if not d.exists(): return []
    exts = ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp")
    files = []
    for e in exts:
        files += list(d.rglob(e))
    return files

def collect(tiles_root: Path):
    """扫描 tiles_v2/<WSI名> 下的四个文件夹，合并出 {wsi_id: {"Lesion":[...], "Normal":[...]}}"""
    out = {}
    for wsi_dir in sorted(tiles_root.iterdir()):
        if not wsi_dir.is_dir():
            continue
        wsi_id = wsi_dir.name
        lesion = iter_images(wsi_dir / "Lesion")
        normal = []
        for b in NORMAL_BUCKETS:
            normal += iter_images(wsi_dir / b)
        if not lesion and not normal:
            continue
        out[wsi_id] = {"Lesion": lesion, "Normal": normal}
    if not out:
        raise SystemExit(f"[ERROR] No tiles found under: {tiles_root}")
    return out

def split_wsi(wsi_ids, val=0.15, test=0.15, seed=42):
    ids = list(sorted(wsi_ids))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_val = int(round(n * val))
    n_test = int(round(n * test))
    n_train = max(0, n - n_val - n_test)
    return {
        "train": set(ids[:n_train]),
        "val":   set(ids[n_train:n_train+n_val]),
        "test":  set(ids[n_train+n_val:])
    }

def ensure_dirs(root: Path):
    for sp in ["train","val","test"]:
        for cls in ["Lesion","Normal"]:
            (root / sp / cls).mkdir(parents=True, exist_ok=True)

def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        try:
            os.link(src, dst)  # 硬链接（Windows/NTFS 也支持）
            return
        except Exception:
            pass  # 链接失败就退回复制
    shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_root", required=True, help="形如 tiles_v2/<WSI名>/{Lesion,Normal,Normal-Far,Normal-Red}")
    ap.add_argument("--out_root",   required=True, help="输出根目录，例如 vit_wsi/data224")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["copy","link"], default="copy", help="复制或硬链接（节省空间）")
    args = ap.parse_args()

    tiles_root = Path(args.tiles_root)
    out_root   = Path(args.out_root)
    ensure_dirs(out_root)

    # 1) 收集
    bank = collect(tiles_root)  # {wsi_id: {"Lesion":[...], "Normal":[...]}}
    wsi_ids = list(bank.keys())
    print(f"[INFO] Found {len(wsi_ids)} WSIs.")

    # 2) 按 WSI 划分
    split = split_wsi(wsi_ids, val=args.val, test=args.test, seed=args.seed)
    for k,v in split.items():
        print(f"[INFO] {k}: {len(v)} WSIs")

    # 3) 拷贝/链接
    stats = {sp: {"Lesion":0, "Normal":0} for sp in ["train","val","test"]}
    for sp, wsi_set in split.items():
        for wsi in sorted(wsi_set):
            for cls in ["Lesion","Normal"]:
                for src in bank[wsi][cls]:
                    dst = out_root / sp / cls / f"{wsi}__{src.name}"
                    copy_or_link(src, dst, args.mode)
                    stats[sp][cls] += 1

    # 4) 打印统计
    print("\n[INFO] Done. Counts per split:")
    for sp in ["train","val","test"]:
        print(f"  {sp:5s}  Lesion={stats[sp]['Lesion']:7d}   Normal={stats[sp]['Normal']:7d}")
    print(f"[INFO] Output dir: {out_root.resolve()}")

if __name__ == "__main__":
    main()
