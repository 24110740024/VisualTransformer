# -*- coding: utf-8 -*-
# predict_dual_tta_safe.py — Case-level inference & scoring for vit_wsi

import os, re, time, argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- import model from project ----
try:
    from model import ViT
except Exception:
    import sys
    sys.path.append(os.path.dirname(__file__) or ".")
    from model import ViT

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# -------------------- helpers --------------------
def list_case_dirs(root: str):
    return sorted([os.path.join(root, d) for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))])

def list_images_recursively(root: str):
    files = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(IMG_EXTS):
                files.append(os.path.join(r, f))
    files.sort()
    return files

def parse_xy_from_name(p: str):
    fn = os.path.splitext(os.path.basename(p))[0]
    m = re.search(r'[xX](\d+)[^0-9]+[yY](\d+)', fn)
    if m: return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d+)[_-](\d+)$', fn)
    if m: return int(m.group(1)), int(m.group(2))
    nums = re.findall(r'\d+', fn)
    if len(nums) >= 2: return int(nums[-2]), int(nums[-1])
    return -1, -1

# ----------- Dataset with HSV tissue filter -----------
class ImageFilesDataset(Dataset):
    def __init__(self, files, image_size=224,
                 use_hsv_filter=False, min_sat=10.0, max_val=245.0):
        """
        use_hsv_filter: 是否启用近白/低饱和过滤
        min_sat: S 平均值低于该阈视为低饱和（0~255）
        max_val: V 平均值高于该阈视为近白/高亮（0~255）
        """
        self.files = files
        self.image_size = image_size
        self.use_hsv_filter = use_hsv_filter
        self.min_sat = float(min_sat)
        self.max_val = float(max_val)

    def __len__(self): return len(self.files)

    def _read_rgb(self, p):
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            return False, None

        # 组织过滤：低饱和或近白则认为无组织/背景
        if self.use_hsv_filter:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv)
            if (S.mean() < self.min_sat) or (V.mean() > self.max_val):
                return False, None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[0], rgb.shape[1]) != (self.image_size, self.image_size):
            rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        f = rgb.astype(np.float32)/255.0
        # ImageNet normalization
        f[...,0]=(f[...,0]-0.485)/0.229
        f[...,1]=(f[...,1]-0.456)/0.224
        f[...,2]=(f[...,2]-0.406)/0.225
        f = np.transpose(f,(2,0,1))
        return True, f

    def __getitem__(self, i):
        p = self.files[i]
        ok, img = self._read_rgb(p)
        if not ok: return None, p
        return torch.from_numpy(img), p

# ----------- safe collate：在拼批阶段就过滤坏样本 -----------
def safe_collate(batch):
    good = [b for b in batch if b is not None and b[0] is not None]
    if not good:
        return None
    imgs = torch.stack([b[0] for b in good], 0)
    paths = [b[1] for b in good]
    return imgs, paths

# --------- aggregate: patch -> case（更保守，减少“泛红”） ---------
def aggregate_case(prob_pos_list, cls_list,
                   topk_ratio=0.05, thr=0.58,
                   topk_gamma=2.0, mean_trim=0.10, area_strict=True):
    """
    - mean_prob：winsor 轻裁剪（抑制极值）
    - area_prop：严格阈值 max(thr, q70)
    - topk_mean：权重∝p^gamma 的加权 top-k
    """
    prob = np.asarray(prob_pos_list, dtype=float)
    n = len(prob)
    if n == 0:
        return dict(n_patches=0, mean_prob=np.nan, med_prob=np.nan,
                    topk_mean=np.nan, area_prop=np.nan,
                    thr=thr, topk_ratio=topk_ratio), None

    # mean_prob winsor
    if 0 < mean_trim < 1 and n >= 5:
        lo = float(np.quantile(prob, mean_trim/2))
        hi = float(np.quantile(prob, 1 - mean_trim/2))
        prob_w = np.clip(prob, lo, hi)
        mean_prob = float(prob_w.mean())
    else:
        mean_prob = float(prob.mean())
    med_prob = float(np.median(prob))

    # area_prop 严格阈
    area_thr = float(max(thr, np.quantile(prob, 0.70))) if area_strict and n >= 10 else float(thr)
    area_prop = float((prob >= area_thr).mean())

    # top-k 加权
    k = max(1, int(round(n * float(topk_ratio))))
    idx = np.argsort(-prob)[:k]
    if topk_gamma is None or topk_gamma == 1.0:
        w = np.ones_like(prob[idx])
    else:
        w = np.power(np.clip(prob[idx], 1e-6, 1.0), float(topk_gamma))
    w = w / (w.sum() + 1e-8)
    topk_mean = float((prob[idx] * w).sum())

    fused = None
    if cls_list is not None and len(cls_list) == n:
        cls = np.stack(cls_list, axis=0)  # [N, D]
        fused = (cls[idx] * w[:, None]).sum(axis=0)

    injury = dict(n_patches=n, mean_prob=mean_prob, med_prob=med_prob,
                  topk_mean=topk_mean, area_prop=area_prop,
                  thr=thr, topk_ratio=topk_ratio)
    return injury, fused

def get_cls_from_seq(model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        seq = model.forward_features(x)
        if isinstance(seq, dict):
            for _, v in seq.items():
                if torch.is_tensor(v):
                    seq = v; break
        if torch.is_tensor(seq):
            if seq.ndim == 3: return seq[:, 0, :]
            if seq.ndim == 2: return seq
    with torch.no_grad():
        seq = model.patch_embed(x)
        seq = seq.flatten(2).transpose(1, 2)
        cls = model.cls_token.expand(x.size(0), 1, model.embed_dim)
        seq = torch.cat([cls, seq], dim=1)
        if hasattr(model, "pos_embed"): seq = seq + model.pos_embed[:, :seq.size(1), :]
        if hasattr(model, "pos_drop"):  seq = model.pos_drop(seq)
        if hasattr(model, "blocks"):
            seq = model.blocks(seq)
            if hasattr(model, "norm"): seq = model.norm(seq)
        elif hasattr(model, "encoder"):
            seq = model.encoder(seq)
        return seq[:, 0, :]

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--amp", action="store_true")

    # 新增：温度缩放（T>1“降温”）
    ap.add_argument("--temp", type=float, default=1.0,
                    help="temperature for softmax calibration (>1 cools probabilities)")

    # HSV 过滤相关开关与阈值
    ap.add_argument("--hsv_filter", action="store_true",
                    help="enable HSV tissue filter to drop near-white/low-saturation patches")
    ap.add_argument("--min_sat", type=float, default=10.0,
                    help="mean S below this (0~255) -> drop")
    ap.add_argument("--max_val", type=float, default=245.0,
                    help="mean V above this (0~255) -> drop")

    # 更保守默认（可被命令行覆盖）
    ap.add_argument("--thr", type=float, default=0.58)
    ap.add_argument("--topk_ratio", type=float, default=0.05)
    ap.add_argument("--topk_gamma", type=float, default=2.0)
    ap.add_argument("--mean_trim",  type=float, default=0.10)
    ap.add_argument("--area_strict", action="store_true")

    ap.add_argument("--tta", choices=["none", "flips4"], default="flips4")
    ap.add_argument("--out_dir", default="./preds_dual")
    ap.add_argument("--dump_patch_csv", action="store_true")
    ap.add_argument("--dump_all_patches", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.out_dir, run_id)
    os.makedirs(run_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    margs = ckpt.get("args", {})
    classes = ckpt.get("classes", ["Lesion", "Normal"])
    num_classes = len(classes)
    pos_idx = classes.index("Lesion") if "Lesion" in classes else 0
    print(f"[info] classes={classes}  pos_idx={pos_idx}")

    model = ViT(
        in_channels=margs.get("in_channels", 3),
        img_size=margs.get("img_size", args.image_size),
        patch_size=margs.get("patch_size", 16),
        num_classes=num_classes,
        embed_dim=margs.get("embed_dim", 768),
        depth=margs.get("depth", 12),
        heads=margs.get("heads", 12),
        mlp_dim=margs.get("mlp_dim", 3072),
        dropout=margs.get("dropout", 0.1),
        pos_dropout=margs.get("pos_dropout", 0.0),
    ).to(device).eval()
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    rows_injury, rows_fusion, rows_allpatch = [], [], []
    bad_files_global = []

    case_dirs = list_case_dirs(args.tiles_root)
    print(f"[info] total cases: {len(case_dirs)}")

    for ci, case_dir in enumerate(case_dirs, 1):
        case = os.path.basename(case_dir.rstrip("/\\"))
        print(f"\n==== [{ci}/{len(case_dirs)}] {case} ====")
        files = list_images_recursively(case_dir)
        if not files:
            print(f"[warn] no images in {case_dir}, skip."); continue

        ds = ImageFilesDataset(
            files, image_size=args.image_size,
            use_hsv_filter=args.hsv_filter,
            min_sat=args.min_sat, max_val=args.max_val
        )
        dl = DataLoader(
            ds, batch_size=args.batch, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.pin_memory,
            persistent_workers=(args.num_workers > 0),
            collate_fn=safe_collate
        )

        prob_pos_list, cls_list, patch_rows = [], [], []

        with torch.no_grad():
            for batch in dl:
                if batch is None:  # 这一批全坏
                    continue
                imgs, paths = batch
                imgs = imgs.float().to(device, non_blocking=True)

                # ---------- TTA ----------
                if args.tta == "flips4":
                    tta_inputs = [imgs,
                                  torch.flip(imgs, dims=[-1]),
                                  torch.flip(imgs, dims=[-2]),
                                  torch.flip(imgs, dims=[-2, -1])]
                else:
                    tta_inputs = [imgs]

                probs_accum, cls_accum = [], []
                for t in tta_inputs:
                    with torch.amp.autocast("cuda", enabled=(args.amp and torch.cuda.is_available())):
                        logits = model(t)
                        # 温度缩放：logits / T 再 softmax
                        T = max(args.temp, 1e-6)
                        probs  = F.softmax(logits / T, dim=1)
                        cls    = get_cls_from_seq(model, t)
                    probs_accum.append(probs)
                    cls_accum.append(cls)

                probs = torch.stack(probs_accum, 0).mean(0)
                cls   = torch.stack(cls_accum,   0).mean(0)
                p = probs[:, pos_idx].cpu().numpy()

                prob_pos_list.extend(p.tolist())
                cls_list.extend(cls.cpu().numpy().tolist())

                for i, path in enumerate(paths):
                    x, y = parse_xy_from_name(path)
                    patch_rows.append({
                        "case": case, "path": path, "x": x, "y": y,
                        "prob_lesion": float(p[i]),
                        "pred": int(p[i] >= args.thr),
                        "thr": args.thr
                    })

        # 统计信息
        injury, fused = aggregate_case(
            prob_pos_list, cls_list,
            topk_ratio=args.topk_ratio, thr=args.thr,
            topk_gamma=args.topk_gamma, mean_trim=args.mean_trim,
            area_strict=args.area_strict
        )

        case_out = os.path.join(run_root, case)
        os.makedirs(case_out, exist_ok=True)

        # 每 case 的 injury
        pd.DataFrame([{"case": case, **injury}]).to_csv(
            os.path.join(case_out, f"{case}_injury.csv"), index=False)

        # 每 case 的 CLS 融合特征
        if fused is not None:
            feat = np.asarray(fused, dtype=float).ravel()
            cols = [f"feat_{i+1:04d}" for i in range(feat.size)]
            pd.DataFrame([[case] + feat.tolist()],
                         columns=["case"] + cols).to_csv(
                os.path.join(case_out, f"{case}_fusion.csv"), index=False)

        # 每 case 的 patch
        if args.dump_patch_csv and patch_rows:
            pd.DataFrame(patch_rows).to_csv(
                os.path.join(case_out, f"{case}_patches.csv"), index=False)

        print(f"[ok] {case}: n={injury['n_patches']} "
              f"mean={injury['mean_prob']:.4f} "
              f"topk_mean={injury['topk_mean']:.4f} "
              f"area_prop={injury['area_prop']:.4f}")

        rows_injury.append({"case": case, **injury})
        if fused is not None:
            out = {"case": case}
            for i, v in enumerate(np.asarray(fused).ravel().tolist(), 1):
                out[f"feat_{i:04d}"] = v
            rows_fusion.append(out)
        if args.dump_all_patches and patch_rows:
            rows_allpatch.extend(patch_rows)

    # 汇总输出
    pd.DataFrame(rows_injury).to_csv(os.path.join(run_root, "injury_scores_case.csv"), index=False)
    if rows_fusion:
        pd.DataFrame(rows_fusion).to_csv(os.path.join(run_root, "fusion_features_case.csv"), index=False)
    if args.dump_all_patches and rows_allpatch:
        pd.DataFrame(rows_allpatch).to_csv(os.path.join(run_root, "all_patches.csv"), index=False)

    print("\n[ALL] saved to:", run_root)

if __name__ == "__main__":
    main()
