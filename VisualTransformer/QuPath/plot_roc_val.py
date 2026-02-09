# -*- coding: utf-8 -*-
# tools/plot_roc_val.py — run ViT on a folder (Lesion/Normal) and plot ROC
import os, argparse, numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)

try:
    from model import ViT
except Exception:
    import sys
    sys.path.append(os.path.dirname(__file__) or ".")
    from model import ViT

IMG_EXTS = (".png",".jpg",".jpeg",".tif",".tiff",".bmp")

def list_files_with_label(root):
    rows=[]
    for cls, y in [("Lesion",1), ("Normal",0)]:
        d = os.path.join(root, cls)
        if not os.path.isdir(d): continue
        for r,_,fs in os.walk(d):
            for f in fs:
                if f.lower().endswith(IMG_EXTS):
                    rows.append((os.path.join(r,f), y))
    rows.sort()
    return rows

class PatchDS(Dataset):
    def __init__(self, items, image_size=224):
        self.items=items; self.image_size=image_size
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None: raise RuntimeError(f"bad image: {p}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (img.shape[0], img.shape[1]) != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255.0
        img[...,0]=(img[...,0]-0.485)/0.229
        img[...,1]=(img[...,1]-0.456)/0.224
        img[...,2]=(img[...,2]-0.406)/0.225
        img = np.transpose(img,(2,0,1))
        return torch.from_numpy(img), np.int64(y), p

def load_model(ckpt_path, image_size=224, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    margs = ckpt.get("args", {})
    classes = ckpt.get("classes", ["Lesion","Normal"])
    num_classes = len(classes)
    pos_idx = classes.index("Lesion") if "Lesion" in classes else 0
    model = ViT(
        in_channels=margs.get("in_channels",3),
        img_size=margs.get("img_size",image_size),
        patch_size=margs.get("patch_size",16),
        num_classes=num_classes,
        embed_dim=margs.get("embed_dim",768),
        depth=margs.get("depth",12),
        heads=margs.get("heads",12),
        mlp_dim=margs.get("mlp_dim",3072),
        dropout=margs.get("dropout",0.1),
        pos_dropout=margs.get("pos_dropout",0.0),
    ).to(device).eval()
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return model, pos_idx

def bootstrap_auc_ci(y_true, y_score, B=1000, alpha=0.95):
    n=len(y_true); aucs=[]
    idx = np.arange(n)
    for _ in range(B):
        samp = rng.choice(idx, size=n, replace=True)
        try:
            aucs.append(roc_auc_score(y_true[samp], y_score[samp]))
        except Exception:
            continue
    if not aucs:
        return float("nan"), (float("nan"), float("nan"))
    aucs = np.array(aucs)
    lo = np.quantile(aucs, (1-alpha)/2)
    hi = np.quantile(aucs, 1-(1-alpha)/2)
    return float(np.mean(aucs)), (float(lo), float(hi))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help=r'folder with "Lesion/Normal" subfolders')
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--title", default="ROC curve")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--bootstrap", type=int, default=1000, help="0 to disable")
    args=ap.parse_args()

    items = list_files_with_label(args.data_dir)
    assert items, f"No images found under {args.data_dir}"
    ds = PatchDS(items, image_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.num_workers, pin_memory=args.pin_memory,
                    persistent_workers=(args.num_workers>0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, pos_idx = load_model(args.ckpt, image_size=args.img_size, device=device)

    ys, ps, paths = [], [], []
    with torch.no_grad():
        for x, y, pths in dl:
            x = x.float().to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(args.amp and torch.cuda.is_available())):
                prob = F.softmax(model(x), dim=1)[:, pos_idx].detach().cpu().numpy()
            ys.append(y.numpy()); ps.append(prob); paths.extend(pths)
    y_true = np.concatenate(ys); y_score = np.concatenate(ps)

    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc  = roc_auc_score(y_true, y_score)

    # 你登：取最大 Youden J = TPR - FPR 的阈值，给一个“代表性”Acc
    j = np.argmax(tpr - fpr)
    tau = thr[j]
    y_pred = (y_score >= tau).astype(int)
    acc = accuracy_score(y_true, y_pred)

    # 95% CI（可选）
    if args.bootstrap and args.bootstrap > 0:
        auc_b, (lo, hi) = bootstrap_auc_ci(y_true, y_score, B=args.bootstrap)
        auc_text = f"AUC = {auc:.3f}  (95% CI {lo:.3f}–{hi:.3f})"
    else:
        auc_text = f"AUC = {auc:.3f}"

    # 出图
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, lw=2, label=f"ViT ({auc_text})")
    plt.plot([0,1],[0,1],"k--", lw=1, label="Chance (AUC=0.5)")
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(args.title)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    plt.savefig(args.out_png, dpi=args.dpi)

    # 保存逐样本 CSV（方便溯源）
    pd.DataFrame({"path":paths, "label":y_true, "prob_lesion":y_score}).to_csv(args.out_csv, index=False)

    print(f"[OK] ROC saved: {args.out_png}")
    print(f"     AUC={auc:.4f};  Acc@Youden(tau={tau:.3f})={acc:.4f}")
    print(f"[OK] CSV saved: {args.out_csv}")

if __name__ == "__main__":
    main()
