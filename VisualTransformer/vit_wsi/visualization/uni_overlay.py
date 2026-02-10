# -*- coding: utf-8 -*-
# Heatmap overlay for vit_wsi — FP-suppressed & smooth grid rendering

import os, re, argparse
import numpy as np
import pandas as pd
import cv2

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# ---------- filename -> (x,y) ----------
def parse_xy_from_name(p: str):
    fn = os.path.splitext(os.path.basename(p))[0]
    m = re.search(r'[xX](\d+)[^0-9]+[yY](\d+)', fn)
    if m: return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d+)[_-](\d+)$', fn)
    if m: return int(m.group(1)), int(m.group(2))
    nums = re.findall(r'\d+', fn)
    if len(nums) >= 2: return int(nums[-2]), int(nums[-1])
    return -1, -1

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

# ---------- stride 推断（稳健吸附常见步长） ----------
def deduce_stride(xs, hint=None):
    xs = sorted(set(int(v) for v in xs))
    if len(xs) <= 1: return hint or 224
    diffs = np.diff(xs)
    diffs = diffs[diffs > 0]
    if diffs.size == 0: return hint or 224
    med = float(np.median(diffs))
    cand = diffs[(diffs >= 0.5*med) & (diffs <= 2.0*med)]
    if cand.size == 0: cand = diffs
    est = float(np.median(cand))
    anchors = np.array([16,32,64,128,224,256,384,512,768,1024], dtype=float)
    stride = int(anchors[np.argmin(np.abs(anchors - est))])
    if hint and abs(hint - est) < abs(stride - est):
        stride = int(hint)
    return max(stride, 1)

# ---------- 加色条 ----------
def _add_colorbar(img, h=200, w=28, margin=24):
    bar = np.linspace(255, 0, h, dtype=np.uint8).reshape(h,1)
    bar = np.repeat(bar, w, axis=1)
    bar = cv2.applyColorMap(bar, cv2.COLORMAP_JET)
    H, W = img.shape[:2]
    y0 = (H - h)//2
    x0 = W - w - margin
    img[y0:y0+h, x0:x0+w] = bar
    for t,txt in [(0.0,"0.0"), (0.5,"0.5"), (1.0,"1.0")]:
        yy = int(y0 + (1.0-t)*(h-1))
        cv2.line(img, (x0+w, yy), (x0+w+8, yy), (0,0,0), 2)
        cv2.putText(img, txt, (x0+w+10, yy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

# ---------- 小连通域剔除（在网格尺度上） ----------
def remove_small_components(bw_grid: np.ndarray, min_tiles: int) -> np.ndarray:
    if min_tiles is None or min_tiles <= 1:
        return bw_grid.astype(np.uint8)
    bw = (bw_grid > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(bw, connectivity=4)
    keep = np.zeros_like(bw, dtype=np.uint8)
    for lab in range(1, num):
        area = int((labels == lab).sum())
        if area >= min_tiles:
            keep[labels == lab] = 1
    return keep

# ---------- 主渲染 ----------
def render_uni(
    df_case, out_dir,
    tile_size=512,
    alpha=0.70,
    gamma=1.35,
    blur_sigma=1.2,
    # 绝对刻度（固定 colormap 到[abs_min, abs_max] 概率）
    abs_min=0.0,
    abs_max=1.0,
    # 备选：分位刻度（若想改回片内归一化，可设置 vmin_q/vmax_q 并把 abs_* 置 None）
    vmin_q=None,
    vmax_q=None,
    # 附加抑制
    hard_thr=0.64,          # 低于此概率不着色
    mask_dilate=0,
    thr_contour=0.60,       # 只描边较高分区
    min_comp_tiles=4,       # 小连通域最少网格数；设 0 关闭
    add_colorbar=True,
    force_stride=256        # 与patches实际stride对齐即可
):
    os.makedirs(out_dir, exist_ok=True)

    df = df_case.copy()
    for c in ["x","y","prob_lesion"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x","y","prob_lesion"])
    if len(df) == 0:
        print("[warn] empty case, skip."); return

    xs, ys = df['x'].values, df['y'].values
    stride_x = int(force_stride) if force_stride else deduce_stride(xs)
    stride_y = int(force_stride) if force_stride else deduce_stride(ys)

    min_x, min_y = float(xs.min()), float(ys.min())
    cols = int(np.round((float(xs.max()) - min_x) / max(1, stride_x))) + 1
    rows = int(np.round((float(ys.max()) - min_y) / max(1, stride_y))) + 1

    # 网格热度与掩膜
    heat = np.zeros((rows, cols), dtype=np.float32)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for _, r in df.iterrows():
        col = int(np.round((float(r.x) - min_x) / max(1, stride_x)))
        row = int(np.round((float(r.y) - min_y) / max(1, stride_y)))
        if 0 <= row < rows and 0 <= col < cols:
            p = float(r.prob_lesion)
            if p > heat[row, col]:  # 取最大
                heat[row, col] = p
            mask[row, col] = 1

    # 硬阈值抠底：直接把低于 hard_thr 的网格置 0（不着色）
    if hard_thr is not None:
        heat = np.where(heat >= float(hard_thr), heat, 0.0)

    # 小连通域剔除：在“>= hard_thr”的区域上连通域过滤
    if hard_thr is not None and min_comp_tiles and min_comp_tiles > 0:
        bw_high = (heat >= float(hard_thr)).astype(np.uint8) * mask
        keep = remove_small_components(bw_high, int(min_comp_tiles))
        heat = heat * keep

    # 掩膜膨胀（通常为 0，避免扩张红面）
    if mask_dilate and mask_dilate > 0:
        k = 2*mask_dilate+1
        kernel = np.ones((k,k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    # 归一化：优先用绝对刻度，否则退回分位刻度
    valid = heat[mask.astype(bool)]
    if valid.size == 0:
        print("[warn] mask empty after suppression, skip."); return

    if abs_min is not None and abs_max is not None:
        lo, hi = float(abs_min), float(abs_max)
    else:
        lo = float(np.quantile(valid, np.clip(vmin_q, 0, 0.98))) if vmin_q is not None else float(valid.min())
        hi = float(np.quantile(valid, np.clip(vmax_q, 0.02, 1.0))) if vmax_q is not None else float(valid.max())
        if hi <= lo:  # 兜底
            lo, hi = float(valid.min()), float(valid.max()+1e-6)

    heat_n = np.clip((heat - lo) / (hi - lo + 1e-12), 0, 1)

    # γ 抑制：压中低分，进一步缩小红面扩张
    if gamma and gamma != 1.0:
        heat_n = np.power(heat_n, float(gamma))

    # 输出尺寸（每格像素边长）
    cell = int(max(32, min(224, int(tile_size))))
    H, W = rows*cell, cols*cell

    # 先线性插值再上色（避免马赛克）
    heat_up_f = cv2.resize(heat_n, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_up   = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    heat_u8 = np.clip(heat_up_f*255, 0, 255).astype(np.uint8)
    cmap    = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    if blur_sigma and blur_sigma > 0:
        k = int(blur_sigma*4+1) | 1
        cmap = cv2.GaussianBlur(cmap, (k, k), blur_sigma)

    # 叠加到白底，仅在有 tile 的位置
    bg  = np.ones((H, W, 3), dtype=np.uint8)*255
    out = bg.copy()
    m   = mask_up.astype(bool)
    out[m] = cv2.addWeighted(cmap[m], float(alpha), bg[m], 1-float(alpha), 0)

    # 高概率等高线（相对于“概率域”的阈）
    if thr_contour is not None:
        thr = float(thr_contour)
        bw  = (heat_up_f >= thr).astype(np.uint8)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255,0,0), 2)

    # 色条
    if add_colorbar:
        _add_colorbar(out)

    base = (str(df['slide_id'].iloc[0]) if 'slide_id' in df.columns
            else (str(df['case'].iloc[0]) if 'case' in df.columns
                  else os.path.basename(os.path.dirname(df['path'].iloc[0]))))
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), out)
    cv2.imwrite(os.path.join(out_dir, f"{base}_heat.png"), cmap)
    np.save(os.path.join(out_dir, f"{base}_heat.npy"), heat_n)
    print(f"[ok] {base}: grid={rows}x{cols} stride=({stride_x},{stride_y}) -> {os.path.join(out_dir, f'{base}_overlay.png')}")

# ---------- CSV 载入 ----------
def load_probs_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "prob_lesion" not in df.columns:
        for c in ["prob","score","p_pos","p_lesion"]:
            if c in df.columns:
                df = df.rename(columns={c:"prob_lesion"}); break
    if "slide_id" not in df.columns:
        if "case" in df.columns:
            df["slide_id"] = df["case"]
        elif "path" in df.columns:
            df["slide_id"] = df["path"].apply(lambda p: os.path.basename(os.path.dirname(p)) if isinstance(p,str) else "UNK")
        else:
            df["slide_id"] = "UNK"
    if "x" not in df.columns or "y" not in df.columns:
        if "path" in df.columns:
            xy = df["path"].apply(lambda p: parse_xy_from_name(p) if isinstance(p,str) else (-1,-1))
            df["x"] = [t[0] for t in xy]; df["y"] = [t[1] for t in xy]
        else:
            raise ValueError("CSV missing x,y and path; cannot infer coordinates.")
    return df.dropna(subset=["prob_lesion"])

def run_from_csv(csv_path, out_dir, **kw):
    os.makedirs(out_dir, exist_ok=True)
    df = load_probs_csv(csv_path)
    for _, df_case in df.groupby("slide_id"):
        render_uni(df_case, out_dir, **kw)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help=r"preds_dual\...\all_patches.csv 或 单个 case 的 *_patches.csv")
    ap.add_argument("--out_dir", default="./vis_uni")

    # 渲染/抑制参数（与 512 tiles 匹配的保守默认）
    ap.add_argument("--tile_size",   type=int,   default=512)
    ap.add_argument("--alpha",       type=float, default=0.70)
    ap.add_argument("--gamma",       type=float, default=1.35)
    ap.add_argument("--blur_sigma",  type=float, default=1.2)

    # 绝对刻度（默认启用：固定到[0,1]）
    ap.add_argument("--abs_min",     type=float, default=0.0)
    ap.add_argument("--abs_max",     type=float, default=1.0)

    # 退路：分位刻度（若要改回片内归一化，把 abs_* 设为 None 并设置这两个）
    ap.add_argument("--vmin_q",      type=float, default=None)
    ap.add_argument("--vmax_q",      type=float, default=None)

    # 抑制参数
    ap.add_argument("--hard_thr",    type=float, default=0.64)
    ap.add_argument("--mask_dilate", type=int,   default=0)
    ap.add_argument("--thr_contour", type=float, default=0.60)
    ap.add_argument("--min_comp_tiles", type=int, default=4,
                    help="小连通域最小网格数，设 0 关闭")
    ap.add_argument("--add_colorbar", action="store_true")

    # 与patches实际步长对齐
    ap.add_argument("--force_stride", type=int, default=256)

    args = ap.parse_args()
    kw = dict(
        tile_size=args.tile_size,
        alpha=args.alpha,
        gamma=args.gamma,
        blur_sigma=args.blur_sigma,
        abs_min=args.abs_min,
        abs_max=args.abs_max,
        vmin_q=args.vmin_q,
        vmax_q=args.vmax_q,
        hard_thr=args.hard_thr,
        mask_dilate=args.mask_dilate,
        thr_contour=args.thr_contour,
        min_comp_tiles=args.min_comp_tiles,
        add_colorbar=args.add_colorbar,
        force_stride=args.force_stride
    )
    run_from_csv(args.csv, args.out_dir, **kw)

if __name__ == "__main__":
    main()
