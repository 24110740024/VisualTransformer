# train_finetune.py
# -*- coding: utf-8 -*-
import os, time, argparse
import numpy as np
import torch, torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from utils import set_seed, get_loaders, compute_class_weights, save_json
from model import ViT

def parse_args():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--train_dir", default="data224/train")
    ap.add_argument("--val_dir",   default="data224/val")
    ap.add_argument("--test_dir",  default="data224/test")
    ap.add_argument("--img_size",  type=int, default=224)
    # 训练
    ap.add_argument("--out_dir",   default="./checkpoints")
    ap.add_argument("--epochs",    type=int, default=50)
    ap.add_argument("--bs",        type=int, default=32)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers",  type=int, default=4)
    ap.add_argument("--seed",         type=int, default=2025)
    ap.add_argument("--weighted_sampler", action="store_true", help="按类别频次做采样")
    ap.add_argument("--class_weight", action="store_true", help="对 CE Loss 加类别权重")
    # ViT
    ap.add_argument("--in_channels", type=int, default=3)
    ap.add_argument("--patch_size",  type=int, default=16)
    ap.add_argument("--embed_dim",   type=int, default=768)
    ap.add_argument("--depth",       type=int, default=12)
    ap.add_argument("--heads",       type=int, default=12)
    ap.add_argument("--mlp_dim",     type=int, default=3072)
    ap.add_argument("--dropout",     type=float, default=0.1)
    ap.add_argument("--pos_dropout", type=float, default=0.0)
    # 早停
    ap.add_argument("--patience",    type=int, default=10, help="早停容忍 epoch 数（按 val_acc）")
    # ===== 新增：微调/冻结 =====
    ap.add_argument("--init_ckpt", type=str, default="", help="初始化权重 .pth（可选，支持 {'model':state_dict} 或纯 state_dict）")
    ap.add_argument("--freeze_stem", action="store_true", help="冻结 patch embedding / stem")
    ap.add_argument("--freeze_n_blocks", type=int, default=0, help="冻结前 N 个 transformer blocks（0 表示不冻结）")
    return ap.parse_args()

def freeze_parameters(model: nn.Module, freeze_stem: bool, freeze_n_blocks: int):
    """按名字粗略冻结 ViT 的部分参数（兼容常见命名：patch_embed / blocks.X / cls_token 等）"""
    total, frozen = 0, 0
    for name, p in model.named_parameters():
        need_freeze = False
        if freeze_stem and ("patch_embed" in name or "pos_embed" in name or "cls_token" in name):
            need_freeze = True
        if freeze_n_blocks > 0:
            # 常见命名：blocks.0, blocks.1, ...
            for i in range(freeze_n_blocks):
                if f"blocks.{i}." in name:
                    need_freeze = True
                    break
        if need_freeze:
            p.requires_grad = False
            frozen += p.numel()
        total += p.numel()
    print(f"[Freeze] frozen {frozen/1e6:.2f}M / {total/1e6:.2f}M params "
          f"(stem={'ON' if freeze_stem else 'OFF'}, first {freeze_n_blocks} blocks)")

def load_init_weights(model: nn.Module, ckpt_path: str):
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        print(f"[WARN] init_ckpt not found: {ckpt_path}")
        return
    print(f"[Init] Loading weights from: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    # 兼容两种保存：纯 state_dict 或 包含 {"model": state_dict}
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Init] loaded with strict=False, missing={len(missing)}, unexpected={len(unexpected)}")

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()

    # Data
    train_loader, val_loader, test_loader = get_loaders(
        args.train_dir, args.val_dir, args.test_dir,
        batch_size=args.bs, num_workers=args.num_workers,
        img_size=args.img_size, weighted_sampler=args.weighted_sampler
    )
    class_names = train_loader.dataset.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Model
    model = ViT(
        in_channels=args.in_channels,
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        pos_dropout=args.pos_dropout
    ).to(device)

    # === 新增：加载初始化权重 & 冻结部分层 ===
    load_init_weights(model, args.init_ckpt)
    if args.freeze_stem or args.freeze_n_blocks > 0:
        freeze_parameters(model, args.freeze_stem, args.freeze_n_blocks)

    # Loss
    if args.class_weight:
        cw = compute_class_weights(args.train_dir, num_classes).to(device)
        print("Class weights:", cw.tolist())
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_acc, best_epoch = 0.0, -1
    best_ckpt = os.path.join(args.out_dir, "vit_best.pth")

    # ========== Training ==========
    no_improve = 0
    for epoch in range(1, args.epochs+1):
        # ---- Train ----
        model.train()
        t0 = time.time()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"):
            x, y = x.to(device, non_blocking=True), y.long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            tr_loss += loss.item() * y.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += y.size(0)
        scheduler.step()
        tr_loss /= max(tr_total, 1)
        tr_acc = tr_correct / max(tr_total, 1)
        print(f"Train  loss={tr_loss:.4f} acc={tr_acc:.4f} time={time.time()-t0:.1f}s")

        # ---- Valid ----
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        va_probs, va_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Valid"):
                x, y = x.to(device, non_blocking=True), y.long().to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = model(x)
                    loss = criterion(logits, y)
                va_loss += loss.item() * y.size(0)
                pred = logits.argmax(1)
                va_correct += (pred == y).sum().item()
                va_total += y.size(0)
                if num_classes == 2:
                    va_probs += logits.softmax(1)[:, 1].detach().cpu().tolist()
                    va_labels += y.detach().cpu().tolist()
        va_loss /= max(va_total, 1)
        va_acc = va_correct / max(va_total, 1)
        if num_classes == 2 and len(set(va_labels)) > 1:
            va_auc = roc_auc_score(va_labels, va_probs)
            print(f"Valid  loss={va_loss:.4f} acc={va_acc:.4f} auc={va_auc:.4f}")
        else:
            print(f"Valid  loss={va_loss:.4f} acc={va_acc:.4f}")

        # 早停 & 保存最优
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": va_acc,
                "args": vars(args),
                "classes": class_names
            }, best_ckpt)
            print(f"[SAVE] best -> {best_ckpt} (val_acc={va_acc:.4f})")
            meta = {"best_epoch": epoch, "val_acc": va_acc, "classes": class_names, "args": vars(args)}
            save_json(meta, os.path.join(args.out_dir, "vit_best.meta.json"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[EARLY STOP] patience {args.patience} reached at epoch {epoch}.")
                break

    # ========== Test with best ckpt ==========
    print("\n[TEST] Loading best ckpt...")
    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"]); model.to(device); model.eval()

    te_loss, te_correct, te_total = 0.0, 0, 0
    te_probs, te_labels, te_preds = [], [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test"):
            x, y = x.to(device, non_blocking=True), y.long().to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            te_loss += loss.item() * y.size(0)
            pred = logits.argmax(1)
            te_correct += (pred == y).sum().item()
            te_total += y.size(0)
            te_preds  += pred.detach().cpu().tolist()
            te_labels += y.detach().cpu().tolist()
            if num_classes == 2:
                te_probs += logits.softmax(1)[:, 1].detach().cpu().tolist()

    te_loss /= max(te_total, 1)
    te_acc = te_correct / max(te_total, 1)
    print(f"Test   loss={te_loss:.4f} acc={te_acc:.4f}")
    if num_classes == 2 and len(set(te_labels)) > 1:
        te_auc = roc_auc_score(te_labels, te_probs); print(f"Test   auc={te_auc:.4f}")

    print("Confusion matrix:\n", confusion_matrix(te_labels, te_preds, labels=list(range(num_classes))))
    print("Classification report:\n", classification_report(te_labels, te_preds, target_names=class_names))

if __name__ == "__main__":
    main()
