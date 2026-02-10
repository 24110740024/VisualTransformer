# utils.py
import os, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from dataset import make_imagefolder

def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loaders(train_dir, val_dir, test_dir, batch_size, num_workers=4, img_size=224,
                weighted_sampler: bool=False):
    train_set = make_imagefolder(train_dir, train=True,  img_size=img_size)
    val_set   = make_imagefolder(val_dir,   train=False, img_size=img_size)
    test_set  = make_imagefolder(test_dir,  train=False, img_size=img_size)

    if weighted_sampler:
        # 按类别频次构造采样权重，缓解类不平衡
        labels = [y for _, y in train_set.samples]
        class_count = np.bincount(labels)
        class_weight = 1. / np.maximum(class_count, 1)
        sample_weight = [class_weight[y] for y in labels]
        sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

def compute_class_weights(train_dir, num_classes):
    # 计算 CrossEntropyLoss 的 class_weight
    ds = make_imagefolder(train_dir, train=True)
    labels = [y for _, y in ds.samples]
    count = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weight = count.sum() / np.maximum(count, 1.0)
    return torch.tensor(weight, dtype=torch.float32)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def evaluate_preds(y_true, y_pred, y_prob=None, class_names=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names) if class_names else 2)))
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4) if class_names else ""
    auc = None
    if y_prob is not None and len(set(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
    return cm, rep, auc
