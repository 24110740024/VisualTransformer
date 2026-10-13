"""Explicit model/optimizer identity for the one prespecified ImageNet candidate."""
from pathlib import Path
import hashlib
import json
import os
import torch
from torch import nn
import torchvision
from torchvision.models import vit_b_16

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
ROOT = NEW.parent
V2 = NEW / 'original_exports_v2'
FORMAT_VERSION = 106
INITIALIZATION = 'torchvision_ViT_B_16_IMAGENET1K_V1'
MODEL_NAME = 'torchvision.vit_b_16'
WEIGHT_SHA256 = 'c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9'
WEIGHT_PATH = Path(os.environ.get('VIT_IMAGENET_WEIGHTS', str(Path(torch.hub.get_dir())/'checkpoints/vit_b_16-c867db91.pth')))
ARCHITECTURE = {
    'model': MODEL_NAME, 'img_size': 224, 'patch_size': 16, 'hidden_dim': 768,
    'num_layers': 12, 'num_heads': 12, 'mlp_dim': 3072,
    'dropout': 0.0, 'attention_dropout': 0.0, 'layer_norm_eps': 1e-6,
    'num_classes': 2, 'classifier_initialization': 'torch.nn.Linear default after pretrained loading',
}
PARAMETERS = dict(epochs=20, bs=32, lr=1e-5, head_lr=1e-4, weight_decay=.05,
                  patience=6, num_workers=4, weighted_sampler=True, class_weight=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_weights():
    require(WEIGHT_PATH.is_file() and sha(WEIGHT_PATH) == WEIGHT_SHA256, 'Set VIT_IMAGENET_WEIGHTS to the official SHA-verified ImageNet file (see docs/REPRODUCING.md)')
    return {'path': str(WEIGHT_PATH), 'sha256': WEIGHT_SHA256,
            'weights_enum': 'ViT_B_16_Weights.IMAGENET1K_V1',
            'source_url': 'https://download.pytorch.org/models/vit_b_16-c867db91.pth'}


def create_model(pretrained=False):
    model=vit_b_16(weights=None, dropout=0.0, attention_dropout=0.0)
    if pretrained:
        verify_weights()
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu', weights_only=True), strict=True)
    model.heads.head=nn.Linear(model.heads.head.in_features,2)
    require(model.image_size == 224 and model.patch_size == 16 and model.hidden_dim == 768,
            'Unexpected torchvision model geometry')
    require(len(model.encoder.layers) == 12 and model.encoder.ln.eps == 1e-6,
            'Unexpected torchvision encoder defaults')
    require(all(p.requires_grad for p in model.parameters()), 'Candidate must finetune all parameters')
    return model


def build_optimizer(model, backbone_lr=1e-5, head_lr=1e-4, weight_decay=.05):
    head=list(model.heads.parameters())
    head_ids={id(p) for p in head}
    backbone=[p for p in model.parameters() if id(p) not in head_ids]
    require(head and backbone and len(head_ids) == len(head), 'Invalid optimizer parameter groups')
    require({id(p) for p in backbone}.isdisjoint(head_ids), 'Optimizer groups overlap')
    require(len(backbone)+len(head) == len(list(model.parameters())), 'Optimizer omitted parameters')
    return torch.optim.AdamW([{'params':backbone,'lr':backbone_lr,'group_name':'backbone'},
                             {'params':head,'lr':head_lr,'group_name':'head'}], weight_decay=weight_decay)


def training_code_hashes():
    return {p.name:sha(p) for p in sorted(HERE.glob('*.py'))}


def enforce_settings(args):
    for key,value in PARAMETERS.items():
        require(getattr(args,key)==value, f'Fixed v6 setting differs: {key}')
    require(args.device=='cuda', 'Production recipe requires CUDA FP16; use CPU tests for software verification')
    require(args.seed in (42,43,44), 'Fixed seeds are 42,43,44')
    require(args.train_manifest.resolve()!=args.val_manifest.resolve(),'Train/val must differ')
    prefix=args.train_manifest.stem.removesuffix('_train')
    require(prefix in ('a6','a5_fold1','a5_fold2','a5_fold3'),'Use one of the published data roles')
    require(args.train_manifest.name==prefix+'_train.csv' and args.val_manifest.name==prefix+'_val.csv','Mismatched role filenames')
    require(prefix!='a6' or args.seed==42,'A6 uses seed42')
    from data import check_published_manifest
    check_published_manifest(args.train_manifest);check_published_manifest(args.val_manifest)


def validate_checkpoint_identity(checkpoint):
    require(checkpoint.get('format_version')==106, 'Use a portable-v6 checkpoint. Original format6 checkpoints require the archived environment, not silent conversion.')
    require(checkpoint.get('classes')==['Lesion','Normal'] and checkpoint.get('positive_class_index')==0,'Wrong class convention')
    require(checkpoint['epoch']==checkpoint['best_epoch'],'Use validation-selected best.pth')
    config=checkpoint['configuration']
    require(config['architecture']==ARCHITECTURE and config['initialization']==INITIALIZATION,'Wrong model identity')
    require(hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==checkpoint['config_sha256'],'Configuration hash mismatch')
    require(config['code_sha256']==training_code_hashes(),'Portable source files changed since training')
    for k,v in PARAMETERS.items():require(config[k]==v,f'Fixed recipe differs: {k}')
    require(config['protocol_sha256']==sha(HERE/'EXECUTION_PROTOCOL.json'),'Portable protocol changed')
    return config


