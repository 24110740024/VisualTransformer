"""Explicit model/optimizer identity for the one prespecified ImageNet candidate."""
from pathlib import Path
import hashlib
import json
import torch
from torch import nn
import torchvision
from torchvision.models import vit_b_16

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
ROOT = NEW.parent
V2 = NEW / 'original_exports_v2'
FORMAT_VERSION = 6
INITIALIZATION = 'torchvision_ViT_B_16_IMAGENET1K_V1'
MODEL_NAME = 'torchvision.vit_b_16'
WEIGHT_SHA256 = 'c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9'
WEIGHT_PATH = NEW / 'pretrained_candidate_v3/weights/vit_b_16-c867db91.pth'
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
    record=read(NEW/'pretrained_candidate_v3/weights/WEIGHTS.json')
    require(record['sha256'] == WEIGHT_SHA256 and sha(WEIGHT_PATH) == WEIGHT_SHA256,
            'Official initial weights changed')
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
    files=[HERE/'train_candidate.py',Path(__file__),HERE/'candidate_guard.py',
           NEW/'train_manifest.py',HERE/'scale_transforms.py',HERE/'RUN_PROTOCOL.md',HERE/'sampling.py',HERE/'sampling_binding.py',HERE/'SAMPLING_FROZEN.json']
    files += sorted((HERE/'sampling').glob('*.csv'))
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def enforce_settings(args):
    for key,value in PARAMETERS.items():
        require(getattr(args,key) == value, f'Candidate setting differs from fixed plan: {key}')
    require(args.device == 'cuda', 'Production candidate explicitly requires CUDA')
    frozen=read(V2/'FROZEN.json')['manifests']
    prefixes=['a6','a5_fold1','a5_fold2','a5_fold3']
    matches=[prefix for prefix in prefixes if args.train_manifest.resolve() == Path(frozen[prefix+'_train']['path']).resolve()
             and args.val_manifest.resolve() == Path(frozen[prefix+'_val']['path']).resolve()]
    require(len(matches) == 1, 'Training must use exactly one frozen v2 train/validation pair')
    prefix=matches[0]
    require(args.seed in ((42,) if prefix == 'a6' else (42,43,44)), 'Unplanned candidate seed')
    for role,path in (('train',args.train_manifest),('val',args.val_manifest)):
        require(sha(path) == frozen[prefix+'_'+role]['sha256'], f'Frozen {role} manifest changed')
    expected_name='a6_seed42' if prefix == 'a6' else f'a5_seed{args.seed}_fold{prefix[-1]}'
    require(args.out_dir.resolve() == HERE/'runs'/expected_name, 'Candidate output is outside its planned run')
    return prefix


def validate_checkpoint_identity(checkpoint, require_best=True):
    require(checkpoint.get('format_version') == FORMAT_VERSION and checkpoint.get('initialization') == INITIALIZATION,
            'Not a pretrained-candidate v6 checkpoint')
    require(checkpoint.get('classes') == ['Lesion','Normal'] and checkpoint.get('positive_class_index') == 0,
            'Candidate class convention differs')
    if require_best:
        require(checkpoint['epoch'] == checkpoint['best_epoch'], 'Not the validation-selected best checkpoint')
    config=checkpoint['configuration']
    require(config.get('initialization') == INITIALIZATION and config.get('model') == MODEL_NAME
            and config.get('architecture') == ARCHITECTURE, 'Candidate architecture or initialization differs')
    for key,value in PARAMETERS.items():
        require(config.get(key) == value, f'Checkpoint training setting differs: {key}')
    require(config['pretrained_weights']['sha256'] == WEIGHT_SHA256, 'Checkpoint initial weight identity differs')
    digest=hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    require(digest == checkpoint['config_sha256'], 'Checkpoint configuration hash mismatch')
    require(config['code_sha256'] == training_code_hashes(), 'Candidate training code changed')
    require(config['protocol_sha256'] == sha(HERE/'EXECUTION_PROTOCOL.json'), 'Candidate protocol changed')
    from sampling_binding import binding_identity
    require(config.get('sampling')==binding_identity(config['train_manifest']), 'Checkpoint sampling binding differs')
    return config
